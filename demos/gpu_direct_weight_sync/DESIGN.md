# Direct Weight Sync Design: FSDP Trainer → TP Generator

## Problem Statement

Current architecture has fundamental bottlenecks:
```
Trainer → TorchStore → Generator
         (200 puts)    (1600 gets)

Total: ~1800 RPC round-trips for ~200 parameters
```

Goal: Reduce to **O(1) coordination + O(FSDP×TP) data transfers**

---

## Push vs Pull Analysis

### Option A: Push (Trainer → Generator)

```
Trainer FSDP ranks push directly to Generator TP ranks
```

**Flow:**
1. Trainer computes what each TP rank needs
2. Each FSDP rank sends to relevant TP ranks
3. Generator receives and assembles

**Problems:**
- Trainer must know TP topology (tight coupling)
- Trainer computes FSDP→TP slice mapping (complex)
- Many-to-many communication pattern
- Trainer blocked until all sends complete

### Option B: Pull (Generator ← Trainer)

```
Generator TP ranks pull from Trainer FSDP ranks
```

**Flow:**
1. Trainer signals "weights ready"
2. Each TP rank computes what slices it needs
3. Each TP rank pulls from relevant FSDP ranks
4. Generator assembles locally

**Advantages:**
- Generator already knows TP slicing logic (existing code)
- Clean separation - trainer just exposes data
- Natural parallelism - each TP rank pulls independently
- Trainer not blocked

### Decision: **PULL is Better**

Key insight: Generator ALREADY has the TP slicing logic in `_compute_all_tp_slices()`.
Moving this to trainer would duplicate code and couple components.

---

## FSDP → TP Transfer Matrix

Consider param shape `[4096, 4096]` with FSDP=2, TP=2:

### Column-Parallel Param (QKV projections)
```
FSDP shards rows, TP shards columns

FSDP0 has [0:2048, :]      FSDP1 has [2048:4096, :]
        ↓                           ↓
TP0 needs [:, 0:2048]      TP1 needs [:, 2048:4096]

Transfer matrix:
┌─────────────────────────────────────────┐
│           │ TP0 (cols 0:2048) │ TP1     │
├───────────┼───────────────────┼─────────┤
│ FSDP0     │ [0:2048, 0:2048]  │ [0:2048,│
│ (rows     │        ✓          │ 2048:]  │
│  0:2048)  │                   │    ✓    │
├───────────┼───────────────────┼─────────┤
│ FSDP1     │ [2048:, 0:2048]   │ [2048:, │
│ (rows     │        ✓          │ 2048:]  │
│  2048:)   │                   │    ✓    │
└─────────────────────────────────────────┘

Each TP rank pulls from ALL FSDP ranks (full rows, partial cols)
```

### Row-Parallel Param (output projections)
```
FSDP shards rows, TP also shards rows

FSDP0 has [0:2048, :]      FSDP1 has [2048:4096, :]
        ↓                           ↓
TP0 needs [0:2048, :]      TP1 needs [2048:4096, :]

Transfer matrix:
┌─────────────────────────────────────────┐
│           │ TP0 (rows 0:2048) │ TP1     │
├───────────┼───────────────────┼─────────┤
│ FSDP0     │ [0:2048, :]       │   ∅     │
│           │      ✓            │ (none)  │
├───────────┼───────────────────┼─────────┤
│ FSDP1     │      ∅            │[2048:,:]│
│           │   (none)          │    ✓    │
└─────────────────────────────────────────┘

Each TP rank pulls from ONLY the FSDP rank with matching rows
```

### Replicated Param (LayerNorm, biases)
```
All FSDP ranks have identical copy (or FSDP rank 0 has full)
All TP ranks need full copy

TP0 pulls from FSDP0 (or any)
TP1 pulls from FSDP0 (or any)
```

---

## Architecture Design

### Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                        Controller                            │
│  - Orchestrates training loop                               │
│  - Triggers weight sync                                     │
└─────────────┬───────────────────────────────┬───────────────┘
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│    Trainer Actor Mesh   │     │    Generator Actor          │
│  (FSDP_world_size ranks)│     │  + Worker Mesh (TP ranks)   │
│                         │     │                             │
│  ┌─────┐ ┌─────┐       │     │  ┌────────┐ ┌────────┐     │
│  │Rank0│ │Rank1│ ...   │     │  │TP Rank0│ │TP Rank1│ ... │
│  └──┬──┘ └──┬──┘       │     │  └────┬───┘ └────┬───┘     │
│     │       │           │     │       │          │         │
│     ▼       ▼           │     │       │          │         │
│  [shard0] [shard1]      │     │       │          │         │
└─────────────────────────┘     └───────┼──────────┼─────────┘
              │                         │          │
              │   WeightReadySignal     │          │
              ├─────────────────────────┤          │
              │                         │          │
              │◄────── pull_shard ──────┤          │
              │◄────── pull_shard ──────┼──────────┤
              │                         │          │
```

### Protocol

#### Step 1: Trainer Signals Ready
```python
@dataclass
class WeightReadySignal:
    version: int
    param_info: dict[str, ParamShardInfo]  # name → {global_shape, fsdp_dim, ...}
    trainer_handles: list[ActorHandle]      # One per FSDP rank
```

#### Step 2: Generator Computes Fetch Plan
```python
class FetchPlan:
    """What each TP rank needs to fetch from which FSDP ranks."""

    def compute(self, param_info: ParamShardInfo, tp_rank: int, tp_size: int):
        my_tp_slice = compute_tp_slice(param_info, tp_rank, tp_size)

        fetch_tasks = []
        for fsdp_rank, fsdp_slice in enumerate(param_info.fsdp_slices):
            intersection = compute_intersection(my_tp_slice, fsdp_slice)
            if intersection.is_nonempty():
                fetch_tasks.append(FetchTask(
                    fsdp_rank=fsdp_rank,
                    slice_in_fsdp_shard=intersection.localize_to(fsdp_slice),
                    slice_in_my_tensor=intersection.localize_to(my_tp_slice),
                ))
        return fetch_tasks
```

#### Step 3: Direct Pull
```python
# On Generator TP Worker
async def pull_weights(self, signal: WeightReadySignal):
    for name, info in signal.param_info.items():
        plan = FetchPlan().compute(info, self.tp_rank, self.tp_size)

        # Parallel fetch from all relevant FSDP ranks
        tasks = [
            signal.trainer_handles[task.fsdp_rank].get_shard.call(
                name, task.slice_in_fsdp_shard
            )
            for task in plan
        ]
        shards = await asyncio.gather(*tasks)

        # Assemble into my local tensor
        my_tensor = torch.empty(info.tp_local_shape, device='cuda')
        for task, shard in zip(plan, shards):
            my_tensor[task.slice_in_my_tensor] = shard

        self.model.load_weight(name, my_tensor)
```

#### Step 4: Trainer Exposes Shards
```python
# On Trainer FSDP Rank
@endpoint
async def get_shard(self, param_name: str, slice_spec: SliceSpec) -> torch.Tensor:
    """Return a slice of my local FSDP shard."""
    local_tensor = self.model.get_parameter(param_name)._local_tensor

    # Extract requested slice
    sliced = local_tensor[slice_spec.to_index()]

    # Return with CUDA IPC for same-node, or serialize for multi-node
    return sliced.contiguous()
```

---

## Communication Patterns

### Case 1: Single GPU (1×1)
```
Trainer (FSDP=1, 1 GPU)  →  Generator (TP=1, 1 GPU)

Signal: {param_info, trainer_handles=[rank0]}
Pull:   TP0 pulls all params from FSDP0

Transfers: 1 round-trip per param (can batch into 1 RPC)
```

### Case 2: FSDP Trainer → Single Generator (2×1)
```
Trainer (FSDP=2, 2 GPUs)  →  Generator (TP=1, 1 GPU)

Signal: {param_info, trainer_handles=[rank0, rank1]}
Pull:   TP0 pulls from FSDP0 and FSDP1, concatenates

Transfers: 2 pulls per param (parallel)
```

### Case 3: Single Trainer → TP Generator (1×2)
```
Trainer (FSDP=1, 1 GPU)  →  Generator (TP=2, 2 GPUs)

Signal: {param_info, trainer_handles=[rank0]}
Pull:   TP0 pulls [:, 0:half] from FSDP0
        TP1 pulls [:, half:] from FSDP0

Transfers: 2 pulls per param (parallel, smaller slices)
```

### Case 4: FSDP Trainer → TP Generator (2×2)
```
Trainer (FSDP=2, 2 GPUs)  →  Generator (TP=2, 2 GPUs)

Signal: {param_info, trainer_handles=[rank0, rank1]}
Pull:   For column-parallel params:
          TP0 pulls [:, 0:half] from both FSDP0 and FSDP1
          TP1 pulls [:, half:] from both FSDP0 and FSDP1
        For row-parallel params:
          TP0 pulls from FSDP0 only
          TP1 pulls from FSDP1 only

Transfers: 2-4 pulls per param depending on param type (parallel)
```

---

## Transport Layer

### Same-Node (CUDA IPC)
```python
class DirectTransfer:
    @staticmethod
    async def send_tensor(tensor: torch.Tensor) -> CudaIPCHandle:
        """Create IPC handle for cross-process GPU access."""
        return create_ipc_handle(tensor)

    @staticmethod
    async def recv_tensor(handle: CudaIPCHandle) -> torch.Tensor:
        """Reconstruct tensor from IPC handle."""
        return handle.reconstruct_tensor()
```

### Multi-Node (NCCL or RPC)
```python
class DirectTransfer:
    @staticmethod
    async def send_tensor(tensor: torch.Tensor) -> bytes:
        """Serialize tensor for network transfer."""
        # Use NCCL send/recv for best bandwidth
        # Fall back to RPC serialization if NCCL not available
        pass
```

### Automatic Selection
```python
def get_transfer_method(src_node: str, dst_node: str):
    if src_node == dst_node:
        return CudaIPCTransfer()  # Same node: use CUDA IPC
    elif nccl_available():
        return NCCLTransfer()      # Cross-node: use NCCL
    else:
        return RPCTransfer()       # Fallback: use Monarch RPC
```

---

## Implementation Plan

### Phase 1: Batched Pull via Existing TorchStore (Quick Win)
**Goal**: Reduce RPC overhead without architectural changes

1. Add `get_batch()` to TorchStore API
2. Modify `_WeightFetcher` to batch all params in one call
3. Expected: 10-20x improvement

```python
# Before: 200 separate calls
for name in param_names:
    param = await ts.get(key)

# After: 1 batched call
params = await ts.get_batch(keys)
```

### Phase 2: Direct Signal + Pull (Major Win)
**Goal**: Eliminate TorchStore hop

1. Add `WeightReadySignal` dataclass
2. Add `get_shard()` endpoint to Trainer
3. Add `pull_weights_direct()` to Generator
4. Modify Controller to pass trainer handles to generator
5. Expected: 5-10x improvement over Phase 1

### Phase 3: CUDA IPC Integration (Maximum Performance)
**Goal**: GPU-direct transfers for same-node

1. Integrate CudaIPC into direct pull
2. Trainer returns IPC handles instead of tensor copies
3. Generator reconstructs without CPU involvement
4. Expected: Near NVLink bandwidth (~600 GB/s)

### Phase 4: NCCL for Multi-Node (Production Ready)
**Goal**: Optimal cross-node performance

1. Establish NCCL communicator between Trainer and Generator meshes
2. Use NCCL send/recv for cross-node transfers
3. Hybrid: CUDA IPC for same-node, NCCL for cross-node
4. Expected: Near network bandwidth (100-400 Gbps)

---

## Performance Projections

| Config | Current | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|---------|
| 1×1 (4B model) | ~78s | ~15s | ~5s | ~2s |
| 2×1 (4B model) | ~90s | ~20s | ~8s | ~3s |
| 1×2 (4B model) | ~85s | ~18s | ~6s | ~2s |
| 2×2 (4B model) | ~100s | ~25s | ~10s | ~4s |

### Theoretical Limits
- 8GB model weights
- NVLink: 600 GB/s → ~13ms
- PCIe 4.0: 32 GB/s → ~250ms
- 100GbE: 12.5 GB/s → ~640ms

Overhead (RPC, slicing, assembly) adds ~2-5x to theoretical.

---

## Key Files to Modify

| File | Changes |
|------|---------|
| `forge/types.py` | Add `WeightReadySignal` dataclass |
| `forge/actors/trainer/titan.py` | Add `get_shard()` endpoint, `signal_weights_ready()` |
| `forge/actors/vllm/v1/generator.py` | Add `pull_weights_direct()` |
| `forge/actors/vllm/v1/forge_executor.py` | Add `receive_weights_direct()` |
| `forge/controller/base.py` | Pass trainer handles to generator |
| `torchstore/transport/cuda_ipc.py` | Already done! |

---

## Questions to Resolve

1. **Batching granularity**: One RPC per param or one RPC for all params?
   - Trade-off: Latency vs memory (large single transfer may OOM)
   - Recommendation: Batch by size threshold (e.g., 1GB batches)

2. **Sync point**: When does trainer block?
   - Option A: Trainer waits until generator confirms receipt
   - Option B: Trainer signals and continues (generator pulls async)
   - Recommendation: Option B (non-blocking signal)

3. **Version management**: What if trainer steps while generator pulls?
   - Need to keep old version alive until pull completes
   - Recommendation: Reference counting or double-buffering

4. **Failure handling**: What if pull fails mid-transfer?
   - Recommendation: Retry with exponential backoff, fallback to TorchStore
