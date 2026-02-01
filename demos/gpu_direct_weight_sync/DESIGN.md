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

## Scale Requirements

### Target: 1000 GPU RL Training
```
Example configuration:
- Trainers: 256 GPUs (32 nodes × 8 GPUs, FSDP=256)
- Generators: 744 GPUs (93 generator groups, each TP=8)
- Model: 70B parameters (~140GB weights)
- Network: 400 Gbps RDMA between nodes
```

### On-Policy vs Off-Policy

| Aspect | On-Policy (PPO, GRPO) | Off-Policy (DQN, SAC) |
|--------|----------------------|----------------------|
| Version consistency | ALL generators MUST use same version | Can use stale weights |
| Sync requirement | Barrier after each step | Async, periodic updates |
| Latency sensitivity | Critical path | Background task |
| Version lifetime | Short (one step) | Long (many steps) |

**The design must support BOTH patterns efficiently.**

---

## Why Not Abandon TorchStore Entirely?

Pure direct pull has problems at scale:

### Problem 1: Fan-in Thundering Herd
```
With 744 generator GPUs pulling from 256 trainer GPUs:
- Each trainer serves ~3 concurrent requests
- All generators starting at once = network saturation
- Memory pressure on trainers
```

### Problem 2: Multi-Node Complexity
```
CUDA IPC only works within a node.
Cross-node needs NCCL or network transfer.
Direct pull requires knowing physical topology.
```

### Problem 3: Version Management
```
On-policy: All generators must use version N before step N+1
Off-policy: Multiple versions coexist, need garbage collection
Pure pull doesn't solve coordination.
```

### Solution: TorchStore as Coordination Layer

**Don't abandon TorchStore - evolve it:**

```
OLD: TorchStore = Data Storage + Coordination
NEW: TorchStore = Coordination + Metadata + Fallback Storage

Data path:  Trainer ──direct──► Generator (fast)
Control path: Trainer ──► TorchStore ──► Generator (lightweight)
```

---

## Revised Architecture: Hybrid Signal-Pull

### Key Insight

Separate the **control plane** (coordination) from the **data plane** (transfer):

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTROL PLANE (TorchStore)                   │
│  - Version registry (what versions exist)                       │
│  - Shard directory (who has what)                               │
│  - Sync barriers (on-policy coordination)                       │
│  - Reference counting (version lifetime)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ lightweight metadata
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PLANE (Direct Transfer)                 │
│  - Same-node: CUDA IPC                                          │
│  - Same-rack: NCCL over NVSwitch                               │
│  - Cross-rack: NCCL over RDMA                                  │
│  - Fallback: RPC serialization                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           WeightRegistry                                 │
│                    (Evolution of TorchStore Controller)                  │
│                                                                          │
│  • register_version(version, shard_locations)                           │
│  • get_fetch_plan(version, tp_rank, tp_size) → [(trainer, slice), ...]  │
│  • barrier_wait(version, participant_id)  # for on-policy               │
│  • barrier_release(version, participant_id)                              │
│  • get_latest_version() → version  # for off-policy                     │
│  • release_version(version)  # decrement refcount                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
┌─────────────────────────────┐             ┌─────────────────────────────┐
│    TrainerWeightServer      │             │    GeneratorWeightClient    │
│    (on each FSDP rank)      │             │    (on each TP rank)        │
│                             │             │                             │
│  • get_shard(param, slice)  │◄──────────│  • fetch_weights(version)   │
│  • get_shard_batch(params)  │   direct   │  • compute_fetch_plan()     │
│  • pin_version(version)     │  transfer  │  • assemble_tensors()       │
│  • unpin_version(version)   │             │  • notify_complete()        │
└─────────────────────────────┘             └─────────────────────────────┘
```

---

## Protocol: On-Policy Weight Sync

For algorithms requiring strict version consistency (PPO, GRPO):

```
Timeline:
─────────────────────────────────────────────────────────────────────►

Trainer:    [train step N] [register v.N] [barrier_wait]    [train step N+1]
                               │              │     ▲
                               │              │     │ all confirmed
                               ▼              ▼     │
Registry:              [v.N available] [track confirmations]
                               │
                               │ notify generators
                               ▼
Generator:         [fetch v.N] [pull shards] [apply] [confirm to registry]
```

### Detailed Steps

```python
# 1. Trainer finishes step, registers version
async def on_train_step_complete(trainer_mesh, version):
    # Each FSDP rank registers its shard locations
    shard_info = await trainer_mesh.get_shard_info.call()  # parallel

    # Register with WeightRegistry (single call)
    await registry.register_version(
        version=version,
        shard_info=shard_info,  # {param: {rank: (offset, shape)}}
        trainer_handles=trainer_mesh.get_handles(),
    )

    # For on-policy: wait until all generators confirm
    await registry.barrier_wait(version, participant="trainer")

# 2. Generator receives notification, fetches weights
async def on_version_available(generator, version):
    # Get fetch plan from registry
    plan = await registry.get_fetch_plan(
        version=version,
        tp_rank=generator.tp_rank,
        tp_size=generator.tp_size,
    )

    # Pull directly from trainers (parallel, batched)
    weights = await pull_weights_direct(plan)

    # Apply to model
    await generator.apply_weights(weights)

    # Confirm to registry
    await registry.barrier_release(version, participant=generator.id)
```

---

## Protocol: Off-Policy Weight Sync

For algorithms tolerating stale weights (DQN, SAC, async PPO):

```
Timeline:
─────────────────────────────────────────────────────────────────────►

Trainer:    [train N] [register v.N] [train N+1] [register v.N+1] ...
                          │                │
                          │                │  (no waiting)
                          ▼                ▼
Registry:          [v.N available]  [v.N+1 available]  [GC old versions]
                          │                │
                          │                │
                          ▼                ▼
Generator:     [using v.N-5] ... [decide to update] [fetch v.N+1] [apply]
```

### Detailed Steps

```python
# 1. Trainer registers without waiting
async def on_train_step_complete(trainer_mesh, version):
    shard_info = await trainer_mesh.get_shard_info.call()
    await registry.register_version(version, shard_info, trainer_handles)
    # No barrier - continue immediately

# 2. Generator periodically checks for updates
async def generator_update_loop(generator, update_interval_steps):
    while True:
        # Check if newer version available
        latest = await registry.get_latest_version()

        if latest > generator.current_version + update_interval_steps:
            # Time to update
            plan = await registry.get_fetch_plan(latest, ...)
            weights = await pull_weights_direct(plan)
            await generator.apply_weights(weights)
            generator.current_version = latest

        await asyncio.sleep(check_interval)

# 3. Registry garbage collects old versions
async def registry_gc_loop(registry, keep_versions=3):
    while True:
        versions = registry.list_versions()
        for v in versions[:-keep_versions]:
            if registry.get_refcount(v) == 0:
                registry.delete_version(v)
        await asyncio.sleep(gc_interval)
```

---

## Scaling to 1000 GPUs

### Challenge 1: Fan-in Mitigation

**Problem**: 744 generators pulling from 256 trainers simultaneously.

**Solution: Staggered Pull with Jitter**
```python
async def fetch_weights_with_jitter(generator, plan):
    # Add random jitter to spread load
    jitter = random.uniform(0, max_jitter_ms) / 1000
    await asyncio.sleep(jitter)

    # Pull with rate limiting
    async with rate_limiter:
        return await pull_weights_direct(plan)
```

**Solution: Hierarchical Relay (for very large scale)**
```
Trainers (256 GPUs)
    │
    ▼
Relay Nodes (1 per rack, 32 nodes)  ← first-level aggregation
    │
    ▼
Generators (744 GPUs)
```

Each relay pulls once from trainers, serves multiple generators in its rack.

### Challenge 2: Multi-Node Transfer

**Solution: Topology-Aware Transfer Selection**
```python
def select_transfer_method(src_rank, dst_rank, topology):
    src_node = topology.get_node(src_rank)
    dst_node = topology.get_node(dst_rank)

    if src_node == dst_node:
        return CudaIPCTransfer()  # ~600 GB/s

    src_rack = topology.get_rack(src_node)
    dst_rack = topology.get_rack(dst_node)

    if src_rack == dst_rack:
        return NCCLTransfer(backend="nvswitch")  # ~900 GB/s (NVSwitch)

    return NCCLTransfer(backend="rdma")  # ~50 GB/s (400Gbps RDMA)
```

### Challenge 3: NCCL Communicator Setup

**Problem**: Can't create NCCL communicator between arbitrary ranks dynamically.

**Solution: Pre-established Communication Groups**
```python
# During initialization, create communicator groups
class WeightSyncCommunicator:
    def __init__(self, trainer_ranks, generator_ranks):
        # Group 1: All trainers (for all-gather if needed)
        self.trainer_group = nccl.new_group(trainer_ranks)

        # Group 2: Trainer-Generator pairs by node
        self.node_groups = {}
        for node in nodes:
            trainers_on_node = [r for r in trainer_ranks if on_node(r, node)]
            generators_on_node = [r for r in generator_ranks if on_node(r, node)]
            if trainers_on_node and generators_on_node:
                self.node_groups[node] = nccl.new_group(
                    trainers_on_node + generators_on_node
                )

        # Group 3: Cross-node broadcast groups
        self.broadcast_groups = create_broadcast_tree(trainer_ranks, generator_ranks)
```

### Challenge 4: Memory Management

**Problem**: Trainer must keep weights in GPU memory while generators pull.

**Solution: Double-Buffering + Reference Counting**
```python
class TrainerWeightServer:
    def __init__(self):
        self.weight_buffers = {}  # version -> {param: tensor}
        self.refcounts = {}       # version -> count

    async def register_version(self, version):
        # Copy current weights to versioned buffer
        self.weight_buffers[version] = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        self.refcounts[version] = 0

    async def pin_version(self, version):
        self.refcounts[version] += 1

    async def unpin_version(self, version):
        self.refcounts[version] -= 1
        if self.refcounts[version] == 0 and version < self.current_version - 1:
            # Safe to free
            del self.weight_buffers[version]

    async def get_shard(self, version, param_name, slice_spec):
        # Pin during transfer
        self.pin_version(version)
        try:
            tensor = self.weight_buffers[version][param_name]
            return tensor[slice_spec.to_index()].contiguous()
        finally:
            self.unpin_version(version)
```

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
Total transfers: FSDP × TP = 4
```

### Row-Parallel Param (output projections)
```
FSDP shards rows, TP also shards rows

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

Each TP rank pulls from ONLY matching FSDP rank
Total transfers: min(FSDP, TP) = 2
```

### At Scale: FSDP=256, TP=8

```
Column-parallel param [32768, 32768]:
- Each FSDP rank has [128, 32768] (rows 128 per rank)
- Each TP rank needs [:, 4096] (cols 4096 per rank)
- Each TP rank pulls 128-row slices from ALL 256 FSDP ranks
- Total: 256 × 8 = 2048 transfers (but each is small: 128×4096 = 1MB)

Row-parallel param [32768, 32768]:
- FSDP rank 0-31 have rows 0-4096 → TP rank 0 needs these
- FSDP rank 32-63 have rows 4096-8192 → TP rank 1 needs these
- Total: 32 × 8 = 256 transfers (each is larger: 128×32768 = 8MB)
```

---

## Communication Patterns at Scale

### Pattern 1: Single Node (1-8 GPUs)
```
All CUDA IPC, no network.
Trainer and Generator on same node.

Transfer: ~600 GB/s (NVLink)
8GB model: ~13ms
```

### Pattern 2: Single Rack with NVSwitch (8-64 GPUs)
```
NCCL over NVSwitch for cross-node within rack.
Trainer: 32 GPUs (4 nodes), Generator: 32 GPUs (4 nodes)

Transfer: ~900 GB/s (NVSwitch bisection)
70GB model: ~78ms
```

### Pattern 3: Multi-Rack with RDMA (64-1000+ GPUs)
```
NCCL over RDMA for cross-rack.
Trainer: 256 GPUs (32 nodes), Generator: 744 GPUs (93 nodes)

Transfer: ~50 GB/s per link (400Gbps RDMA)
With parallel transfers: ~200 GB/s aggregate
70GB model: ~350ms (dominated by serialization)
```

### Pattern 4: Hierarchical for Extreme Scale (1000+ GPUs)
```
Two-level hierarchy:
1. Trainers → Relay nodes (1 per rack): NCCL broadcast
2. Relay nodes → Generators: NCCL broadcast within rack

70GB model: ~500ms (but much better load distribution)
```

---

## Data Structures

### WeightReadySignal
```python
@dataclass
class WeightReadySignal:
    version: int
    timestamp: float
    param_metadata: dict[str, ParamMetadata]
    trainer_handles: list[ActorHandle]  # One per FSDP rank

@dataclass
class ParamMetadata:
    name: str
    global_shape: tuple[int, ...]
    dtype: torch.dtype
    fsdp_shards: list[ShardInfo]  # One per FSDP rank
    tp_shard_dim: int | None      # Which dim to shard for TP (None = replicated)

@dataclass
class ShardInfo:
    rank: int
    offset: tuple[int, ...]  # Offset in global tensor
    shape: tuple[int, ...]   # Local shard shape
    node: str                # Physical node for topology-aware transfer
```

### FetchPlan
```python
@dataclass
class FetchPlan:
    """Computed by generator, describes what to fetch from where."""
    version: int
    tasks: list[FetchTask]

@dataclass
class FetchTask:
    param_name: str
    source_rank: int           # FSDP rank to fetch from
    source_handle: ActorHandle # Direct handle
    source_slice: SliceSpec    # What slice of source's shard
    target_slice: SliceSpec    # Where to put it in my tensor
    transfer_method: TransferMethod  # CUDA_IPC, NCCL, RPC
```

---

## Implementation Plan

### Phase 1: Batched TorchStore (Quick Win, 1-2 days)
**Goal**: Reduce RPC overhead without architectural changes

Changes:
1. Add `get_batch()` to TorchStore API
2. Modify `_WeightFetcher` to batch params

Expected: **10-20x improvement** (78s → 5-8s)

### Phase 2: Direct Pull Protocol (Major Win, 1 week)
**Goal**: Eliminate TorchStore data hop

Changes:
1. Add `WeightRegistry` (evolve TorchStore controller)
2. Add `TrainerWeightServer` endpoint
3. Add `GeneratorWeightClient`
4. Implement on-policy barrier
5. Implement off-policy version management

Expected: **5-10x improvement** over Phase 1 (5s → 0.5-1s for same-node)

### Phase 3: Transport Optimization (Maximum Performance, 1 week)
**Goal**: Optimal transfer for each topology

Changes:
1. Integrate CUDA IPC for same-node
2. Add NCCL communicator setup
3. Topology-aware transfer selection
4. Double-buffering for memory management

Expected: **Near hardware limits** (~100ms for 8GB same-node, ~500ms for 70GB multi-node)

### Phase 4: Scale Testing & Tuning (Production Ready, 1-2 weeks)
**Goal**: Validate at 1000 GPU scale

Changes:
1. Add hierarchical relay for extreme scale
2. Tune jitter/rate limiting parameters
3. Add monitoring and alerting
4. Failure recovery and fallback paths

---

## Comparison with TorchStore

| Aspect | Current TorchStore | New Hybrid Design |
|--------|-------------------|-------------------|
| Data path | Trainer→Store→Generator | Trainer→Generator (direct) |
| Control path | Implicit in put/get | Explicit registry |
| On-policy sync | Manual barrier | Built-in barrier |
| Off-policy | N/A | Version management + GC |
| Multi-version | Yes | Yes (with refcounting) |
| Fallback | N/A | TorchStore as backup |
| CUDA IPC | Transport option | Primary for same-node |
| NCCL | Not supported | Primary for multi-node |
| Topology aware | No | Yes |

### TorchStore's New Role

```python
# TorchStore becomes a coordination service + fallback

class WeightRegistry(TorchStoreController):
    """Evolution of TorchStore for weight sync coordination."""

    # NEW: Version management
    async def register_version(self, version, shard_info, handles): ...
    async def get_fetch_plan(self, version, tp_rank, tp_size): ...

    # NEW: Synchronization primitives
    async def barrier_wait(self, version, participant): ...
    async def barrier_release(self, version, participant): ...

    # EXISTING: Fallback storage (for recovery, debugging)
    async def put(self, key, value): ...  # Still available
    async def get(self, key): ...          # Still available
```

---

## Failure Handling

### Trainer Failure During Transfer
```python
async def pull_weights_with_retry(plan, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await pull_weights_direct(plan)
        except TrainerUnavailableError as e:
            if attempt < max_retries - 1:
                # Refresh plan (trainer may have recovered to different rank)
                plan = await registry.get_fetch_plan(plan.version, ...)
                await asyncio.sleep(backoff(attempt))
            else:
                # Fall back to TorchStore if data was also pushed there
                return await pull_from_torchstore_fallback(plan.version)
```

### Generator Failure During Barrier
```python
async def barrier_wait_with_timeout(version, timeout=60):
    try:
        await asyncio.wait_for(
            registry.barrier_wait(version, "trainer"),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        # Some generators didn't confirm - check which ones
        missing = await registry.get_missing_confirmations(version)
        logger.warning(f"Generators {missing} didn't confirm v{version}")
        # Option 1: Continue anyway (for fault tolerance)
        # Option 2: Trigger generator recovery
        # Option 3: Abort and retry
```

---

## Performance Projections

### Single Node (8 GPUs, 8GB model)
| Phase | Latency | Bottleneck |
|-------|---------|------------|
| Current | ~78s | RPC round-trips |
| Phase 1 | ~5s | RPC serialization |
| Phase 2 | ~500ms | Memory copy |
| Phase 3 | ~50ms | NVLink bandwidth |

### Multi-Node (64 GPUs, 70GB model)
| Phase | Latency | Bottleneck |
|-------|---------|------------|
| Current | ~600s | RPC round-trips |
| Phase 1 | ~60s | RPC serialization |
| Phase 2 | ~10s | Network bandwidth |
| Phase 3 | ~2s | NCCL optimization |

### Large Scale (1000 GPUs, 70GB model)
| Phase | Latency | Bottleneck |
|-------|---------|------------|
| Phase 3 | ~5s | Fan-in, network |
| Phase 4 (hierarchical) | ~2s | Broadcast tree depth |

---

## Open Questions

1. **Relay node placement**: How many relays? Where to place them?
   - Heuristic: 1 relay per rack, on node with most local generators

2. **Version retention policy**: How many versions to keep?
   - On-policy: 1-2 versions (current + in-flight)
   - Off-policy: Based on staleness tolerance (e.g., keep last 10)

3. **Partial failure**: What if only some generators confirm?
   - Option A: Strict - wait for all or timeout
   - Option B: Quorum - proceed if N% confirmed
   - Option C: Best-effort - log warning, continue

4. **Mixed on/off-policy**: Can we support both in same deployment?
   - Yes: Use barrier for on-policy generators, async for off-policy
   - Registry tracks which generators need which sync mode

5. **Checkpoint integration**: How does this interact with checkpointing?
   - WeightRegistry can trigger checkpoint on version boundaries
   - TorchStore fallback doubles as checkpoint storage
