# GPU-Direct Weight Sync Analysis

## Git Checkpoint
```bash
# torchstore: commit 4b0306c
# torchforge: commit a3ccdd6
```

## Why CudaIPC is Slow

### The Data
**PUT (Trainer → StorageVolume):** FAST
- 741.88MB in 19.73ms = **~37 GB/s** (good!)
- Individual param reconstruction: 2-10ms

**GET (StorageVolume → Generator):** VERY SLOW
- 741.88MB in 1077ms = **~0.7 GB/s** (50x slower!)
- 0.00MB tensors take 1655-2073ms (!)

### Root Cause: NOT the Transport

The fact that **0-byte tensors take 1-2 seconds** proves the bottleneck is NOT data transfer.

The real problems are:

1. **Per-Parameter RPC Overhead**
   - 8 weight fetchers × 200+ params = 1600+ RPC calls
   - Each `ts.get()` is a full RPC round-trip
   - Even with fast IPC, 1600 round-trips dominate

2. **TorchStore as Intermediary (Extra Hop)**
   ```
   Current: Trainer → TorchStore → Generator (2 hops)
   Better:  Trainer → Generator (1 hop, direct)
   ```

3. **SharedMemory Path**
   ```
   Current flow:
   GPU (Trainer) → GPU (TorchStore) → CPU SharedMem → GPU (Generator)

   This is crazy for same-node GPU-to-GPU!
   ```

4. **Sequential Fetcher Design**
   - `_WeightFetcher.fetch()` loops through params one at a time
   - No batching of RPC calls

## Does CudaIPC Go Through CPU Memory?

**No, CudaIPC itself is GPU-direct.** The IPC handle is just 66 bytes of metadata.

However, the current TorchStore architecture forces:
1. IPC handle creation (fast, GPU)
2. Serialize TransportBuffer (including handle) → RPC (slow, CPU)
3. RPC send/receive (slow, network/IPC)
4. Deserialize TransportBuffer → reconstruct tensor (fast, GPU)

The CPU involvement is in the RPC layer, not the GPU transfer.

## Why Use Put/Get at All?

Current design philosophy:
- TorchStore is a "distributed tensor store"
- Decouples Trainer from Generator
- Enables multi-node deployments
- Supports checkpoint/restore semantics

But for **same-node weight sync**, this is massive overkill.

## The Better Approach: Direct Push

### Architecture Comparison

**Current (TorchStore Intermediary):**
```
Trainer                    TorchStore                Generator
   |                           |                         |
   |--ts.put(param1)---------->|                         |
   |--ts.put(param2)---------->|                         |
   |    (200+ RPC calls)       |                         |
   |                           |<------ts.get(param1)----|
   |                           |<------ts.get(param2)----|
   |                           |    (1600+ RPC calls)    |

Total: 200 + 1600 = 1800 RPC round-trips
```

**Better (Direct Push):**
```
Trainer                                           Generator
   |                                                  |
   |--------direct_push(all_weights)---------------->|
   |                                                  |

Total: 1 RPC round-trip (batched)
```

### Implementation Options

**Option 1: NCCL Broadcast (Best for GPU-GPU)**
```python
# Setup: Create NCCL communicator between Trainer and Generator
# Trainer rank 0, Generator rank 1

# Push weights:
for name, param in state_dict.items():
    nccl.broadcast(param, src=0)  # Trainer broadcasts, Generator receives
```
- Pro: Uses NVLink directly (~600 GB/s)
- Pro: No serialization overhead
- Con: Requires NCCL setup between actors

**Option 2: Direct Actor Call with CudaIPC (Simpler)**
```python
# In Trainer:
async def push_weights_direct(self, generator_workers):
    # Create IPC handles for all params (batched)
    handles = {name: create_ipc_handle(param) for name, param in state_dict.items()}

    # Single RPC call to generator with all handles
    await generator_workers.receive_weights.call(handles)

# In Generator Worker:
async def receive_weights(self, handles):
    for name, handle in handles.items():
        tensor = handle.reconstruct_tensor()
        self.model.load_weight(name, tensor)
```
- Pro: Uses existing Monarch RPC
- Pro: CudaIPC for GPU-direct
- Pro: Single RPC call (batched)
- Con: Still has RPC serialization for handles

**Option 3: Shared GPU Memory Pool**
```python
# Pre-allocate shared GPU buffer that both Trainer and Generator can access
# Trainer writes weights, Generator reads directly
```
- Pro: Zero-copy
- Con: Requires shared CUDA context (complex setup)

## Recommended Plan

### Phase 1: Batch the RPC Calls (Quick Win)
Modify `_WeightFetcher` to batch `ts.get()` calls:
```python
# Current: 200 separate ts.get() calls
for name in param_names:
    param = await ts.get(key)  # 200 RPC calls

# Better: Single batched call
params = await ts.get_batch(keys)  # 1 RPC call
```

Expected improvement: 10-50x on GET operations

### Phase 2: Direct Push (Bigger Win)
Add `push_weights_direct()` to Trainer that sends directly to Generator:
```python
await trainer.push_weights_direct(generator.workers)
```

Expected improvement: 10-15x total (eliminates TorchStore hop)

### Phase 3: NCCL Integration (Maximum Performance)
Use NCCL for the actual GPU-GPU transfer:
```python
# During setup: create NCCL communicator
# During weight sync: NCCL broadcast
```

Expected improvement: Near theoretical NVLink bandwidth

## Performance Targets

| Approach | Push Time | Update Time | Total |
|----------|-----------|-------------|-------|
| Current (MonarchRPC) | ~15s | ~63s | ~78s |
| CudaIPC (per-param) | ~17s | ~33s* | ~50s |
| Batched RPC | ~5s | ~10s | ~15s |
| Direct Push | ~3s | ~5s | ~8s |
| NCCL Broadcast | ~1s | ~3s | ~4s |

*GET phase didn't complete in testing

## Files to Modify

1. **TorchStore `api.py`**: Add `get_batch()` for batched fetching
2. **Generator `_fetch_weights`**: Use batched API
3. **Trainer `titan.py`**: Add `push_weights_direct()`
4. **Generator**: Add `receive_weights_direct()` endpoint
5. **ForgeController**: Pass generator ref to trainer for direct communication
