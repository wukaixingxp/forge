# Broadcast-Based Weight Updates

This document explains how to use broadcast-based weight updates in the Forge framework, which allows efficient synchronization of model weights between training and inference processes without reading from disk.

## Overview

The broadcast-based weight update mechanism enables efficient weight synchronization using PyTorch's distributed communication primitives. This is particularly useful for:

- **Online learning**: Training and inference happen concurrently
- **Low latency updates**: Avoid disk I/O overhead
- **Distributed training**: Synchronize weights across multiple processes efficiently

## Architecture

The implementation follows the pattern from vLLM's RLHF example:

```
Training Process (Rank 0)          PolicyWorker Instances (Ranks 1, 2, ...)
   [GPU 0]                              [GPU 1]  [GPU 2]  ...
      |                                    |        |
      |  1. init_broadcast_group()         |        |
      |<-----------------------------------+--------+
      |                                    |        |
      |  2. for each parameter:            |        |
      |     - update_broadcast.call()      |        |
      |       (prepare to receive)         |        |
      |                                    |        |
      |  3. broadcast(param, src=0)        |        |
      +------------------------------------+--------+-->
                                           |        |
                                    [Receive weight] [Receive weight]
```

## Key Components

### 1. PolicyWorker Changes

The `/home/kaiwu/work/kaiwu/forge/src/forge/actors/policy.py` file has been updated with:

- **`init_broadcast_group()`**: Initialize a process group for broadcast communication
- **`update_broadcast()`**: Receive a single broadcasted parameter
- **`update(use_broadcast=True)`**: Switch to broadcast mode for weight updates

### 2. New Fields

- `broadcast_group`: Stores the PyTorch distributed process group

## Usage

### Step 1: Initialize Broadcast Group

Before broadcasting weights, initialize the process group on both training and inference processes:

```python
import torch.distributed as dist
from vllm.utils import get_ip, get_open_port

# Get communication parameters
master_address = get_ip()
master_port = get_open_port()

# Training process is rank 0
# PolicyWorkers are ranks 1, 2, ..., N
world_size = 1 + num_policy_workers

# Initialize on PolicyWorker instances
await policy.policy_worker.init_broadcast_group.call(
    master_address=master_address,
    master_port=master_port,
    rank_offset=1,  # Offset because training is rank 0
    world_size=world_size,
)

# Initialize on training process (rank 0)
os.environ["MASTER_ADDR"] = master_address
os.environ["MASTER_PORT"] = str(master_port)

dist.init_process_group(
    backend="nccl",
    init_method=f"tcp://{master_address}:{master_port}",
    rank=0,
    world_size=world_size,
)
```

### Step 2: Broadcast Weights

After training updates, broadcast each parameter:

```python
import torch
from transformers import AutoModelForCausalLM

# Load training model
train_model = AutoModelForCausalLM.from_pretrained("model_name")
train_model.to("cuda:0")

# Perform training updates
# ... training code ...

# Broadcast updated weights
new_version = 1
for name, param in train_model.named_parameters():
    # Prepare PolicyWorker to receive this parameter
    await policy.policy_worker.update_broadcast.call(
        version=new_version,
        param_name=name,
        dtype=param.dtype,
        shape=param.shape,
    )

    # Broadcast from training process (rank 0) to all workers
    dist.broadcast(param, src=0)
```

### Step 3: Update with Broadcast Mode

When calling `update()`, you can now use broadcast mode:

```python
# Old way: read from disk
await policy.update_weights.call(policy_version=1)

# New way: use broadcast
await policy.policy_worker.update.call(version=1, use_broadcast=True)
```

## Complete Example

See `/home/kaiwu/work/kaiwu/forge/examples/broadcast_weight_update.py` for a complete working example.

## Comparison: Disk-based vs Broadcast-based

### Disk-based (Original)

**Advantages:**
- Simple implementation
- Works with existing checkpointing infrastructure
- No need to coordinate processes

**Disadvantages:**
- High latency due to disk I/O
- Requires torchstore/DCP setup
- Not suitable for frequent updates

### Broadcast-based (New)

**Advantages:**
- Low latency (no disk I/O)
- Efficient for frequent updates
- Better for online learning scenarios
- Direct memory-to-memory transfer

**Disadvantages:**
- Requires process group coordination
- All processes must be available during broadcast
- More complex setup

## Implementation Details

### Process Group Initialization

The `init_broadcast_group()` method initializes a NCCL process group for GPU-to-GPU communication:

```python
@endpoint
async def init_broadcast_group(
    self,
    master_address: str,
    master_port: int,
    rank_offset: int,
    world_size: int,
    backend: str = "nccl",
):
    """Initialize a process group for broadcast-based weight updates."""
    os.environ["MASTER_ADDR"] = master_address
    os.environ["MASTER_PORT"] = str(master_port)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            rank=self.rank + rank_offset,
            world_size=world_size,
        )

    self.broadcast_group = dist.group.WORLD
```

### Receiving Broadcasts

The `update_broadcast()` method receives a single parameter via broadcast:

```python
@endpoint
async def update_broadcast(self, version: int, param_name: str, dtype, shape):
    """Update a single model weight via broadcast from source rank."""
    model = self.worker.model_runner.model
    device = torch.device(f"cuda:{self.rank}")

    # Create tensor to receive broadcast
    param = torch.empty(shape, dtype=dtype, device=device)

    # Receive broadcast from rank 0
    dist.broadcast(param, src=0, group=self.broadcast_group)

    # Load into model
    loaded = model.load_weights([(param_name, param)])

    del param
```

## Best Practices

1. **Synchronization**: Ensure all processes are ready before broadcasting
2. **Error Handling**: Add try-except blocks around broadcast operations
3. **Memory Management**: Delete tensors after loading to free memory
4. **Logging**: Add detailed logging to track broadcast progress
5. **Version Control**: Use version numbers to track weight updates

## Troubleshooting

### Issue: "Broadcast group not initialized"

**Solution**: Call `init_broadcast_group()` before attempting to broadcast weights.

### Issue: Timeout during broadcast

**Solution**:
- Check that all processes have initialized the group
- Verify network connectivity between processes
- Increase timeout: `dist.init_process_group(..., timeout=timedelta(minutes=10))`

### Issue: Shape mismatch errors

**Solution**: Ensure the training model and inference model have identical architectures.

## References

- Original vLLM RLHF example: `/home/kaiwu/work/kaiwu/forge/rlhf.py`
- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
- vLLM Documentation: https://docs.vllm.ai/

## Future Improvements

Potential enhancements:

1. **Automatic sharding**: Handle tensor-parallel sharding automatically
2. **Asynchronous updates**: Non-blocking weight updates
3. **Compression**: Compress weights during broadcast
4. **Checkpointing**: Fallback to disk if broadcast fails
5. **Multi-node support**: Extend to multi-node clusters
