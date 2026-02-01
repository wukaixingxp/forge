#!/usr/bin/env python3
"""Verify TorchComms RDMA and NVLink P2P availability."""
import os
import torch

print("=" * 60)
print("TorchComms RDMA Support Check")
print("=" * 60)

# Check 1: InfiniBand devices
print(f"\n1. /dev/infiniband exists: {os.path.exists('/dev/infiniband')}")

# Check 2: torchcomms import
try:
    import torchcomms
    version = getattr(torchcomms, "__version__", "unknown")
    print(f"2. torchcomms installed: {version}")
except ImportError as e:
    print(f"2. torchcomms NOT installed: {e}")

# Check 3: _transport module
try:
    from torchcomms._transport import RdmaTransport, RdmaMemory
    print(f"3. RdmaTransport.supported(): {RdmaTransport.supported()}")
except ImportError as e:
    print(f"3. _transport NOT available: {e}")

# Check 4: Monarch RDMA
try:
    from monarch.rdma import is_rdma_available, RDMABuffer
    print(f"4. monarch.rdma.is_rdma_available(): {is_rdma_available()}")
except ImportError as e:
    print(f"4. monarch.rdma NOT available: {e}")

# Check 5: GPU P2P (NVLink)
print("\n5. GPU P2P Access (NVLink):")
device_count = torch.cuda.device_count()
print(f"   Number of GPUs: {device_count}")
if device_count > 1:
    for i in range(min(4, device_count)):
        for j in range(min(4, device_count)):
            if i != j:
                can_access = torch.cuda.can_device_access_peer(i, j)
                print(f"   GPU {i} -> GPU {j}: {can_access}")

# Check 6: TorchStore transport availability
print("\n6. TorchStore Transport Status:")
try:
    from torchstore.transport import TransportType, get_available_transport
    from torchstore.transport.torchcomms.cache import torchcomms_rdma_available
    from torchstore.transport.monarch_rdma import monarch_rdma_transport_available
    from torchstore.transport.cuda_ipc import cuda_ipc_available

    print(f"   Available transport types: {[t.name for t in TransportType]}")
    print(f"   Default available transport: {get_available_transport().name}")
    print(f"   TorchComms RDMA available: {torchcomms_rdma_available()}")
    print(f"   Monarch RDMA available: {monarch_rdma_transport_available()}")
    print(f"   CUDA IPC available: {cuda_ipc_available()}")
except ImportError as e:
    print(f"   TorchStore transport import error: {e}")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
if os.path.exists('/dev/infiniband'):
    print("  InfiniBand hardware: AVAILABLE")
else:
    print("  InfiniBand hardware: NOT AVAILABLE")

if device_count > 1:
    all_p2p = all(
        torch.cuda.can_device_access_peer(i, j)
        for i in range(min(4, device_count))
        for j in range(min(4, device_count))
        if i != j
    )
    if all_p2p:
        print("  NVLink P2P: FULLY CONNECTED")
    else:
        print("  NVLink P2P: PARTIAL CONNECTIVITY")
else:
    print("  NVLink P2P: N/A (single GPU)")

print("\nRecommendation:")
if os.path.exists('/dev/infiniband'):
    print("  Use TorchComms RDMA or Monarch RDMA transport (best for multi-node)")
elif device_count > 1:
    print("  Use CudaIPC transport for GPU-direct weight sync (single-node)")
else:
    print("  Use MonarchRPC transport (CPU staging)")
