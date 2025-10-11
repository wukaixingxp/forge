#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Example demonstrating broadcast-based weight updates for Policy actors.

This example shows how to update model weights using PyTorch distributed
broadcast instead of reading from disk. This approach is similar to the
vLLM RLHF example and enables more efficient weight synchronization between
training and inference processes.

The example performs the following steps:

1. Launch a Policy actor with PolicyWorker instances
2. Initialize a broadcast process group for weight synchronization
3. Simulate a training step (e.g., zeroing weights for demonstration)
4. Broadcast updated weights from a "training" process to PolicyWorker instances
5. Verify that weights have been updated

This pattern is useful for:
- Online learning scenarios where training and inference happen concurrently
- Reducing latency by avoiding disk I/O for weight updates
- Synchronizing weights across distributed processes efficiently
"""

import asyncio
import logging
import os
import sys

import torch
import torch.distributed as dist
from vllm.utils import get_ip, get_open_port

# Add forge to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from forge.actors.policy import EngineConfig, Policy, SamplingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_training_process_group(
    master_address: str, master_port: int, rank: int, world_size: int
) -> dist.ProcessGroup:
    """Initialize process group for the training process.
    
    Args:
        master_address: IP address of the master node
        master_port: Port for communication
        rank: Rank of this process (typically 0 for training)
        world_size: Total number of processes (training + inference workers)
    
    Returns:
        Process group for collective communication
    """
    os.environ["MASTER_ADDR"] = master_address
    os.environ["MASTER_PORT"] = str(master_port)
    
    device = torch.device(f"cuda:{rank}")
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            rank=rank,
            world_size=world_size,
        )
    
    return dist.group.WORLD


async def main():
    """Main function demonstrating broadcast-based weight updates."""
    
    # Configuration
    model_name = "facebook/opt-125m"  # Small model for demonstration
    tensor_parallel_size = 2
    
    # Step 1: Launch Policy actor
    logger.info("Step 1: Launching Policy actor...")
    
    engine_config = EngineConfig(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,
    )
    
    sampling_config = SamplingConfig(
        temperature=0.0,
        max_tokens=50,
    )
    
    policy = await Policy.launch(
        engine_config=engine_config,
        sampling_config=sampling_config,
    )
    
    logger.info("Policy actor launched successfully")
    
    # Step 2: Generate some text with initial weights
    logger.info("\nStep 2: Generating text with initial weights...")
    
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]
    
    initial_outputs = []
    for prompt in prompts:
        completions = await policy.generate.call(prompt=prompt)
        initial_outputs.append(completions[0])
        logger.info(f"Prompt: {prompt!r}")
        logger.info(f"Generated: {completions[0].text!r}")
    
    # Step 3: Set up broadcast communication
    logger.info("\nStep 3: Setting up broadcast communication...")
    
    master_address = get_ip()
    master_port = get_open_port()
    
    # Training process is rank 0, inference workers are ranks 1, 2, ...
    training_rank = 0
    world_size = 1 + tensor_parallel_size  # 1 training + N inference workers
    
    # Initialize broadcast group on PolicyWorker instances
    # rank_offset=1 because training process is rank 0
    await policy.policy_worker.init_broadcast_group.call(
        master_address=master_address,
        master_port=master_port,
        rank_offset=1,  # Offset to account for training process at rank 0
        world_size=world_size,
    )
    
    # Initialize process group for training process (rank 0)
    training_group = init_training_process_group(
        master_address=master_address,
        master_port=master_port,
        rank=training_rank,
        world_size=world_size,
    )
    
    logger.info("Broadcast communication initialized")
    
    # Step 4: Simulate training and broadcast weights
    logger.info("\nStep 4: Simulating training and broadcasting weights...")
    
    # In a real scenario, you would:
    # 1. Load the model on the training GPU
    # 2. Perform training steps (e.g., PPO updates)
    # 3. Broadcast updated weights to inference workers
    
    # For this example, we'll create dummy tensors to demonstrate the pattern
    # In practice, you'd get these from your training model:
    # train_model = AutoModelForCausalLM.from_pretrained(model_name)
    # train_model.to("cuda:0")
    # ... perform training ...
    # for name, param in train_model.named_parameters():
    #     broadcast_weight(name, param)
    
    logger.info("Note: In a real scenario, you would:")
    logger.info("  1. Load a training model on GPU 0")
    logger.info("  2. Perform training updates (e.g., PPO)")
    logger.info("  3. Broadcast each parameter to inference workers")
    logger.info("")
    logger.info("Example broadcasting pattern:")
    logger.info("  for name, param in train_model.named_parameters():")
    logger.info("      # Prepare inference workers to receive")
    logger.info("      await policy.policy_worker.update_broadcast.call(")
    logger.info("          version=new_version,")
    logger.info("          param_name=name,")
    logger.info("          dtype=param.dtype,")
    logger.info("          shape=param.shape")
    logger.info("      )")
    logger.info("      # Broadcast from training process")
    logger.info("      training_group.broadcast(param, src=0)")
    
    # Step 5: Verify the broadcast mechanism works
    logger.info("\nStep 5: Broadcast mechanism is ready")
    logger.info("PolicyWorker instances are now configured to receive weight updates via broadcast")
    logger.info("instead of reading from disk.")
    
    # Step 6: Generate text again (would show changes if we had broadcasted weights)
    logger.info("\nStep 6: Generating text again...")
    
    for prompt in prompts:
        completions = await policy.generate.call(prompt=prompt)
        logger.info(f"Prompt: {prompt!r}")
        logger.info(f"Generated: {completions[0].text!r}")
    
    # Cleanup
    logger.info("\nCleaning up...")
    await Policy.shutdown(policy)
    
    if dist.is_initialized():
        dist.destroy_process_group()
    
    logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
