# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class SimpleGRPOLoss(nn.Module):
    """Simplified GRPO Loss for simplified single step updates
    Inspired by the Hugging Face TRL implementation:
        https://github.com/huggingface/trl/blob/417915a3e4d3e3bc8d7b196594308b8eabf928be/trl/trainer/grpo_trainer.py#L1624.
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, logprobs, ref_logprobs, advantages, padding_mask):
        kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
        per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages
        per_token_loss = -(per_token_policy_loss - self.beta * kl)
        loss = (
            ((per_token_loss * padding_mask).sum(dim=1))
            / (padding_mask.sum(dim=1).clamp(min=1.0))
        ).mean()
        return loss


class BNPOLoss(nn.Module):
    """BNPO Loss: Normalizes loss by total tokens across the batch.

    This loss variant normalizes by the total number of non-padded tokens
    in the entire batch, rather than per-sequence normalization.

    Reference:
        https://github.com/huggingface/trl/blob/417915a3e4d3e3bc8d7b196594308b8eabf928be/trl/trainer/grpo_trainer.py#L1703

    Args:
        beta (float): KL divergence coefficient. Default: 0.1
        epsilon_low (float): Lower clipping bound. Default: 0.1
        epsilon_high (float): Upper clipping bound. Default: 0.1
    """

    def __init__(
        self, beta: float = 0.1, epsilon_low: float = 0.1, epsilon_high: float = 0.1
    ):
        super().__init__()
        self.beta = beta
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high

    def forward(self, logprobs, old_logprobs, ref_logprobs, advantages, padding_mask):
        """
        Args:
            logprobs: Per-token log probabilities from current policy (B, T)
            old_logprobs: Per-token log probabilities from old policy (B, T)
            ref_logprobs: Per-token log probabilities from reference model (B, T)
            advantages: Advantages for each sequence (B,)
            padding_mask: Mask for valid tokens (B, T)

        Returns:
            loss: Scalar loss value
        """
        # Compute KL divergence with reference model
        if self.beta != 0.0:
            per_token_kl = (
                torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
            )

        # Compute importance sampling ratio
        log_ratio = logprobs - old_logprobs
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Clipped policy loss
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Add KL penalty
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # BNPO: Normalize by total number of tokens
        loss = (per_token_loss * padding_mask).sum() / padding_mask.sum().clamp(min=1.0)

        return loss


class DRGRPOLoss(nn.Module):
    """DR-GRPO Loss: Normalizes loss by batch size * max completion length.

    This loss variant uses a fixed normalization based on the maximum possible
    number of tokens (batch_size * max_completion_length).

    Reference:
        https://github.com/huggingface/trl/blob/417915a3e4d3e3bc8d7b196594308b8eabf928be/trl/trainer/grpo_trainer.py#L1706

    Args:
        beta (float): KL divergence coefficient. Default: 0.1
        epsilon_low (float): Lower clipping bound. Default: 0.1
        epsilon_high (float): Upper clipping bound. Default: 0.1
        max_completion_length (int): Maximum completion length
    """

    def __init__(
        self,
        beta: float = 0.1,
        epsilon_low: float = 0.1,
        epsilon_high: float = 0.1,
        max_completion_length: int = 512,
    ):
        super().__init__()
        self.beta = beta
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.max_completion_length = max_completion_length

    def forward(
        self,
        logprobs,
        old_logprobs,
        ref_logprobs,
        advantages,
        padding_mask,
    ):
        """
        Args:
            logprobs: Per-token log probabilities from current policy (B, T)
            old_logprobs: Per-token log probabilities from old policy (B, T)
            ref_logprobs: Per-token log probabilities from reference model (B, T)
            advantages: Advantages for each sequence (B,)
            padding_mask: Mask for valid tokens (B, T)

        Returns:
            loss: Scalar loss value
        """
        # Compute KL divergence with reference model
        if self.beta != 0.0:
            per_token_kl = (
                torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
            )

        # Compute importance sampling ratio
        log_ratio = logprobs - old_logprobs
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Clipped policy loss
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Add KL penalty
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # DR-GRPO: Normalize by batch_size * max_completion_length
        batch_size = per_token_loss.size(0)
        normalizer = batch_size * self.max_completion_length
        loss = (per_token_loss * padding_mask).sum() / normalizer

        return loss


class DAPOLoss(nn.Module):
    """DAPO Loss: Normalizes loss by total items in batch across all processes.

    This loss variant is designed for distributed training, normalizing by
    the total number of completion tokens across the entire global batch.
    Note: This requires passing num_items_in_batch from the training loop.

    Reference:
        https://github.com/huggingface/trl/blob/417915a3e4d3e3bc8d7b196594308b8eabf928be/trl/trainer/grpo_trainer.py#L1709

    Args:
        beta (float): KL divergence coefficient. Default: 0.1
        epsilon_low (float): Lower clipping bound. Default: 0.1
        epsilon_high (float): Upper clipping bound. Default: 0.1
    """

    def __init__(
        self,
        beta: float = 0.1,
        epsilon_low: float = 0.1,
        epsilon_high: float = 0.1,
    ):
        super().__init__()
        self.beta = beta
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high

    def forward(
        self,
        logprobs,
        old_logprobs,
        ref_logprobs,
        advantages,
        padding_mask,
        num_items_in_batch: int,
        num_processes: int = 1,
    ):
        """
        Args:
            logprobs: Per-token log probabilities from current policy (B, T)
            old_logprobs: Per-token log probabilities from old policy (B, T)
            ref_logprobs: Per-token log probabilities from reference model (B, T)
            advantages: Advantages for each sequence (B,)
            padding_mask: Mask for valid tokens (B, T)
            num_items_in_batch: Total number of items in the global batch
            num_processes: Number of processes for distributed training

        Returns:
            loss: Scalar loss value
        """
        # Compute KL divergence with reference model
        if self.beta != 0.0:
            per_token_kl = (
                torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
            )

        # Compute importance sampling ratio
        log_ratio = logprobs - old_logprobs
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Clipped policy loss
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Add KL penalty
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # DAPO: Normalize by total items across all processes
        normalizer = num_items_in_batch / num_processes
        loss = (per_token_loss * padding_mask).sum() / normalizer

        return loss
