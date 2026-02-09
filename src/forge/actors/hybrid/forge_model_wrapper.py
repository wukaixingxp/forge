# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model wrapper to add explicit position support to TorchTitan models.

This module patches TorchTitan models to accept explicit positions parameter,
enabling proper varlen format support for inference with KV cache.
"""

import torch
import torch.nn as nn
import types
import logging
import os

logger = logging.getLogger(__name__)

# Debug mode controlled by environment variable
# Set FORGE_DEBUG=1 to enable verbose [MODEL] logging
DEBUG_MODE = os.environ.get('FORGE_DEBUG', '0') == '1'


def patch_model_for_positions(model: nn.Module) -> nn.Module:
    """Patch TorchTitan model to accept explicit positions parameter.

    This function modifies the model's forward method and all layer forward
    methods to accept and pass through an explicit positions parameter.

    Args:
        model: TorchTitan model to patch

    Returns:
        Patched model (same object, modified in-place)
    """
    # Patch the main model forward
    original_forward = model.forward

    def model_forward_with_positions(self, input_ids: torch.Tensor, positions: torch.Tensor = None):
        """Modified forward that accepts positions and passes to layers."""
        # If positions not provided, create sequential positions
        if positions is None:
            if input_ids.dim() == 2:
                # Batched: [batch, seq_len]
                batch_size, seq_len = input_ids.shape
                positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
                if DEBUG_MODE:
                    logger.info(f"[MODEL] No positions provided, defaulting to sequential: shape={positions.shape}, values={positions[0].tolist() if batch_size > 0 else []}")
            else:
                # Varlen: [total_tokens]
                positions = torch.arange(input_ids.shape[0], device=input_ids.device)
                if DEBUG_MODE:
                    logger.info(f"[MODEL] No positions provided, defaulting to sequential (varlen): shape={positions.shape}, values={positions.tolist()[:10]}")

        # Embeddings
        h = self.tok_embeddings(input_ids)
        if DEBUG_MODE:
            logger.info(f"[MODEL] After embeddings - shape: {h.shape}, has NaN: {torch.isnan(h).any()}, dtype: {h.dtype}")

        # Get rope_cache
        rope_cache = self.rope_cache

        # Pass through layers with positions
        for i, layer_name in enumerate(self.layers.keys()):
            layer = self.layers[layer_name]
            # Call layer with positions
            h = layer(h, rope_cache, attention_masks=None, positions=positions)
            # Only log critical checkpoints when debug mode is enabled
            if DEBUG_MODE and (i in [0, 17, 35] or i == len(self.layers) - 1):
                logger.info(f"[MODEL] After layer {i} - has NaN: {torch.isnan(h).any()}")

        # Final norm and output
        h = self.norm(h)
        if DEBUG_MODE:
            logger.info(f"[MODEL] After final norm - shape: {h.shape}, has NaN: {torch.isnan(h).any()}")

        output = self.output(h)
        if DEBUG_MODE:
            logger.info(f"[MODEL] After output projection - shape: {output.shape}, has NaN: {torch.isnan(output).any()}")

        return output

    # Bind the new forward method
    model.forward = types.MethodType(model_forward_with_positions, model)
    logger.info("Patched model forward to accept positions parameter")

    # Patch each layer's forward
    if hasattr(model, 'layers'):
        for layer_name in model.layers.keys():
            layer = model.layers[layer_name]
            _patch_layer_forward(layer)

    return model


def _patch_layer_forward(layer: nn.Module):
    """Patch a single TransformerBlock layer to accept positions."""
    original_forward = layer.forward

    def layer_forward_with_positions(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks=None,
        positions: torch.Tensor = None
    ):
        """Modified layer forward that passes positions to attention."""
        # Pre-norm + attention + residual
        attn_input = self.attention_norm(x)

        # ForgeAttention uses nano-vllm signature: forward(positions, hidden_states)
        # It handles rope_cache internally
        attn_output = self.attention(positions, attn_input)

        x = x + attn_output

        # FFN
        ffn_input = self.ffn_norm(x)
        ffn_output = self.feed_forward(ffn_input)
        x = x + ffn_output

        return x

    # Bind the new forward method
    layer.forward = types.MethodType(layer_forward_with_positions, layer)


def create_model_wrapper_class(base_model_class):
    """Create a wrapper class that adds position support to a model class.

    This is an alternative to monkey-patching. Creates a new class that
    wraps the base model and adds position parameter support.

    Args:
        base_model_class: The base model class to wrap

    Returns:
        Wrapper class
    """
    class ModelWrapperWithPositions(nn.Module):
        """Wrapper that adds explicit position support to TorchTitan models."""

        def __init__(self, base_model: nn.Module):
            super().__init__()
            self.model = base_model

            # Expose important attributes
            self.tok_embeddings = base_model.tok_embeddings
            self.layers = base_model.layers
            self.norm = base_model.norm
            self.output = base_model.output
            self.rope_cache = base_model.rope_cache

        def forward(self, input_ids: torch.Tensor, positions: torch.Tensor = None):
            """Forward with explicit positions parameter.

            Args:
                input_ids: Input token IDs [batch, seq_len] or [total_tokens]
                positions: Position indices [batch, seq_len] or [total_tokens]

            Returns:
                logits: Output logits
            """
            # If positions not provided, create sequential positions
            if positions is None:
                if input_ids.dim() == 2:
                    # Batched: [batch, seq_len]
                    batch_size, seq_len = input_ids.shape
                    positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
                else:
                    # Varlen: [total_tokens]
                    positions = torch.arange(input_ids.shape[0], device=input_ids.device)

            # Embeddings
            h = self.model.tok_embeddings(input_ids)

            # Pass through layers with positions
            for layer_name in self.model.layers.keys():
                layer = self.model.layers[layer_name]
                # Call layer - it should have positions parameter via patching
                h = layer(h, self.model.rope_cache, attention_masks=None, positions=positions)

            # Final norm and output
            h = self.model.norm(h)
            return self.model.output(h)

    return ModelWrapperWithPositions
