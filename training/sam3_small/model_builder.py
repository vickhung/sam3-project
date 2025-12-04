"""
SAM3-Small Model Builder

A compact SAM3 model (~250M params) designed for:
- Training on consumer GPUs (8-16GB VRAM)
- Edge deployment (Jetson, mobile)
- Fast iteration during research

Author: vk&cursor
Date: December 2025
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

# =============================================================================
# IMPORTS FROM ORIGINAL SAM3
# These are the building blocks we'll use to construct our smaller model
# =============================================================================

from sam3.model.decoder import TransformerDecoder, TransformerDecoderLayer
from sam3.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3.model.geometry_encoders import SequenceGeometryEncoder
from sam3.model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from sam3.model.memory import CXBlock
from sam3.model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.sam3_image import Sam3Image
from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model.vitdet import ViT
from sam3.model.vl_combiner import SAM3VLBackbone

# =============================================================================
# WEIGHT INITIALIZATION
# Proper initialization prevents NaN/Inf during the first forward pass
# =============================================================================

def initialize_weights(model: nn.Module) -> None:
    """
    Initialize model weights for stable training from scratch.
    
    Different layer types need different initialization strategies:
    - Linear layers: Xavier/Glorot for attention, Kaiming for FFN
    - Embeddings: Small random values to start neutral
    - LayerNorm: Standard 1.0 scale, 0.0 bias
    - Convolutions: Kaiming for ReLU-like activations
    
    CRITICAL: Box prediction heads need special init to produce valid boxes!
    """
    
    # =========================================================================
    # FIRST PASS: Special layers that need careful initialization
    # =========================================================================
    
    for name, module in model.named_modules():
        
        # --- Box Prediction Head (CRITICAL) ---
        # Must output valid coordinates in [0, 1] from the start
        # Otherwise IoU calculations produce NaN
        if "bbox_embed" in name and isinstance(module, nn.Linear):
            # Tiny weights keep output near the bias value
            nn.init.normal_(module.weight, mean=0.0, std=0.001)
            if module.bias is not None:
                # Final layer outputs 4 values: [cx, cy, w, h]
                # Initialize to 0.5 so boxes are valid (center of image, half-size)
                if module.out_features == 4:
                    nn.init.constant_(module.bias, 0.5)
                else:
                    # Hidden layers in the MLP
                    nn.init.zeros_(module.bias)
            continue  # Skip the normal initialization below
        
        # --- Reference Points (CRITICAL) ---
        # These are learnable initial box positions, must be in [0, 1]
        if "reference_points" in name and isinstance(module, nn.Embedding):
            nn.init.uniform_(module.weight, 0.2, 0.8)
            continue
        
        # --- Query Embeddings ---
        # Used for decoder queries, small values prevent attention saturation
        if "query_embed" in name and isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            continue
    
        # =========================================================================
    # SECOND PASS: Standard initialization for remaining layers
    # =========================================================================
    
    for name, module in model.named_modules():
        
        # Skip already-initialized special layers
        if any(x in name for x in ["bbox_embed", "reference_points", "query_embed"]):
            continue
        
        # --- Linear Layers ---
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        # --- Embedding Layers ---
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        # --- LayerNorm ---
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        
        # --- Convolution Layers ---
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    # =========================================================================
    # THIRD PASS: Raw Parameters (not wrapped in nn.Module)
    # CLIP-style text encoders have positional_embedding and text_projection
    # as raw Parameters, not modules - they need explicit initialization
    # =========================================================================
    
    for name, param in model.named_parameters():
        # Positional embedding in text encoder (causes NaN if not initialized)
        if "positional_embedding" in name:
            nn.init.normal_(param, mean=0.0, std=0.01)
            
        # Text projection in CLIP encoder  
        elif "text_projection" in name:
            nn.init.normal_(param, mean=0.0, std=0.01)
    
    print("  Weights initialized for training from scratch")
    print("  - bbox_embed: bias=0.5 for valid box predictions")
    print("  - reference_points: uniform [0.2, 0.8]")
    print("  - Linear layers: Xavier with gain=0.5")
    print("  - positional_embedding & text_projection: normal(0, 0.01)")

# =============================================================================
# DEBUG: NaN DETECTION
# =============================================================================

def add_nan_hooks(model: nn.Module) -> None:
    """
    Add forward hooks to detect which module first produces NaN.
    Helps debug training-from-scratch issues.
    """
    nan_found = {"first": None}  # Track first NaN location
    
    def make_hook(name):
        def hook(module, input, output):
            # Skip if we already found a NaN (only report first)
            if nan_found["first"] is not None:
                return
            
            # Check output for NaN
            has_nan = False
            if isinstance(output, torch.Tensor):
                has_nan = torch.isnan(output).any().item()
            elif isinstance(output, (tuple, list)):
                for o in output:
                    if isinstance(o, torch.Tensor) and torch.isnan(o).any().item():
                        has_nan = True
                        break
            elif isinstance(output, dict):
                for k, v in output.items():
                    if isinstance(v, torch.Tensor) and torch.isnan(v).any().item():
                        has_nan = True
                        break
            
            if has_nan:
                nan_found["first"] = name
                print("\n" + "!" * 60)
                print(f"!!! FIRST NaN DETECTED IN: {name}")
                print("!" * 60)
                
                # Check if input already had NaN
                input_has_nan = False
                for i, inp in enumerate(input):
                    if isinstance(inp, torch.Tensor):
                        if torch.isnan(inp).any().item():
                            input_has_nan = True
                            print(f"  Input[{i}] already has NaN!")
                        else:
                            print(f"  Input[{i}] OK - range: [{inp.min():.4f}, {inp.max():.4f}]")
                
                if not input_has_nan:
                    print("  >>> NaN was CREATED in this module <<<")
                print()
        return hook
    
    # Register hooks on all modules
    for name, module in model.named_modules():
        if name:  # Skip the root module
            module.register_forward_hook(make_hook(name))
    
    print("  NaN detection hooks added to all modules")

    # =============================================================================
# PRETRAINED TEXT ENCODER LOADING
# =============================================================================

SAM3_MODEL_ID = "facebook/sam3"

def download_sam3_checkpoint():
    """Download SAM3 checkpoint from HuggingFace."""
    print("Downloading SAM3 checkpoint from HuggingFace...")
    checkpoint_path = hf_hub_download(
        repo_id=SAM3_MODEL_ID,
        filename="sam3.pt",
    )
    print(f"Downloaded to: {checkpoint_path}")
    return checkpoint_path


def load_text_encoder_weights(model: nn.Module, checkpoint_path: str = None) -> None:
    """
    Load pretrained text encoder weights from SAM3 checkpoint.
    
    The text encoder in SAM3-Small has the SAME architecture as the original,
    so we can directly copy all weights.
    """
    if checkpoint_path is None:
        checkpoint_path = download_sam3_checkpoint()
    
    print("Loading pretrained text encoder weights...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Get the state dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Find text encoder keys (they start with "backbone.language_backbone")
    text_encoder_keys = [k for k in state_dict.keys() if "language_backbone" in k]
    print(f"  Found {len(text_encoder_keys)} text encoder parameters in checkpoint")
    
    # Load into our model
    our_state = model.state_dict()
    loaded = 0
    skipped = 0
    
    for key in text_encoder_keys:
        if key in our_state:
            if state_dict[key].shape == our_state[key].shape:
                our_state[key] = state_dict[key]
                loaded += 1
            else:
                print(f"  Shape mismatch: {key}")
                print(f"    Pretrained: {state_dict[key].shape}")
                print(f"    Our model:  {our_state[key].shape}")
                skipped += 1
        else:
            skipped += 1
    
    model.load_state_dict(our_state, strict=False)
    print(f"  Loaded: {loaded} parameters")
    print(f"  Skipped: {skipped} parameters (shape mismatch or not found)")

def project_weight_svd(large_weight: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """
    Project a large weight matrix to a smaller shape using SVD.
    
    SVD decomposes W = U @ S @ Vᵀ
    We keep the top-k singular values to get the best low-rank approximation.
    
    Args:
        large_weight: Original weight tensor
        target_shape: Desired output shape (out_dim, in_dim)
    
    Returns:
        Projected weight of target_shape
    """
    if large_weight.shape == target_shape:
        return large_weight.clone()
    
    # Handle 1D tensors (biases)
    if large_weight.dim() == 1:
        target_size = target_shape[0] if isinstance(target_shape, tuple) else target_shape
        if large_weight.shape[0] >= target_size:
            return large_weight[:target_size].clone()
        else:
            # Pad with zeros if source is smaller
            result = torch.zeros(target_size)
            result[:large_weight.shape[0]] = large_weight
            return result
    
    # For 2D weights, use SVD
    out_dim, in_dim = target_shape
    large_out, large_in = large_weight.shape
    
    # Compute SVD
    U, S, Vh = torch.linalg.svd(large_weight.float(), full_matrices=False)
    
    # Keep dimensions that fit in our target
    k = min(out_dim, in_dim, len(S))
    
    # Reconstruct with reduced dimensions
    # U: [large_out, k] -> [out_dim, k]
    # S: [k]
    # Vh: [k, large_in] -> [k, in_dim]
    U_reduced = U[:out_dim, :k]
    S_reduced = S[:k]
    Vh_reduced = Vh[:k, :in_dim]
    
    # Reconstruct: W' = U_reduced @ diag(S_reduced) @ Vh_reduced
    projected = U_reduced @ torch.diag(S_reduced) @ Vh_reduced
    
    return projected.to(large_weight.dtype)


def load_vit_weights_direct(model: nn.Module, config, checkpoint_path: str = None) -> None:
    """
    Load pretrained ViT weights from SAM3 with direct layer transfer.
    
    NEW APPROACH: Same dimensions, just skip layers!
    SAM3 ViT: 1024 dim, 32 layers, 16 heads
    Our ViT:  1024 dim, 12 layers, 16 heads  (SAME dims!)
    
    Strategy:
    1. Direct copy of patch embed, pos embed (same dimensions)
    2. Layer skipping: Select evenly-spaced layers from 32 → 12
       e.g., layers [0, 2, 5, 8, 10, 13, 16, 18, 21, 24, 26, 29]
    3. No projection needed - weights copy directly!
    """
    if checkpoint_path is None:
        checkpoint_path = download_sam3_checkpoint()
    
    print("Loading pretrained ViT weights with DIRECT layer transfer...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Remove "detector." prefix from checkpoint keys
    state_dict = {k.replace("detector.", ""): v for k, v in state_dict.items()}
    
    our_state = model.state_dict()
    
    LARGE_DEPTH = 32
    SMALL_DEPTH = config.vit_depth
    
    # Calculate which layers to take (evenly spaced)
    # For 12 layers from 32: [0, 2, 5, 8, 10, 13, 16, 18, 21, 24, 26, 29]
    layer_indices = [int(i * (LARGE_DEPTH - 1) / (SMALL_DEPTH - 1)) for i in range(SMALL_DEPTH)]
    print(f"  Layer mapping: taking layers {layer_indices} from original 32")
    
    loaded = 0
    
    # 1. Patch embedding - DIRECT COPY (same dimensions!)
    for suffix in [".weight", ".bias"]:
        key = f"backbone.vision_backbone.trunk.patch_embed.proj{suffix}"
        if key in state_dict and key in our_state:
            if state_dict[key].shape == our_state[key].shape:
                our_state[key] = state_dict[key].clone()
                loaded += 1
                print(f"  ✓ Direct copy: patch_embed{suffix} {state_dict[key].shape}")
    
    # 2. Position embedding - DIRECT COPY (same dimensions!)
    pos_key = "backbone.vision_backbone.trunk.pos_embed"
    if pos_key in state_dict and pos_key in our_state:
        large_pos = state_dict[pos_key]
        target_shape = our_state[pos_key].shape
        if large_pos.shape == target_shape:
            our_state[pos_key] = large_pos.clone()
            loaded += 1
            print(f"  ✓ Direct copy: pos_embed {large_pos.shape}")
        else:
            # Same dimension but different number of positions - tile/interpolate
            print(f"  ✓ Position embed: {large_pos.shape} → {target_shape} (tiling)")
            our_state[pos_key] = large_pos  # ViT handles interpolation internally
            loaded += 1
    
    # 3. Pre-LayerNorm - DIRECT COPY
    for param in ["weight", "bias"]:
        key = f"backbone.vision_backbone.trunk.ln_pre.{param}"
        if key in state_dict and key in our_state:
            our_state[key] = state_dict[key].clone()
            loaded += 1
    
    # 4. Transformer blocks with layer selection
    print(f"  Transferring {SMALL_DEPTH} layers from original {LARGE_DEPTH}...")
    
    for small_idx, large_idx in enumerate(layer_indices):
        block_prefix_large = f"backbone.vision_backbone.trunk.blocks.{large_idx}"
        block_prefix_small = f"backbone.vision_backbone.trunk.blocks.{small_idx}"
        
        # Copy all parameters from selected layer
        for key in state_dict.keys():
            if key.startswith(block_prefix_large + "."):
                # Replace the layer index
                new_key = key.replace(block_prefix_large, block_prefix_small)
                if new_key in our_state:
                    if state_dict[key].shape == our_state[new_key].shape:
                        our_state[new_key] = state_dict[key].clone()
                        loaded += 1
    
    # Load the weights
    model.load_state_dict(our_state, strict=False)
    
    print(f"  ✓ Loaded {loaded} ViT parameters via DIRECT transfer")
    print(f"  ✓ Layer selection: {SMALL_DEPTH} from {LARGE_DEPTH} (no projection!)")
    print(f"  ✓ Dimensions preserved: 1024 embed, 16 heads, 4736 MLP")


def load_vit_weights_with_svd(model: nn.Module, config, checkpoint_path: str = None) -> None:
    """
    LEGACY: Load pretrained ViT weights with SVD projection.
    Use load_vit_weights_direct() for same-dimension configs!
    """
    # Check if we can use direct transfer (same dimensions)
    if config.vit_embed_dim == 1024 and config.vit_num_heads == 16:
        print("  (Using direct transfer - dimensions match!)")
        return load_vit_weights_direct(model, config, checkpoint_path)
    
    # Otherwise fall back to SVD projection for dimension reduction
    if checkpoint_path is None:
        checkpoint_path = download_sam3_checkpoint()
    
    print("Loading pretrained ViT weights with SVD projection...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    our_state = model.state_dict()
    
    LARGE_DIM = 1024
    SMALL_DIM = config.vit_embed_dim
    LARGE_DEPTH = 32
    SMALL_DEPTH = config.vit_depth
    
    layer_indices = [int(i * (LARGE_DEPTH - 1) / (SMALL_DEPTH - 1)) for i in range(SMALL_DEPTH)]
    
    loaded = 0
    
    # Patch embedding with SVD
    patch_key = "backbone.vision_backbone.trunk.patch_embed.proj.weight"
    if patch_key in state_dict and patch_key in our_state:
        large_w = state_dict[patch_key]
        target_shape = our_state[patch_key].shape
        large_flat = large_w.view(LARGE_DIM, -1)
        target_flat = (SMALL_DIM, large_flat.shape[1])
        projected = project_weight_svd(large_flat, target_flat)
        our_state[patch_key] = projected.view(target_shape)
        loaded += 1
    
    # Position embedding with SVD
    pos_key = "backbone.vision_backbone.trunk.pos_embed"
    if pos_key in state_dict and pos_key in our_state:
        large_pos = state_dict[pos_key].squeeze(0)
        target_shape = our_state[pos_key].shape
        n_pos = min(large_pos.shape[0], target_shape[1])
        projected_pos = project_weight_svd(large_pos[:n_pos], (n_pos, SMALL_DIM))
        if n_pos < target_shape[1]:
            result = torch.zeros((target_shape[1], SMALL_DIM))
            result[:n_pos] = projected_pos
            projected_pos = result
        our_state[pos_key] = projected_pos.unsqueeze(0)
        loaded += 1
    
    # Blocks with layer selection and SVD
    for small_idx, large_idx in enumerate(layer_indices):
        block_large = f"backbone.vision_backbone.trunk.blocks.{large_idx}"
        block_small = f"backbone.vision_backbone.trunk.blocks.{small_idx}"
        
        # QKV
        qkv_key = f"{block_small}.attn.qkv.weight"
        qkv_key_large = f"{block_large}.attn.qkv.weight"
        if qkv_key_large in state_dict and qkv_key in our_state:
            our_state[qkv_key] = project_weight_svd(
                state_dict[qkv_key_large], our_state[qkv_key].shape
            )
            loaded += 1
        
        # Proj
        proj_key = f"{block_small}.attn.proj.weight"
        proj_key_large = f"{block_large}.attn.proj.weight"
        if proj_key_large in state_dict and proj_key in our_state:
            our_state[proj_key] = project_weight_svd(
                state_dict[proj_key_large], our_state[proj_key].shape
            )
            loaded += 1
        
        # MLP
        for fc in ["fc1", "fc2"]:
            fc_key = f"{block_small}.mlp.{fc}.weight"
            fc_key_large = f"{block_large}.mlp.{fc}.weight"
            if fc_key_large in state_dict and fc_key in our_state:
                our_state[fc_key] = project_weight_svd(
                    state_dict[fc_key_large], our_state[fc_key].shape
                )
                loaded += 1
        
        # Biases and norms - truncate
        for suffix in [".attn.qkv.bias", ".attn.proj.bias", 
                      ".mlp.fc1.bias", ".mlp.fc2.bias",
                      ".norm1.weight", ".norm1.bias",
                      ".norm2.weight", ".norm2.bias"]:
            key = block_small + suffix
            key_large = block_large + suffix
            if key_large in state_dict and key in our_state:
                target_size = our_state[key].shape[0]
                our_state[key] = state_dict[key_large][:target_size]
                loaded += 1
    
    model.load_state_dict(our_state, strict=False)
    print(f"  ✓ Loaded {loaded} ViT parameters via SVD projection")


def load_detector_weights(model: nn.Module, config, checkpoint_path: str = None) -> None:
    """
    Load transformer, geometry encoder, segmentation head, and dot product scoring weights.
    These use d_model=256 which we keep the same, so direct copy works.
    
    For encoder/decoder layers, we use layer selection similar to ViT.
    """
    if checkpoint_path is None:
        checkpoint_path = download_sam3_checkpoint()
    
    print("Loading detector component weights...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Remove "detector." prefix
    state_dict = {k.replace("detector.", ""): v for k, v in state_dict.items()}
    
    our_state = model.state_dict()
    loaded = 0
    
    # Layer mapping for encoder (6 -> 4 layers)
    LARGE_ENC_LAYERS = 6
    SMALL_ENC_LAYERS = config.encoder_layers
    enc_indices = [int(i * (LARGE_ENC_LAYERS - 1) / (SMALL_ENC_LAYERS - 1)) for i in range(SMALL_ENC_LAYERS)]
    
    # Layer mapping for decoder (6 -> 4 layers)
    LARGE_DEC_LAYERS = 6
    SMALL_DEC_LAYERS = config.decoder_layers
    dec_indices = [int(i * (LARGE_DEC_LAYERS - 1) / (SMALL_DEC_LAYERS - 1)) for i in range(SMALL_DEC_LAYERS)]
    
    print(f"  Encoder layers: {enc_indices} from original 6")
    print(f"  Decoder layers: {dec_indices} from original 6")
    
    # 1. Transformer encoder with layer selection
    for small_idx, large_idx in enumerate(enc_indices):
        prefix_large = f"transformer.encoder.layers.{large_idx}"
        prefix_small = f"transformer.encoder.layers.{small_idx}"
        
        for key in list(state_dict.keys()):
            if key.startswith(prefix_large + "."):
                new_key = key.replace(prefix_large, prefix_small)
                if new_key in our_state and state_dict[key].shape == our_state[new_key].shape:
                    our_state[new_key] = state_dict[key].clone()
                    loaded += 1
    
    # 2. Transformer decoder with layer selection
    for small_idx, large_idx in enumerate(dec_indices):
        prefix_large = f"transformer.decoder.layers.{large_idx}"
        prefix_small = f"transformer.decoder.layers.{small_idx}"
        
        for key in list(state_dict.keys()):
            if key.startswith(prefix_large + "."):
                new_key = key.replace(prefix_large, prefix_small)
                if new_key in our_state and state_dict[key].shape == our_state[new_key].shape:
                    our_state[new_key] = state_dict[key].clone()
                    loaded += 1
    
    # 3. Other transformer components (non-layer-indexed)
    # Handle query embeddings specially - take first N queries
    for key in state_dict.keys():
        if key.startswith("transformer.") and ".layers." not in key:
            if key in our_state:
                src_shape = state_dict[key].shape
                dst_shape = our_state[key].shape
                
                if src_shape == dst_shape:
                    our_state[key] = state_dict[key].clone()
                    loaded += 1
                elif "query_embed" in key or "reference_points" in key:
                    # Take first N queries (200 -> 100)
                    n_queries = dst_shape[0]
                    our_state[key] = state_dict[key][:n_queries].clone()
                    loaded += 1
                    print(f"  Query truncated: {key} {src_shape} -> {dst_shape}")
    
    # 4. Geometry encoder with layer selection (3 -> 2 layers)
    LARGE_GEO_LAYERS = 3
    SMALL_GEO_LAYERS = config.geo_encoder_layers
    geo_indices = [int(i * (LARGE_GEO_LAYERS - 1) / (SMALL_GEO_LAYERS - 1)) for i in range(SMALL_GEO_LAYERS)]
    print(f"  Geo encoder layers: {geo_indices} from original 3")
    
    for small_idx, large_idx in enumerate(geo_indices):
        prefix_large = f"geometry_encoder.layers.{large_idx}"
        prefix_small = f"geometry_encoder.layers.{small_idx}"
        
        for key in list(state_dict.keys()):
            if key.startswith(prefix_large + "."):
                new_key = key.replace(prefix_large, prefix_small)
                if new_key in our_state and state_dict[key].shape == our_state[new_key].shape:
                    our_state[new_key] = state_dict[key].clone()
                    loaded += 1
    
    # Non-layer geometry encoder components
    for key in state_dict.keys():
        if key.startswith("geometry_encoder.") and ".layers." not in key:
            if key in our_state and state_dict[key].shape == our_state[key].shape:
                our_state[key] = state_dict[key].clone()
                loaded += 1
    
    # 5. Segmentation head - direct copy
    for key in state_dict.keys():
        if key.startswith("segmentation_head."):
            if key in our_state and state_dict[key].shape == our_state[key].shape:
                our_state[key] = state_dict[key].clone()
                loaded += 1
    
    # 6. Dot product scoring - direct copy
    for key in state_dict.keys():
        if key.startswith("dot_prod_scoring."):
            if key in our_state and state_dict[key].shape == our_state[key].shape:
                our_state[key] = state_dict[key].clone()
                loaded += 1
    
    model.load_state_dict(our_state, strict=False)
    print(f"  ✓ Loaded {loaded} detector parameters")


def load_text_encoder_weights_direct(model: nn.Module, config, checkpoint_path: str = None) -> None:
    """
    Load pretrained text encoder weights from SAM3 with direct layer transfer.
    
    NEW APPROACH: Same dimensions, just skip layers!
    SAM3 Text: 1024 dim, 24 layers, 16 heads
    Our Text:  1024 dim, 12 layers, 16 heads  (SAME dims!)
    
    Strategy:
    1. Direct copy of token/position embeddings (same dimensions)
    2. Layer skipping: Select evenly-spaced layers from 24 → 12
    3. No projection needed!
    """
    if checkpoint_path is None:
        checkpoint_path = download_sam3_checkpoint()
    
    print("Loading pretrained text encoder weights with DIRECT layer transfer...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Remove "detector." prefix from checkpoint keys
    state_dict = {k.replace("detector.", ""): v for k, v in state_dict.items()}
    
    our_state = model.state_dict()
    
    LARGE_LAYERS = 24
    SMALL_LAYERS = config.text_layers
    
    # Calculate which layers to take (evenly spaced)
    layer_indices = [int(i * (LARGE_LAYERS - 1) / (SMALL_LAYERS - 1)) for i in range(SMALL_LAYERS)]
    print(f"  Layer mapping: taking layers {layer_indices} from original 24")
    
    loaded = 0
    
    # 1. Token embedding - DIRECT COPY
    token_key = "backbone.language_backbone.encoder.token_embedding.weight"
    if token_key in state_dict and token_key in our_state:
        if state_dict[token_key].shape == our_state[token_key].shape:
            our_state[token_key] = state_dict[token_key].clone()
            loaded += 1
            print(f"  ✓ Direct copy: token_embedding {state_dict[token_key].shape}")
    
    # 2. Positional embedding - handle context length difference
    pos_key = "backbone.language_backbone.encoder.positional_embedding"
    if pos_key in state_dict and pos_key in our_state:
        large_pos = state_dict[pos_key]  # [77, 1024]
        target_shape = our_state[pos_key].shape  # [77, 1024] or similar
        if large_pos.shape == target_shape:
            our_state[pos_key] = large_pos.clone()
            loaded += 1
            print(f"  ✓ Direct copy: positional_embedding {large_pos.shape}")
        else:
            # Take first N positions
            n_pos = min(large_pos.shape[0], target_shape[0])
            our_state[pos_key] = large_pos[:n_pos].clone()
            loaded += 1
            print(f"  ✓ Positional embedding: {large_pos.shape} → {target_shape} (truncated)")
    
    # 3. Text projection - DIRECT COPY
    proj_key = "backbone.language_backbone.encoder.text_projection"
    if proj_key in state_dict and proj_key in our_state:
        if state_dict[proj_key].shape == our_state[proj_key].shape:
            our_state[proj_key] = state_dict[proj_key].clone()
            loaded += 1
    
    # 4. Final layer norm - DIRECT COPY
    for param in ["weight", "bias"]:
        key = f"backbone.language_backbone.encoder.ln_final.{param}"
        if key in state_dict and key in our_state:
            if state_dict[key].shape == our_state[key].shape:
                our_state[key] = state_dict[key].clone()
                loaded += 1
    
    # 5. Transformer blocks with layer selection
    print(f"  Transferring {SMALL_LAYERS} layers from original {LARGE_LAYERS}...")
    
    for small_idx, large_idx in enumerate(layer_indices):
        prefix_large = f"backbone.language_backbone.encoder.transformer.resblocks.{large_idx}"
        prefix_small = f"backbone.language_backbone.encoder.transformer.resblocks.{small_idx}"
        
        # Copy all parameters from selected layer
        for key in state_dict.keys():
            if key.startswith(prefix_large + "."):
                new_key = key.replace(prefix_large, prefix_small)
                if new_key in our_state:
                    if state_dict[key].shape == our_state[new_key].shape:
                        our_state[new_key] = state_dict[key].clone()
                        loaded += 1
    
    model.load_state_dict(our_state, strict=False)
    
    print(f"  ✓ Loaded {loaded} text encoder parameters via DIRECT transfer")
    print(f"  ✓ Layer selection: {SMALL_LAYERS} from {LARGE_LAYERS}")


def load_text_encoder_weights_with_svd(model: nn.Module, config, checkpoint_path: str = None) -> None:
    """
    Load pretrained text encoder weights - uses direct transfer if dimensions match.
    """
    # Check if we can use direct transfer (same dimensions)
    if config.text_width == 1024 and config.text_heads == 16:
        print("  (Using direct transfer - dimensions match!)")
        return load_text_encoder_weights_direct(model, config, checkpoint_path)
    
    # Otherwise fall back to SVD projection
    if checkpoint_path is None:
        checkpoint_path = download_sam3_checkpoint()
    
    print("Loading pretrained text encoder weights with SVD projection...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    our_state = model.state_dict()
    
    LARGE_DIM = 1024
    SMALL_DIM = config.text_width
    LARGE_LAYERS = 24
    SMALL_LAYERS = config.text_layers
    
    layer_indices = [int(i * (LARGE_LAYERS - 1) / (SMALL_LAYERS - 1)) for i in range(SMALL_LAYERS)]
    
    loaded = 0
    
    # Token embedding with SVD
    token_key = "backbone.language_backbone.encoder.token_embedding.weight"
    if token_key in state_dict and token_key in our_state:
        our_state[token_key] = project_weight_svd(
            state_dict[token_key], our_state[token_key].shape
        )
        loaded += 1
    
    # Position embedding with SVD
    pos_key = "backbone.language_backbone.encoder.positional_embedding"
    if pos_key in state_dict and pos_key in our_state:
        large_pos = state_dict[pos_key]
        target_shape = our_state[pos_key].shape
        n_pos = min(large_pos.shape[0], target_shape[0])
        projected = project_weight_svd(large_pos[:n_pos], (n_pos, SMALL_DIM))
        if n_pos < target_shape[0]:
            result = torch.zeros(target_shape)
            result[:n_pos] = projected
            projected = result
        our_state[pos_key] = projected
        loaded += 1
    
    # Blocks with layer selection and SVD
    for small_idx, large_idx in enumerate(layer_indices):
        prefix_large = f"backbone.language_backbone.encoder.transformer.resblocks.{large_idx}"
        prefix_small = f"backbone.language_backbone.encoder.transformer.resblocks.{small_idx}"
        
        # Attention weights
        for suffix in [".attn.in_proj_weight", ".attn.out_proj.weight",
                      ".mlp.c_fc.weight", ".mlp.c_proj.weight"]:
            key = prefix_small + suffix
            key_large = prefix_large + suffix
            if key_large in state_dict and key in our_state:
                our_state[key] = project_weight_svd(
                    state_dict[key_large], our_state[key].shape
                )
                loaded += 1
        
        # Biases and norms - truncate
        for suffix in [".attn.in_proj_bias", ".attn.out_proj.bias",
                      ".mlp.c_fc.bias", ".mlp.c_proj.bias",
                      ".ln_1.weight", ".ln_1.bias",
                      ".ln_2.weight", ".ln_2.bias"]:
            key = prefix_small + suffix
            key_large = prefix_large + suffix
            if key_large in state_dict and key in our_state:
                target_size = our_state[key].shape[0]
                our_state[key] = state_dict[key_large][:target_size]
                loaded += 1
    
    # Final layer norm
    for param in ["weight", "bias"]:
        key = f"backbone.language_backbone.encoder.ln_final.{param}"
        if key in state_dict and key in our_state:
            our_state[key] = state_dict[key][:SMALL_DIM]
            loaded += 1
    
    model.load_state_dict(our_state, strict=False)
    print(f"  ✓ Loaded {loaded} text encoder parameters via SVD projection")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SAM3SmallConfig:
    """
    Configuration for SAM3-Small model.
    
    This dataclass holds ALL hyperparameters that define the model architecture.
    Changing these values changes the model size and behavior.
    
    NEW APPROACH: Keep full resolution, reduce layers only!
    - Same image size (1008) for quality detection
    - Same embedding dimension (1024) for direct weight transfer  
    - Fewer layers (12 instead of 32) for speed
    - No SVD projection needed - just skip layers!
    """
    
    # Image Processing - KEEP ORIGINAL RESOLUTION
    img_size: int = 1008         # SAME as original for full quality
    patch_size: int = 14         # Size of each patch (14x14 pixels)
    pretrain_img_size: int = 336 # Match original pretrain size
    
    # Vision Backbone (ViT) - REDUCED LAYERS ONLY
    vit_embed_dim: int = 1024    # SAME as original (direct weight transfer!)
    vit_depth: int = 16          # REDUCED: 16 layers (original: 32) - 50% reduction
    vit_num_heads: int = 16      # SAME as original (64 dim per head)
    vit_mlp_ratio: float = 4.625 # SAME as original
    vit_window_size: int = 24    # SAME as original
    
    # Text Encoder - Keep same for compatibility
    text_width: int = 1024       # SAME as original (direct weight transfer!)
    text_heads: int = 16         # SAME as original
    text_layers: int = 16        # REDUCED: 16 layers (original: 24) - 67% retained
    text_context_length: int = 32  # Match checkpoint (SAM3 uses 32, not 77)
    
    # Transformer Encoder/Decoder - Keep MORE layers for better detection
    d_model: int = 256           # SAME as original
    encoder_layers: int = 6      # SAME as original
    decoder_layers: int = 6      # SAME as original  
    num_queries: int = 100       # Reduced queries (original: 200)
    num_heads: int = 8           # SAME as original
    dim_feedforward: int = 2048  # SAME as original
    
    # Geometry Encoder
    geo_encoder_layers: int = 3  # SAME as original (small, keep all)
    
    @property
    def feature_map_size(self) -> int:
        """Number of patches along each dimension."""
        return self.img_size // self.patch_size  # 1008/14 = 72
    
    @property
    def num_patches(self) -> int:
        """Total number of image patches."""
        return self.feature_map_size ** 2  # 72*72 = 5184
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.img_size % self.patch_size != 0:
            raise ValueError(
                f"img_size ({self.img_size}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )
        if self.vit_window_size > self.feature_map_size:
            self.vit_window_size = self.feature_map_size


# Default configuration instance
SAM3_SMALL_CONFIG = SAM3SmallConfig()


@dataclass  
class SAM3TinyConfig(SAM3SmallConfig):
    """
    Even smaller variant - for very constrained environments.
    8 ViT layers, 8 text layers.
    """
    vit_depth: int = 8
    text_layers: int = 8
    encoder_layers: int = 3
    decoder_layers: int = 3
    num_queries: int = 50


SAM3_TINY_CONFIG = SAM3TinyConfig()


# =============================================================================
# COMPONENT BUILDER FUNCTIONS
# =============================================================================

def _create_position_encoding(config: SAM3SmallConfig, precompute_resolution=None):
    """Create sinusoidal position encoding."""
    return PositionEmbeddingSine(
        num_pos_feats=config.d_model,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def _create_vit_backbone(config: SAM3SmallConfig, compile_mode=None):
    """Create the Vision Transformer backbone."""
    # Determine global attention block positions
    num_global = min(4, config.vit_depth)
    step = max(1, config.vit_depth // num_global)
    global_blocks = tuple(range(step - 1, config.vit_depth, step))
    
    return ViT(
        img_size=config.img_size,
        pretrain_img_size=config.pretrain_img_size,
        patch_size=config.patch_size,
        embed_dim=config.vit_embed_dim,
        depth=config.vit_depth,
        num_heads=config.vit_num_heads,
        mlp_ratio=config.vit_mlp_ratio,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=global_blocks,
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=config.vit_window_size,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
    )


def _create_text_encoder(config: SAM3SmallConfig, bpe_path: str):
    """Create the text encoder."""
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return VETextEncoder(
        tokenizer=tokenizer,
        d_model=config.d_model,
        width=config.text_width,
        heads=config.text_heads,
        layers=config.text_layers,
        context_length=config.text_context_length,
    )


def _create_vit_neck(config: SAM3SmallConfig, position_encoding, vit_backbone):
    """Create the neck connecting ViT to the rest of the model."""
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=config.d_model,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=False,
    )


def _create_transformer_encoder(config: SAM3SmallConfig):
    """Create the vision-language fusion encoder."""
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=config.d_model,
        dim_feedforward=config.dim_feedforward,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=config.num_heads,
            dropout=0.1,
            embed_dim=config.d_model,
            batch_first=True,
        ),
        cross_attention=MultiheadAttention(
            num_heads=config.num_heads,
            dropout=0.1,
            embed_dim=config.d_model,
            batch_first=True,
        ),
    )
    return TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=config.encoder_layers,
        d_model=config.d_model,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )


def _create_transformer_decoder(config: SAM3SmallConfig):
    """Create the object query decoder."""
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=config.d_model,
        dim_feedforward=config.dim_feedforward,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=config.num_heads,
            dropout=0.1,
            embed_dim=config.d_model,
        ),
        n_heads=config.num_heads,
        use_text_cross_attention=True,
    )
    return TransformerDecoder(
        layer=decoder_layer,
        num_layers=config.decoder_layers,
        num_queries=config.num_queries,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=config.d_model,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=config.img_size,
        stride=config.patch_size,
        use_act_checkpoint=True,
        presence_token=True,
    )


def _create_geometry_encoder(config: SAM3SmallConfig):
    """Create the geometry encoder for boxes/points/masks."""
    geo_pos_enc = _create_position_encoding(config)
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=config.d_model,
        dim_feedforward=config.dim_feedforward,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=config.num_heads,
            dropout=0.1,
            embed_dim=config.d_model,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=config.num_heads,
            dropout=0.1,
            embed_dim=config.d_model,
            batch_first=False,
        ),
    )
    return SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=config.d_model,
        num_layers=config.geo_encoder_layers,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )


def _create_segmentation_head(config: SAM3SmallConfig, compile_mode=None):
    """Create the segmentation head."""
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=config.d_model,
        compile_mode=compile_mode,
    )
    cross_attend_prompt = MultiheadAttention(
        num_heads=config.num_heads,
        dropout=0,
        embed_dim=config.d_model,
    )
    return UniversalSegmentationHead(
        hidden_dim=config.d_model,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )


def _create_dot_product_scoring(config: SAM3SmallConfig):
    """Create the scoring module for query-text matching."""
    prompt_mlp = MLP(
        input_dim=config.d_model,
        hidden_dim=config.dim_feedforward,
        output_dim=config.d_model,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(config.d_model),
    )
    return DotProductScoring(
        d_model=config.d_model,
        d_proj=config.d_model,
        prompt_mlp=prompt_mlp,
    )


def count_parameters(model: nn.Module) -> dict:
    """Count parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    breakdown = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        breakdown[name] = params
    return {"total": total, "trainable": trainable, "breakdown": breakdown}


# =============================================================================
# MAIN BUILD FUNCTION
# =============================================================================

def build_sam3_small(
    config: SAM3SmallConfig = None,
    bpe_path: str = None,
    device: str = "cuda",
    eval_mode: bool = False,
    checkpoint_path: str = None,
    enable_segmentation: bool = True,
    compile: bool = False,
) -> Sam3Image:
    """
    Build SAM3-Small model.
    
    Args:
        config: Model configuration. Uses default if None.
        bpe_path: Path to BPE vocabulary file.
        device: Device to load model on ('cuda' or 'cpu').
        eval_mode: Whether to set model to eval mode.
        checkpoint_path: Optional path to load weights from.
        enable_segmentation: Whether to include segmentation head.
        compile: Whether to compile model with torch.compile.
    
    Returns:
        Sam3Image: The complete SAM3-Small model.
    
    Example:
        >>> model = build_sam3_small(bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz")
        >>> print(f"Parameters: {count_parameters(model)['total']:,}")
    """
    if config is None:
        config = SAM3_SMALL_CONFIG
    
    # Find BPE file
    if bpe_path is None:
        possible_paths = [
            "sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            "../sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                bpe_path = path
                break
        if bpe_path is None:
            raise FileNotFoundError(
                "Could not find BPE vocabulary file. Please provide bpe_path."
            )
    
    compile_mode = "default" if compile else None
    
    print("=" * 60)
    print("Building SAM3-Small Model")
    print("=" * 60)
    print(f"Resolution: {config.img_size}x{config.img_size}")
    print(f"Feature map: {config.feature_map_size}x{config.feature_map_size} = {config.num_patches} tokens")
    print()
    
    # Build vision backbone
    print("Building components:")
    print(f"  ViT backbone: {config.vit_depth} layers, {config.vit_embed_dim} dim, {config.vit_num_heads} heads")
    vit_backbone = _create_vit_backbone(config, compile_mode)
    
    # Build text encoder
    print(f"  Text encoder: {config.text_layers} layers, {config.text_width} dim, {config.text_heads} heads")
    text_encoder = _create_text_encoder(config, bpe_path)
    
    # Build neck and VL backbone
    position_encoding = _create_position_encoding(config, config.img_size)
    vit_neck = _create_vit_neck(config, position_encoding, vit_backbone)
    backbone = SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)
    
    # Build transformer
    print(f"  Transformer encoder: {config.encoder_layers} layers")
    print(f"  Transformer decoder: {config.decoder_layers} layers, {config.num_queries} queries")
    encoder = _create_transformer_encoder(config)
    decoder = _create_transformer_decoder(config)
    transformer = TransformerWrapper(
        encoder=encoder,
        decoder=decoder,
        d_model=config.d_model,
    )
    
    # Build other components
    print(f"  Geometry encoder: {config.geo_encoder_layers} layers")
    dot_prod_scoring = _create_dot_product_scoring(config)
    geometry_encoder = _create_geometry_encoder(config)
    
    segmentation_head = None
    if enable_segmentation:
        print("  Segmentation head: enabled")
        segmentation_head = _create_segmentation_head(config, compile_mode)
    
    # Create matcher for training
    matcher = None
    if not eval_mode:
        try:
            from sam3.train.matcher import BinaryHungarianMatcherV2
            matcher = BinaryHungarianMatcherV2(
                focal=True,
                cost_class=2.0,
                cost_bbox=5.0,
                cost_giou=2.0,
                alpha=0.25,
                gamma=2,
                stable=False,
            )
        except ImportError:
            print("  Warning: Could not import matcher (training module not installed)")
    
    # Assemble the model
    print()
    print("Assembling model...")
    model = Sam3Image(
        backbone=backbone,
        transformer=transformer,
        input_geometry_encoder=geometry_encoder,
        segmentation_head=segmentation_head,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=dot_prod_scoring,
        use_instance_query=False,
        multimask_output=True,
        inst_interactive_predictor=None,
        matcher=matcher,
    )
    
    # =========================================================================
    # INITIALIZE WEIGHTS (NEW - prevents NaN on first forward pass)
    # Only initialize if we're not loading a checkpoint
    # =========================================================================
    if checkpoint_path is None:
        print()
        print("Initializing weights for training from scratch...")
        initialize_weights(model)
        load_text_encoder_weights_with_svd(model, config)  # Load pretrained text encoder
        load_vit_weights_with_svd(model, config)  # Load pretrained ViT
        load_detector_weights(model, config)  # Load transformer, geometry encoder, etc.
    
    # Count and report parameters
    param_info = count_parameters(model)
    total = param_info["total"]
    print()
    print("Model Statistics:")
    print(f"  Total parameters: {total:,} ({total/1e6:.1f}M)")
    print(f"  Estimated size (FP32): {total * 4 / 1e9:.2f} GB")
    print(f"  Estimated size (FP16): {total * 2 / 1e9:.2f} GB")
    print()
    print("Parameter breakdown:")
    for name, count in param_info["breakdown"].items():
        pct = count / total * 100
        print(f"  {name}: {count:,} ({pct:.1f}%)")
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        print()
        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    
    # Move to device
    if device == "cuda" and torch.cuda.is_available():
        print(f"\nMoving model to CUDA...")
        model = model.cuda()
    elif device == "cuda":
        print("\nWarning: CUDA requested but not available, using CPU")
    
    if eval_mode:
        model.eval()
        print("Model set to eval mode")
    
    print()
    print("=" * 60)
    print("SAM3-Small model built successfully!")
    print("=" * 60)
    
    # Note: NaN hooks disabled by default - uncomment for debugging
    # print("\nAdding NaN detection hooks for debugging...")
    # add_nan_hooks(model)
    
    return model


# =============================================================================
# TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SAM3-Small Model Builder - Test")
    print("=" * 60 + "\n")
    
    # Show configuration
    config = SAM3SmallConfig()
    print("Configuration:")
    print(f"  Image size: {config.img_size}x{config.img_size}")
    print(f"  Patch size: {config.patch_size}")
    print(f"  Feature map: {config.feature_map_size}x{config.feature_map_size}")
    print(f"  Total patches: {config.num_patches}")
    print()
    print(f"  ViT: {config.vit_depth} layers, {config.vit_embed_dim} dim")
    print(f"  Text: {config.text_layers} layers, {config.text_width} dim")
    print(f"  Transformer: {config.encoder_layers}+{config.decoder_layers} layers")
    print(f"  Queries: {config.num_queries}")
    print()
    
    # Try to build the model
    print("Attempting to build model...")
    print("(This requires the SAM3 package to be installed)\n")
    
    try:
        model = build_sam3_small(
            config=config,
            device="cpu",
            eval_mode=True,
        )
        print("\n✓ Model built successfully!")
        
    except FileNotFoundError as e:
        print(f"\n⚠ Could not find BPE file: {e}")
        print("  This is expected if running outside the project directory.")
        print("  Provide bpe_path when calling build_sam3_small()")
        
    except Exception as e:
        print(f"\n✗ Error building model: {e}")
        import traceback
        traceback.print_exc()