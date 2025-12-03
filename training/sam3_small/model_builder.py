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


def load_vit_weights_with_svd(model: nn.Module, config, checkpoint_path: str = None) -> None:
    """
    Load pretrained ViT weights from SAM3, projecting dimensions with SVD.
    
    SAM3 ViT: 1024 dim, 32 layers, 16 heads
    Our ViT:  768 dim, 16 layers, 12 heads
    
    Strategy:
    1. Layer skipping: Take every 2nd layer (0, 2, 4, ... 30) 
    2. SVD projection: Project 1024 → 768 dimensions
    3. Head reduction: Take first 12 of 16 heads
    """
    if checkpoint_path is None:
        checkpoint_path = download_sam3_checkpoint()
    
    print("Loading pretrained ViT weights with SVD projection...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Get the state dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Our model's state dict
    our_state = model.state_dict()
    
    # Configuration
    LARGE_DIM = 1024
    SMALL_DIM = config.vit_embed_dim  # 768
    LARGE_DEPTH = 32
    SMALL_DEPTH = config.vit_depth  # 16
    LARGE_HEADS = 16
    SMALL_HEADS = config.vit_num_heads  # 12
    
    # Layer mapping: our layer i → SAM3 layer i*2
    layer_stride = LARGE_DEPTH // SMALL_DEPTH  # 32/16 = 2
    
    loaded = 0
    skipped = 0
    
    # 1. Patch embedding: [1024, 3, 14, 14] → [768, 3, 14, 14]
    patch_key = "backbone.vision_backbone.trunk.patch_embed.proj.weight"
    if patch_key in state_dict and patch_key in our_state:
        large_w = state_dict[patch_key]  # [1024, 3, 14, 14]
        target_shape = our_state[patch_key].shape  # [768, 3, 14, 14]
        # Reshape, project, reshape back
        large_flat = large_w.view(LARGE_DIM, -1)  # [1024, 588]
        target_flat = (SMALL_DIM, large_flat.shape[1])
        projected = project_weight_svd(large_flat, target_flat)
        our_state[patch_key] = projected.view(target_shape)
        loaded += 1
        print(f"  ✓ Projected patch_embed: {large_w.shape} → {target_shape}")
    
    # Patch embed bias
    patch_bias_key = "backbone.vision_backbone.trunk.patch_embed.proj.bias"
    if patch_bias_key in state_dict and patch_bias_key in our_state:
        our_state[patch_bias_key] = state_dict[patch_bias_key][:SMALL_DIM]
        loaded += 1
    
    # 2. Position embedding: [1, N, 1024] → [1, N', 768]
    pos_key = "backbone.vision_backbone.trunk.pos_embed"
    if pos_key in state_dict and pos_key in our_state:
        large_pos = state_dict[pos_key]  # [1, 577, 1024] for 336x336/14
        target_shape = our_state[pos_key].shape  # [1, 324, 768] for 252x252/14
        
        # Project each position's embedding
        large_2d = large_pos.squeeze(0)  # [577, 1024]
        target_2d = (target_shape[1], target_shape[2])  # [324, 768]
        
        # Take first N positions and project dimension
        n_pos = min(large_2d.shape[0], target_2d[0])
        projected_pos = project_weight_svd(large_2d[:n_pos, :], (n_pos, SMALL_DIM))
        
        # If we need more positions, interpolate or pad
        if n_pos < target_2d[0]:
            # Pad with learned initialization (zeros work for now)
            result = torch.zeros(target_2d)
            result[:n_pos, :] = projected_pos
            projected_pos = result
        
        our_state[pos_key] = projected_pos.unsqueeze(0)
        loaded += 1
        print(f"  ✓ Projected pos_embed: {large_pos.shape} → {target_shape}")
    
    # 3. Transformer blocks with layer skipping
    print(f"  Transferring {SMALL_DEPTH} layers (skipping every {layer_stride}nd from {LARGE_DEPTH})...")
    
    for small_idx in range(SMALL_DEPTH):
        large_idx = small_idx * layer_stride  # 0→0, 1→2, 2→4, etc.
        
        block_prefix_large = f"backbone.vision_backbone.trunk.blocks.{large_idx}"
        block_prefix_small = f"backbone.vision_backbone.trunk.blocks.{small_idx}"
        
        # QKV projection: [3*1024, 1024] → [3*768, 768]
        qkv_key = f"{block_prefix_small}.attn.qkv.weight"
        qkv_key_large = f"{block_prefix_large}.attn.qkv.weight"
        if qkv_key_large in state_dict and qkv_key in our_state:
            large_qkv = state_dict[qkv_key_large]  # [3072, 1024]
            target_shape = our_state[qkv_key].shape  # [2304, 768]
            projected = project_weight_svd(large_qkv, target_shape)
            our_state[qkv_key] = projected
            loaded += 1
        
        # QKV bias
        qkv_bias_key = f"{block_prefix_small}.attn.qkv.bias"
        qkv_bias_key_large = f"{block_prefix_large}.attn.qkv.bias"
        if qkv_bias_key_large in state_dict and qkv_bias_key in our_state:
            large_bias = state_dict[qkv_bias_key_large]  # [3072]
            target_size = our_state[qkv_bias_key].shape[0]  # 2304
            our_state[qkv_bias_key] = large_bias[:target_size]
            loaded += 1
        
        # Attention output projection: [1024, 1024] → [768, 768]
        proj_key = f"{block_prefix_small}.attn.proj.weight"
        proj_key_large = f"{block_prefix_large}.attn.proj.weight"
        if proj_key_large in state_dict and proj_key in our_state:
            large_proj = state_dict[proj_key_large]
            target_shape = our_state[proj_key].shape
            projected = project_weight_svd(large_proj, target_shape)
            our_state[proj_key] = projected
            loaded += 1
        
        proj_bias_key = f"{block_prefix_small}.attn.proj.bias"
        proj_bias_key_large = f"{block_prefix_large}.attn.proj.bias"
        if proj_bias_key_large in state_dict and proj_bias_key in our_state:
            our_state[proj_bias_key] = state_dict[proj_bias_key_large][:SMALL_DIM]
            loaded += 1
        
        # MLP fc1: [4096, 1024] → [3072, 768] (mlp_ratio * dim)
        fc1_key = f"{block_prefix_small}.mlp.fc1.weight"
        fc1_key_large = f"{block_prefix_large}.mlp.fc1.weight"
        if fc1_key_large in state_dict and fc1_key in our_state:
            large_fc1 = state_dict[fc1_key_large]
            target_shape = our_state[fc1_key].shape
            projected = project_weight_svd(large_fc1, target_shape)
            our_state[fc1_key] = projected
            loaded += 1
        
        fc1_bias_key = f"{block_prefix_small}.mlp.fc1.bias"
        fc1_bias_key_large = f"{block_prefix_large}.mlp.fc1.bias"
        if fc1_bias_key_large in state_dict and fc1_bias_key in our_state:
            large_bias = state_dict[fc1_bias_key_large]
            target_size = our_state[fc1_bias_key].shape[0]
            our_state[fc1_bias_key] = large_bias[:target_size]
            loaded += 1
        
        # MLP fc2: [1024, 4096] → [768, 3072]
        fc2_key = f"{block_prefix_small}.mlp.fc2.weight"
        fc2_key_large = f"{block_prefix_large}.mlp.fc2.weight"
        if fc2_key_large in state_dict and fc2_key in our_state:
            large_fc2 = state_dict[fc2_key_large]
            target_shape = our_state[fc2_key].shape
            projected = project_weight_svd(large_fc2, target_shape)
            our_state[fc2_key] = projected
            loaded += 1
        
        fc2_bias_key = f"{block_prefix_small}.mlp.fc2.bias"
        fc2_bias_key_large = f"{block_prefix_large}.mlp.fc2.bias"
        if fc2_bias_key_large in state_dict and fc2_bias_key in our_state:
            our_state[fc2_bias_key] = state_dict[fc2_bias_key_large][:SMALL_DIM]
            loaded += 1
        
        # LayerNorms: [1024] → [768]
        for norm_name in ["norm1", "norm2"]:
            for param in ["weight", "bias"]:
                key = f"{block_prefix_small}.{norm_name}.{param}"
                key_large = f"{block_prefix_large}.{norm_name}.{param}"
                if key_large in state_dict and key in our_state:
                    our_state[key] = state_dict[key_large][:SMALL_DIM]
                    loaded += 1
    
    # 4. Final layer norm
    for param in ["weight", "bias"]:
        key = f"backbone.vision_backbone.trunk.ln_pre.{param}"
        if key in state_dict and key in our_state:
            our_state[key] = state_dict[key][:SMALL_DIM]
            loaded += 1
    
    # Load the projected weights
    model.load_state_dict(our_state, strict=False)
    
    print(f"  ✓ Loaded {loaded} ViT parameters via SVD projection")
    print(f"  ✓ Layer mapping: every {layer_stride}th layer from SAM3")
    print(f"  ✓ Dimension projection: {LARGE_DIM} → {SMALL_DIM}")


def load_text_encoder_weights_with_svd(model: nn.Module, config, checkpoint_path: str = None) -> None:
    """
    Load pretrained text encoder weights from SAM3, projecting dimensions with SVD.
    
    SAM3 Text Encoder: 1024 dim, 24 layers, 16 heads
    Our Text Encoder:  512 dim, 12 layers, 8 heads
    
    Strategy:
    1. Layer skipping: Take every 2nd layer (0, 2, 4, ... 22)
    2. SVD projection: Project 1024 → 512 dimensions
    """
    if checkpoint_path is None:
        checkpoint_path = download_sam3_checkpoint()
    
    print("Loading pretrained text encoder weights with SVD projection...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    our_state = model.state_dict()
    
    # Configuration
    LARGE_DIM = 1024
    SMALL_DIM = config.text_width  # 512
    LARGE_LAYERS = 24
    SMALL_LAYERS = config.text_layers  # 12
    
    layer_stride = LARGE_LAYERS // SMALL_LAYERS  # 24/12 = 2
    
    loaded = 0
    
    # 1. Token embedding: [vocab_size, 1024] → [vocab_size, 512]
    token_key = "backbone.language_backbone.encoder.token_embedding.weight"
    if token_key in state_dict and token_key in our_state:
        large_emb = state_dict[token_key]  # [49408, 1024]
        target_shape = our_state[token_key].shape  # [49408, 512]
        # Project each token's embedding
        projected = project_weight_svd(large_emb, target_shape)
        our_state[token_key] = projected
        loaded += 1
        print(f"  ✓ Projected token_embedding: {large_emb.shape} → {target_shape}")
    
    # 2. Positional embedding: [context_length, 1024] → [context_length, 512]
    pos_key = "backbone.language_backbone.encoder.positional_embedding"
    if pos_key in state_dict and pos_key in our_state:
        large_pos = state_dict[pos_key]  # [77, 1024]
        target_shape = our_state[pos_key].shape  # [32, 512]
        # Take first N positions and project
        n_pos = min(large_pos.shape[0], target_shape[0])
        projected = project_weight_svd(large_pos[:n_pos], (n_pos, SMALL_DIM))
        if n_pos < target_shape[0]:
            result = torch.zeros(target_shape)
            result[:n_pos] = projected
            projected = result
        our_state[pos_key] = projected
        loaded += 1
        print(f"  ✓ Projected positional_embedding: {large_pos.shape} → {target_shape}")
    
    # 3. Text projection: [1024, 1024] → [512, 512] (or may not exist in our model)
    proj_key = "backbone.language_backbone.encoder.text_projection"
    if proj_key in state_dict and proj_key in our_state:
        large_proj = state_dict[proj_key]
        target_shape = our_state[proj_key].shape
        projected = project_weight_svd(large_proj, target_shape)
        our_state[proj_key] = projected
        loaded += 1
    
    # 4. Transformer blocks with layer skipping
    print(f"  Transferring {SMALL_LAYERS} layers (skipping every {layer_stride}nd from {LARGE_LAYERS})...")
    
    for small_idx in range(SMALL_LAYERS):
        large_idx = small_idx * layer_stride
        
        prefix_large = f"backbone.language_backbone.encoder.transformer.resblocks.{large_idx}"
        prefix_small = f"backbone.language_backbone.encoder.transformer.resblocks.{small_idx}"
        
        # Attention in_proj (QKV combined): [3*1024, 1024] → [3*512, 512]
        in_proj_key = f"{prefix_small}.attn.in_proj_weight"
        in_proj_key_large = f"{prefix_large}.attn.in_proj_weight"
        if in_proj_key_large in state_dict and in_proj_key in our_state:
            large_w = state_dict[in_proj_key_large]  # [3072, 1024]
            target_shape = our_state[in_proj_key].shape  # [1536, 512]
            projected = project_weight_svd(large_w, target_shape)
            our_state[in_proj_key] = projected
            loaded += 1
        
        in_proj_bias_key = f"{prefix_small}.attn.in_proj_bias"
        in_proj_bias_key_large = f"{prefix_large}.attn.in_proj_bias"
        if in_proj_bias_key_large in state_dict and in_proj_bias_key in our_state:
            large_b = state_dict[in_proj_bias_key_large]
            target_size = our_state[in_proj_bias_key].shape[0]
            our_state[in_proj_bias_key] = large_b[:target_size]
            loaded += 1
        
        # Attention out_proj: [1024, 1024] → [512, 512]
        out_proj_key = f"{prefix_small}.attn.out_proj.weight"
        out_proj_key_large = f"{prefix_large}.attn.out_proj.weight"
        if out_proj_key_large in state_dict and out_proj_key in our_state:
            large_w = state_dict[out_proj_key_large]
            target_shape = our_state[out_proj_key].shape
            projected = project_weight_svd(large_w, target_shape)
            our_state[out_proj_key] = projected
            loaded += 1
        
        out_proj_bias_key = f"{prefix_small}.attn.out_proj.bias"
        out_proj_bias_key_large = f"{prefix_large}.attn.out_proj.bias"
        if out_proj_bias_key_large in state_dict and out_proj_bias_key in our_state:
            our_state[out_proj_bias_key] = state_dict[out_proj_bias_key_large][:SMALL_DIM]
            loaded += 1
        
        # MLP c_fc (first layer): [4096, 1024] → [2048, 512]
        c_fc_key = f"{prefix_small}.mlp.c_fc.weight"
        c_fc_key_large = f"{prefix_large}.mlp.c_fc.weight"
        if c_fc_key_large in state_dict and c_fc_key in our_state:
            large_w = state_dict[c_fc_key_large]
            target_shape = our_state[c_fc_key].shape
            projected = project_weight_svd(large_w, target_shape)
            our_state[c_fc_key] = projected
            loaded += 1
        
        c_fc_bias_key = f"{prefix_small}.mlp.c_fc.bias"
        c_fc_bias_key_large = f"{prefix_large}.mlp.c_fc.bias"
        if c_fc_bias_key_large in state_dict and c_fc_bias_key in our_state:
            large_b = state_dict[c_fc_bias_key_large]
            target_size = our_state[c_fc_bias_key].shape[0]
            our_state[c_fc_bias_key] = large_b[:target_size]
            loaded += 1
        
        # MLP c_proj (second layer): [1024, 4096] → [512, 2048]
        c_proj_key = f"{prefix_small}.mlp.c_proj.weight"
        c_proj_key_large = f"{prefix_large}.mlp.c_proj.weight"
        if c_proj_key_large in state_dict and c_proj_key in our_state:
            large_w = state_dict[c_proj_key_large]
            target_shape = our_state[c_proj_key].shape
            projected = project_weight_svd(large_w, target_shape)
            our_state[c_proj_key] = projected
            loaded += 1
        
        c_proj_bias_key = f"{prefix_small}.mlp.c_proj.bias"
        c_proj_bias_key_large = f"{prefix_large}.mlp.c_proj.bias"
        if c_proj_bias_key_large in state_dict and c_proj_bias_key in our_state:
            our_state[c_proj_bias_key] = state_dict[c_proj_bias_key_large][:SMALL_DIM]
            loaded += 1
        
        # LayerNorms
        for ln_name in ["ln_1", "ln_2"]:
            for param in ["weight", "bias"]:
                key = f"{prefix_small}.{ln_name}.{param}"
                key_large = f"{prefix_large}.{ln_name}.{param}"
                if key_large in state_dict and key in our_state:
                    our_state[key] = state_dict[key_large][:SMALL_DIM]
                    loaded += 1
    
    # 5. Final layer norm
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
    """
    
    # Image Processing
    img_size: int = 252          # Input resolution (must be divisible by patch_size)
    patch_size: int = 14         # Size of each patch (14x14 pixels)
    pretrain_img_size: int = 224 # Used for position embedding initialization
    
    # Vision Backbone (ViT) - ~40% of total parameters
    vit_embed_dim: int = 768     # Hidden dimension (original: 1024)
    vit_depth: int = 16          # Number of transformer layers (original: 32)
    vit_num_heads: int = 12      # Attention heads (original: 16)
    vit_mlp_ratio: float = 4.0   # MLP hidden = embed_dim * mlp_ratio
    vit_window_size: int = 18    # Window attention size
    
    # Text Encoder - ~30% of total parameters
    text_width: int = 512        # Hidden dimension (original: 1024)
    text_heads: int = 8          # Attention heads (original: 16)
    text_layers: int = 12        # Number of layers (original: 24)
    text_context_length: int = 32
    
    # Transformer Encoder/Decoder - ~20% of parameters
    d_model: int = 256           # Model dimension (keep same as original)
    encoder_layers: int = 4      # Fusion encoder layers (original: 6)
    decoder_layers: int = 4      # Object decoder layers (original: 6)
    num_queries: int = 100       # Object queries (original: 200)
    num_heads: int = 8           # Attention heads
    dim_feedforward: int = 1024  # FFN hidden dim (original: 2048)
    
    # Geometry Encoder - ~5% of parameters
    geo_encoder_layers: int = 2  # Layers for encoding boxes/points (original: 3)
    
    @property
    def feature_map_size(self) -> int:
        """Number of patches along each dimension."""
        return self.img_size // self.patch_size
    
    @property
    def num_patches(self) -> int:
        """Total number of image patches."""
        return self.feature_map_size ** 2
    
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
        load_text_encoder_weights_with_svd(model, config)  # Load pretrained text encoder with SVD
        load_vit_weights_with_svd(model,config)  # Load pretrained ViT with SVD
    
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
    
    # Add NaN detection for debugging training from scratch
    print("\nAdding NaN detection hooks for debugging...")
    add_nan_hooks(model)
    
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