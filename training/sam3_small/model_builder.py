"""
SAM3-Small Model Builder

A compact SAM3 model (~250M params) designed for:
- Training on consumer GPUs (8-16GB VRAM)
- Edge deployment (Jetson, mobile)
- Fast iteration during research

Author: [Your Name]
Date: December 2024
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

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