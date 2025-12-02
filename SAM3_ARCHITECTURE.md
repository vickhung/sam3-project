# SAM3 Model Architecture

A comprehensive breakdown of the SAM3 (Segment Anything Model 3) architecture, showing each layer, its function, and dimensions.

---

## High-Level Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    SAM3 IMAGE MODEL                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                        VISUAL-LANGUAGE BACKBONE (SAM3VLBackbone)                  │   │
│  │  ┌────────────────────────────┐    ┌────────────────────────────────────────┐    │   │
│  │  │    VISION BACKBONE         │    │         TEXT ENCODER                    │    │   │
│  │  │    (ViT + Neck)            │    │      (VETextEncoder)                    │    │   │
│  │  │  Input: 1008×1008×3        │    │   Input: Text Tokens (max 32)           │    │   │
│  │  │  Output: 72×72×256         │    │   Output: Seq×Batch×256                 │    │   │
│  │  └────────────────────────────┘    └────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                            ↓                              ↓                              │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                         GEOMETRY ENCODER                                          │   │
│  │                   (SequenceGeometryEncoder)                                       │   │
│  │               Encodes: Boxes, Points, Masks                                       │   │
│  │            Output: Geometric Features (Seq×Batch×256)                             │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                        ↓                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                     TRANSFORMER ENCODER                                           │   │
│  │                  (TransformerEncoderFusion)                                       │   │
│  │     Fuses: Vision Features + Text Features + Geometric Features                   │   │
│  │                    6 Layers × d_model=256                                         │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                        ↓                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                     TRANSFORMER DECODER                                           │   │
│  │                    (TransformerDecoder)                                           │   │
│  │        200 Object Queries × 6 Layers × d_model=256                                │   │
│  │        Box Refinement + Presence Token                                            │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                        ↓                                                 │
│  ┌─────────────────────────┐   ┌───────────────────────────┐   ┌───────────────────┐   │
│  │   SEGMENTATION HEAD     │   │    DOT PRODUCT SCORING    │   │   BOX HEAD        │   │
│  │ (UniversalSegmentation) │   │   (DotProductScoring)     │   │   (MLP)           │   │
│  │  Pixel Decoder +        │   │   Query-Text Matching     │   │   Box Prediction  │   │
│  │  Mask Predictor         │   │                           │   │                   │   │
│  └─────────────────────────┘   └───────────────────────────┘   └───────────────────┘   │
│                 ↓                           ↓                           ↓               │
│         pred_masks               pred_logits (scores)           pred_boxes              │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Vision Backbone (ViT - Vision Transformer)

**File:** `sam3/model/vitdet.py`  
**Class:** `ViT`  
**Function:** Extracts visual features from input images using a Vision Transformer backbone.

### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `img_size` | 1008 | Input image resolution |
| `pretrain_img_size` | 336 | Pretrained image size |
| `patch_size` | 14 | Size of image patches |
| `embed_dim` | 1024 | Embedding dimension |
| `depth` | 32 | Number of transformer blocks |
| `num_heads` | 16 | Number of attention heads |
| `mlp_ratio` | 4.625 | MLP hidden dimension ratio |
| `window_size` | 24 | Window size for window attention |
| `global_att_blocks` | (7, 15, 23, 31) | Indices of global attention blocks |

### Architecture Breakdown

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              VISION TRANSFORMER (ViT)                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  1. PATCH EMBEDDING (PatchEmbed)                                         │ │
│  │     Input:  (B, 3, 1008, 1008)                                           │ │
│  │     Conv2d: kernel=14×14, stride=14                                      │ │
│  │     Output: (B, 72, 72, 1024)                                            │ │
│  │     ↓ Flattened: (B, 5184, 1024)                                         │ │
│  │     Params: 3 × 14 × 14 × 1024 = 602,112                                 │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  2. POSITIONAL EMBEDDING                                                 │ │
│  │     Pretrain Size: (1, 577, 1024)  [576 patches + 1 cls token]           │ │
│  │     Interpolated/Tiled to match input size                               │ │
│  │     Uses: Absolute + RoPE (Rotary Position Embedding)                    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  3. LN_PRE (LayerNorm)                                                   │ │
│  │     Dimension: 1024                                                      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  4. TRANSFORMER BLOCKS (×32)                                             │ │
│  │                                                                           │ │
│  │  ┌────────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Block Structure:                                                   │  │ │
│  │  │  ├── LayerNorm1 (dim=1024)                                         │  │ │
│  │  │  ├── Attention (Multi-Head Self-Attention)                         │  │ │
│  │  │  │   ├── QKV Linear: (1024 → 3072)                                 │  │ │
│  │  │  │   ├── 16 heads × 64 dim/head                                    │  │ │
│  │  │  │   ├── 2D RoPE (Rotary Position Encoding)                        │  │ │
│  │  │  │   └── Projection: (1024 → 1024)                                 │  │ │
│  │  │  ├── DropPath (rate=0.0-0.1)                                       │  │ │
│  │  │  ├── LayerNorm2 (dim=1024)                                         │  │ │
│  │  │  ├── MLP                                                           │  │ │
│  │  │  │   ├── Linear: (1024 → 4736)  [ratio=4.625]                      │  │ │
│  │  │  │   ├── GELU                                                       │  │ │
│  │  │  │   └── Linear: (4736 → 1024)                                      │  │ │
│  │  │  └── Residual Connections                                           │  │ │
│  │  └────────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                           │ │
│  │  Block Types:                                                             │ │
│  │  • Blocks 0-6, 8-14, 16-22, 24-30: WINDOW Attention (window=24)          │ │
│  │  • Blocks 7, 15, 23, 31: GLOBAL Attention (full attention)               │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  5. LN_POST (LayerNorm) - Applied at last global attention block         │ │
│  │     Dimension: 1024                                                      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  Output: (B, 1024, 72, 72)                                                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Per-Block Parameters
| Component | Parameters |
|-----------|------------|
| LayerNorm1 | 2 × 1024 = 2,048 |
| QKV Linear | 1024 × 3072 + 3072 = 3,148,800 |
| Proj Linear | 1024 × 1024 + 1024 = 1,049,600 |
| LayerNorm2 | 2 × 1024 = 2,048 |
| MLP Linear1 | 1024 × 4736 + 4736 = 4,853,504 |
| MLP Linear2 | 4736 × 1024 + 1024 = 4,849,664 |
| **Total per block** | **~13.9M** |
| **32 Blocks** | **~445M** |

---

## 2. Feature Pyramid Network (Neck)

**File:** `sam3/model/necks.py`  
**Class:** `Sam3DualViTDetNeck`  
**Function:** Creates multi-scale feature pyramid from ViT output for different resolution processing.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          FEATURE PYRAMID NETWORK (Neck)                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Input: ViT Features (B, 1024, 72, 72)                                        │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Scale 4.0× (High Resolution)                                            │ │
│  │  ├── ConvTranspose2d: (1024→512), k=2, s=2  → (B, 512, 144, 144)         │ │
│  │  ├── GELU                                                                 │ │
│  │  ├── ConvTranspose2d: (512→256), k=2, s=2   → (B, 256, 288, 288)         │ │
│  │  ├── Conv2d: (256→256), k=1                  → (B, 256, 288, 288)         │ │
│  │  └── Conv2d: (256→256), k=3, p=1             → (B, 256, 288, 288)         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Scale 2.0× (Medium-High Resolution)                                     │ │
│  │  ├── ConvTranspose2d: (1024→512), k=2, s=2  → (B, 512, 144, 144)         │ │
│  │  ├── Conv2d: (512→256), k=1                  → (B, 256, 144, 144)         │ │
│  │  └── Conv2d: (256→256), k=3, p=1             → (B, 256, 144, 144)         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Scale 1.0× (Original Resolution)                                        │ │
│  │  ├── Conv2d: (1024→256), k=1                 → (B, 256, 72, 72)          │ │
│  │  └── Conv2d: (256→256), k=3, p=1             → (B, 256, 72, 72)          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  Note: Scale 0.5× (Low Resolution) is discarded by scalp=1 setting           │
│                                                                               │
│  Output: List of features at multiple scales, all with 256 channels          │
│  Primary output used: (B, 256, 72, 72) [72×72 = 5184 tokens]                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Text Encoder

**File:** `sam3/model/text_encoder_ve.py`  
**Class:** `VETextEncoder` → `TextTransformer` → `Transformer`  
**Function:** Encodes text prompts into feature representations for vision-language understanding.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              TEXT ENCODER (VETextEncoder)                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  1. TOKENIZER (SimpleTokenizer)                                          │ │
│  │     Vocab Size: 49,408                                                    │ │
│  │     Max Context Length: 32 tokens                                         │ │
│  │     Input: List of strings                                                │ │
│  │     Output: (Batch, 32) token IDs                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  2. TOKEN EMBEDDING                                                       │ │
│  │     Embedding: (49408, 1024)                                              │ │
│  │     Output: (B, 32, 1024)                                                 │ │
│  │     Params: 49,408 × 1024 = 50,593,792                                    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  3. POSITIONAL EMBEDDING                                                  │ │
│  │     Learnable: (32, 1024)                                                 │ │
│  │     Params: 32,768                                                        │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  4. TEXT TRANSFORMER (24 Layers)                                          │ │
│  │                                                                           │ │
│  │  ┌────────────────────────────────────────────────────────────────────┐  │ │
│  │  │  ResidualAttentionBlock (×24)                                       │  │ │
│  │  │  ├── LayerNorm1 (dim=1024)                                         │  │ │
│  │  │  ├── Multi-Head Self-Attention                                      │  │ │
│  │  │  │   ├── 16 heads × 64 dim/head                                     │  │ │
│  │  │  │   ├── Causal Mask (for autoregressive)                           │  │ │
│  │  │  │   └── d_model = 1024                                             │  │ │
│  │  │  ├── LayerScale (optional)                                          │  │ │
│  │  │  ├── LayerNorm2 (dim=1024)                                         │  │ │
│  │  │  ├── MLP                                                            │  │ │
│  │  │  │   ├── Linear: (1024 → 4096)  [ratio=4.0]                         │  │ │
│  │  │  │   ├── GELU                                                        │  │ │
│  │  │  │   └── Linear: (4096 → 1024)                                       │  │ │
│  │  │  └── LayerScale (optional)                                          │  │ │
│  │  └────────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                           │ │
│  │  Per-Layer Params: ~8.4M                                                  │ │
│  │  Total (24 layers): ~200M                                                 │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  5. LN_FINAL (LayerNorm)                                                  │ │
│  │     Dimension: 1024                                                       │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  6. RESIZER (Linear Projection)                                           │ │
│  │     Linear: (1024 → 256)                                                  │ │
│  │     Aligns text features with vision model dimension                       │ │
│  │     Params: 1024 × 256 + 256 = 262,400                                    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  Output: (32, Batch, 256) [Seq-first format]                                  │
│  + Attention Mask: (Batch, 32)                                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Geometry Encoder

**File:** `sam3/model/geometry_encoders.py`  
**Class:** `SequenceGeometryEncoder`  
**Function:** Encodes geometric prompts (boxes, points, masks) into feature representations.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          GEOMETRY ENCODER                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Input Types:                                                                 │
│  • Boxes: (N_boxes, Batch, 4) [cx, cy, w, h normalized 0-1]                  │
│  • Points: (N_points, Batch, 2) [x, y normalized 0-1]                        │
│  • Masks: (N_masks, Batch, 1, H, W)                                          │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  POINT ENCODING (3 methods combined additively)                           │ │
│  │                                                                           │ │
│  │  1. Direct Projection: Linear (2 → 256)                                   │ │
│  │  2. Feature Pooling: grid_sample from backbone + Linear (256 → 256)       │ │
│  │  3. Position Encoding: Sine encoding + Linear (256 → 256)                 │ │
│  │                                                                           │ │
│  │  + Label Embedding: Embedding(6, 256) [pos/neg × point/box_tl/box_br]     │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  BOX ENCODING (3 methods combined additively)                             │ │
│  │                                                                           │ │
│  │  1. Direct Projection: Linear (4 → 256)                                   │ │
│  │  2. Feature Pooling: ROI Align (7×7) + Conv2d (256 → 256)                 │ │
│  │  3. Position Encoding: Sine encoding + Linear (258 → 256)                 │ │
│  │                                                                           │ │
│  │  + Label Embedding: Embedding(2, 256) [positive/negative]                 │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  MASK ENCODING (via FusedMaskEncoder)                                     │ │
│  │                                                                           │ │
│  │  1. MaskDownsampler: Progressive downsampling with Conv layers            │ │
│  │  2. Pixel Feature Projection: Conv2d (256 → 256)                          │ │
│  │  3. Fusion: Add mask + pixel features                                     │ │
│  │  4. Fuser: 2× CXBlock (ConvNeXt-style blocks)                             │ │
│  │  5. Position Encoding: Sine 2D                                            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  CLS TOKEN + PROJECTION                                                   │ │
│  │                                                                           │ │
│  │  • CLS Embedding: Embedding(1, 256)                                       │ │
│  │  • Final Projection: Linear (256 → 256) + LayerNorm                       │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  GEOMETRY TRANSFORMER (3 Layers)                                          │ │
│  │                                                                           │ │
│  │  TransformerEncoderLayer (×3):                                            │ │
│  │  ├── Self-Attention: 8 heads, dim=256, FFN=2048                           │ │
│  │  ├── Cross-Attention to Image Features: 8 heads                           │ │
│  │  ├── Pre-Norm Architecture                                                │ │
│  │  └── Dropout: 0.1                                                         │ │
│  │                                                                           │ │
│  │  Final: LayerNorm (256)                                                   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  Output: (N_geo_tokens, Batch, 256)                                           │
│  + Attention Mask: (Batch, N_geo_tokens)                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Transformer Encoder (Vision-Language Fusion)

**File:** `sam3/model/encoder.py`  
**Class:** `TransformerEncoderFusion`  
**Function:** Fuses visual features with text and geometric prompts.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     TRANSFORMER ENCODER FUSION                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Inputs:                                                                      │
│  • Image Features: (5184, Batch, 256) [72×72 flattened]                       │
│  • Text Features: (32, Batch, 256) [text prompt]                              │
│  • Geometry Features: (N_geo, Batch, 256) [geometric prompts]                 │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  OPTIONAL: Text Pooling + Projection                                      │ │
│  │  Pool text features → Add to image features                               │ │
│  │  Linear (256 → 256)                                                       │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  ENCODER LAYERS (×6)                                                      │ │
│  │                                                                           │ │
│  │  TransformerEncoderLayer:                                                 │ │
│  │  ┌────────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Pre-Norm Architecture                                              │  │ │
│  │  │                                                                     │  │ │
│  │  │  1. Self-Attention (on image features)                              │  │ │
│  │  │     ├── LayerNorm (256)                                             │  │ │
│  │  │     ├── Multi-Head Attention: 8 heads, dim=256                      │  │ │
│  │  │     ├── Q, K with position encoding                                 │  │ │
│  │  │     └── Dropout: 0.1                                                │  │ │
│  │  │                                                                     │  │ │
│  │  │  2. Cross-Attention (image → text+geo prompts)                      │  │ │
│  │  │     ├── LayerNorm (256)                                             │  │ │
│  │  │     ├── Multi-Head Attention: 8 heads, dim=256                      │  │ │
│  │  │     │   Query: image features                                       │  │ │
│  │  │     │   Key/Value: concatenated prompts                             │  │ │
│  │  │     └── Dropout: 0.1                                                │  │ │
│  │  │                                                                     │  │ │
│  │  │  3. Feed-Forward Network                                            │  │ │
│  │  │     ├── LayerNorm (256)                                             │  │ │
│  │  │     ├── Linear: (256 → 2048)                                        │  │ │
│  │  │     ├── ReLU                                                        │  │ │
│  │  │     ├── Dropout: 0.1                                                │  │ │
│  │  │     └── Linear: (2048 → 256)                                        │  │ │
│  │  │                                                                     │  │ │
│  │  │  Residual connections after each sub-layer                          │  │ │
│  │  └────────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                           │ │
│  │  Per-Layer Parameters:                                                    │ │
│  │  • Self-Attn: 256×256×4 + 256×256 = 327,680                              │ │
│  │  • Cross-Attn: 256×256×4 + 256×256 = 327,680                             │ │
│  │  • FFN: 256×2048 + 2048 + 2048×256 + 256 = 1,050,624                     │ │
│  │  • LayerNorms: 3 × 2 × 256 = 1,536                                        │ │
│  │  • Total per layer: ~1.7M                                                 │ │
│  │  • Total (6 layers): ~10.2M                                               │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  Output:                                                                      │
│  • Encoded Memory: (5184, Batch, 256)                                         │
│  • Position Embeddings: (5184, Batch, 256)                                    │
│  • Spatial Shapes: [(72, 72)]                                                 │
│  • Valid Ratios: (Batch, 1, 2)                                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Transformer Decoder

**File:** `sam3/model/decoder.py`  
**Class:** `TransformerDecoder`  
**Function:** Processes object queries to produce detection and segmentation outputs.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          TRANSFORMER DECODER                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Configuration:                                                               │
│  • num_queries: 200                                                           │
│  • d_model: 256                                                               │
│  • num_layers: 6                                                              │
│  • box_refine: True (iterative box refinement)                               │
│  • DAC: True (Divide-and-Conquer training)                                   │
│  • presence_token: True                                                       │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  QUERY INITIALIZATION                                                     │ │
│  │                                                                           │ │
│  │  • Query Embedding: Embedding(200, 256) [learnable]                       │ │
│  │  • Reference Points: Embedding(200, 4) [learnable box coords]             │ │
│  │  • Presence Token: Embedding(1, 256) [scene-level presence]               │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  DECODER LAYERS (×6)                                                      │ │
│  │                                                                           │ │
│  │  TransformerDecoderLayer:                                                 │ │
│  │  ┌────────────────────────────────────────────────────────────────────┐  │ │
│  │  │                                                                     │  │ │
│  │  │  1. Query Position Encoding                                         │  │ │
│  │  │     ├── Sine embedding of reference boxes                           │  │ │
│  │  │     └── MLP: (512 → 256 → 256)                                      │  │ │
│  │  │                                                                     │  │ │
│  │  │  2. Self-Attention (on queries)                                     │  │ │
│  │  │     ├── Multi-Head Attention: 8 heads, dim=256                      │  │ │
│  │  │     ├── Includes presence token                                     │  │ │
│  │  │     ├── DAC: Only first half during training                        │  │ │
│  │  │     └── LayerNorm + Dropout                                         │  │ │
│  │  │                                                                     │  │ │
│  │  │  3. Cross-Attention to Text                                         │  │ │
│  │  │     ├── Multi-Head Attention: 8 heads, dim=256                      │  │ │
│  │  │     ├── Query: decoder queries                                      │  │ │
│  │  │     ├── Key/Value: text features                                    │  │ │
│  │  │     └── LayerNorm + Dropout                                         │  │ │
│  │  │                                                                     │  │ │
│  │  │  4. Cross-Attention to Image (with Box RPB)                         │  │ │
│  │  │     ├── Multi-Head Attention: 8 heads, dim=256                      │  │ │
│  │  │     ├── Query: decoder queries                                      │  │ │
│  │  │     ├── Key/Value: encoder memory                                   │  │ │
│  │  │     ├── Box Relative Position Bias (boxRPB)                         │  │ │
│  │  │     │   └── MLP: (4 → 256 → 8) per spatial direction                │  │ │
│  │  │     └── LayerNorm + Dropout                                         │  │ │
│  │  │                                                                     │  │ │
│  │  │  5. Feed-Forward Network                                            │  │ │
│  │  │     ├── Linear: (256 → 2048)                                        │  │ │
│  │  │     ├── ReLU + Dropout                                              │  │ │
│  │  │     ├── Linear: (2048 → 256)                                        │  │ │
│  │  │     └── LayerNorm                                                   │  │ │
│  │  │                                                                     │  │ │
│  │  │  6. Iterative Box Refinement                                        │  │ │
│  │  │     ├── Box Head MLP: (256 → 256 → 4)                               │  │ │
│  │  │     └── reference_boxes = sigmoid(prev_boxes + delta)               │  │ │
│  │  └────────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                           │ │
│  │  Per-Layer Parameters: ~3.5M                                              │ │
│  │  Total (6 layers): ~21M                                                   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  OUTPUT HEADS                                                             │ │
│  │                                                                           │ │
│  │  • LayerNorm (256)                                                        │ │
│  │  • Box Embed MLP: (256 → 256 → 4) [3 layers]                              │ │
│  │  • Presence Head MLP: (256 → 256 → 1) [3 layers]                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  Outputs per layer:                                                           │
│  • Query Features: (Batch, 200, 256)                                          │
│  • Reference Boxes: (Batch, 200, 4)                                           │
│  • Presence Logits: (Batch, 1)                                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Segmentation Head

**File:** `sam3/model/maskformer_segmentation.py`  
**Class:** `UniversalSegmentationHead`  
**Function:** Generates high-resolution mask predictions from decoder outputs.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         UNIVERSAL SEGMENTATION HEAD                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Inputs:                                                                      │
│  • Backbone Features: [(B,256,288,288), (B,256,144,144), (B,256,72,72)]       │
│  • Object Queries: (6, Batch, 200, 256) [from decoder layers]                 │
│  • Encoder Hidden States: (5184, Batch, 256)                                  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  CROSS-ATTENTION TO PROMPT (Optional)                                     │ │
│  │                                                                           │ │
│  │  • LayerNorm (256)                                                        │ │
│  │  • Multi-Head Attention: 8 heads, dim=256                                 │ │
│  │  • Query: encoder hidden states                                           │ │
│  │  • Key/Value: text/geometric prompts                                      │ │
│  │  • Residual connection                                                    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  PIXEL DECODER (3 Upsampling Stages)                                      │ │
│  │                                                                           │ │
│  │  Input: Multi-scale backbone features + encoder states                    │ │
│  │                                                                           │ │
│  │  Stage 1: (72, 72) → (144, 144)                                           │ │
│  │  ├── Add FPN features                                                     │ │
│  │  ├── Nearest interpolation ×2                                             │ │
│  │  ├── Conv2d (256 → 256), k=3, p=1                                         │ │
│  │  ├── GroupNorm (8 groups)                                                 │ │
│  │  └── ReLU                                                                 │ │
│  │                                                                           │ │
│  │  Stage 2: (144, 144) → (288, 288)                                         │ │
│  │  ├── Same structure as Stage 1                                            │ │
│  │  └── Output: (B, 256, 288, 288)                                           │ │
│  │                                                                           │ │
│  │  Stage 3: (288, 288) → (576, 576)                                         │ │
│  │  ├── Same structure as Stage 1                                            │ │
│  │  └── Output: (B, 256, 576, 576) [or input resolution dependent]           │ │
│  │                                                                           │ │
│  │  Parameters: 3 × (256×256×9 + 256) = ~1.8M                                │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  SEMANTIC SEGMENTATION HEAD                                               │ │
│  │  Conv2d: (256 → 1), k=1                                                   │ │
│  │  Output: (B, 1, H_out, W_out) [background/foreground]                     │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  INSTANCE SEGMENTATION HEAD                                               │ │
│  │  Conv2d: (256 → 256), k=1                                                 │ │
│  │  Output: (B, 256, H_out, W_out) [per-pixel embeddings]                    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  MASK PREDICTOR                                                           │ │
│  │                                                                           │ │
│  │  Mask Embed MLP: (256 → 256 → 256) [3 layers]                             │ │
│  │  Operation: einsum("bqc,bchw->bqhw", mask_embed(queries), pixel_embed)    │ │
│  │                                                                           │ │
│  │  Output: (B, 200, H_out, W_out) [per-query masks]                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  PRESENCE HEAD (Optional)                                                 │ │
│  │                                                                           │ │
│  │  • Pool encoder hidden states                                             │ │
│  │  • Dot Product Scoring with prompt features                               │ │
│  │  Output: (B, 1) [scene presence logit]                                    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  Outputs:                                                                     │
│  • pred_masks: (B, 200, H_out, W_out)                                         │
│  • semantic_seg: (B, 1, H_out, W_out)                                         │
│  • presence_logit: (B, 1)                                                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Scoring Module

**File:** `sam3/model/model_misc.py`  
**Class:** `DotProductScoring`  
**Function:** Computes similarity scores between queries and text prompts.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          DOT PRODUCT SCORING                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Inputs:                                                                      │
│  • Query Features: (6, Batch, 200, 256) [from decoder layers]                 │
│  • Prompt Features: (Seq, Batch, 256) [text + geometric]                      │
│  • Prompt Mask: (Batch, Seq)                                                  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  PROMPT MLP (for prompt projection)                                       │ │
│  │                                                                           │ │
│  │  MLP Structure:                                                           │ │
│  │  ├── Linear: (256 → 2048)                                                 │ │
│  │  ├── ReLU + Dropout (0.1)                                                 │ │
│  │  ├── Linear: (2048 → 256)                                                 │ │
│  │  ├── Residual connection                                                  │ │
│  │  └── LayerNorm (256)                                                      │ │
│  │                                                                           │ │
│  │  Parameters: 256×2048 + 2048 + 2048×256 + 256 = ~1.05M                    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  SCORING COMPUTATION                                                      │ │
│  │                                                                           │ │
│  │  1. Project prompts: prompt_proj = prompt_mlp(prompt)                     │ │
│  │  2. Normalize: query_norm = L2_norm(query)                                │ │
│  │                prompt_norm = L2_norm(prompt_proj)                         │ │
│  │  3. Dot product: scores = query_norm @ prompt_norm.T                      │ │
│  │  4. Scale: scores = scores * d_proj                                       │ │
│  │  5. Mask: Apply prompt mask to invalid positions                          │ │
│  │  6. Pool: Max-pool over prompt dimension                                  │ │
│  │                                                                           │ │
│  │  Output: (6, Batch, 200, 1) [per-layer, per-query scores]                 │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Position Encoding

**File:** `sam3/model/position_encoding.py`  
**Class:** `PositionEmbeddingSine`  
**Function:** Generates sinusoidal position embeddings for spatial features.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          POSITION ENCODING                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Configuration:                                                               │
│  • num_pos_feats: 256 (or 64 for memory backbone)                             │
│  • temperature: 10000                                                         │
│  • normalize: True                                                            │
│  • scale: 2π                                                                  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  2D SINUSOIDAL ENCODING                                                   │ │
│  │                                                                           │ │
│  │  For each (x, y) position:                                                │ │
│  │  1. Normalize: x_norm = x / W, y_norm = y / H                             │ │
│  │  2. Scale: x_scaled = x_norm * 2π, y_scaled = y_norm * 2π                 │ │
│  │  3. Compute frequencies:                                                  │ │
│  │     dim_t = temperature^(2i / num_pos_feats)                              │ │
│  │  4. Encode X:                                                             │ │
│  │     pos_x = [sin(x/dim_t), cos(x/dim_t)]  [interleaved]                   │ │
│  │  5. Encode Y:                                                             │ │
│  │     pos_y = [sin(y/dim_t), cos(y/dim_t)]  [interleaved]                   │ │
│  │  6. Concatenate: pos = [pos_y; pos_x]                                     │ │
│  │                                                                           │ │
│  │  Output: (B, 256, H, W) or (B, 128, H, W) depending on config             │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  BOX ENCODING (for geometry encoder)                                      │ │
│  │                                                                           │ │
│  │  For box (cx, cy, w, h):                                                  │ │
│  │  1. Encode center: pos_cx, pos_cy (same as point encoding)                │ │
│  │  2. Encode size: pos_w = log(w), pos_h = log(h)                           │ │
│  │  3. Concatenate: [pos_cx; pos_cy; pos_w; pos_h]                           │ │
│  │                                                                           │ │
│  │  Output: (N_boxes, Batch, 256+2) → Linear → (N_boxes, Batch, 256)         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Memory Components (for Video/Tracking)

**File:** `sam3/model/memory.py`  
**Classes:** `SimpleMaskDownSampler`, `CXBlock`, `SimpleFuser`, `SimpleMaskEncoder`  
**Function:** Memory encoding for video tracking and temporal propagation.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           MEMORY COMPONENTS                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  MASK DOWNSAMPLER                                                         │ │
│  │                                                                           │ │
│  │  Purpose: Progressively downsample masks for memory encoding              │ │
│  │                                                                           │ │
│  │  Configuration:                                                           │ │
│  │  • kernel_size: 3, stride: 2, padding: 1                                  │ │
│  │  • interpol_size: [1152, 1152] (interpolate input first)                  │ │
│  │  • total_stride: 16 (4 downsampling stages)                               │ │
│  │                                                                           │ │
│  │  Structure (per stage):                                                   │ │
│  │  ├── Conv2d: (in_ch → out_ch×4), k=3, s=2, p=1                            │ │
│  │  ├── LayerNorm2d                                                          │ │
│  │  └── GELU                                                                 │ │
│  │                                                                           │ │
│  │  Final: Conv2d (out_ch → 256), k=1                                        │ │
│  │                                                                           │ │
│  │  Input: (B, 1, H, W) [mask]                                               │ │
│  │  Output: (B, 256, H/16, W/16)                                             │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  CXBlock (ConvNeXt Block)                                                 │ │
│  │                                                                           │ │
│  │  Configuration:                                                           │ │
│  │  • dim: 256                                                               │ │
│  │  • kernel_size: 7, padding: 3                                             │ │
│  │  • layer_scale_init_value: 1e-6                                           │ │
│  │  • use_dwconv: True (depthwise convolution)                               │ │
│  │                                                                           │ │
│  │  Structure:                                                               │ │
│  │  ├── Depthwise Conv2d: (256 → 256), k=7, p=3, groups=256                  │ │
│  │  ├── LayerNorm2d (256)                                                    │ │
│  │  ├── Permute: (N,C,H,W) → (N,H,W,C)                                       │ │
│  │  ├── Linear: (256 → 1024)  [pointwise conv1]                              │ │
│  │  ├── GELU                                                                 │ │
│  │  ├── Linear: (1024 → 256)  [pointwise conv2]                              │ │
│  │  ├── LayerScale (γ=1e-6)                                                  │ │
│  │  ├── Permute: (N,H,W,C) → (N,C,H,W)                                       │ │
│  │  └── Residual connection                                                  │ │
│  │                                                                           │ │
│  │  Parameters per block: ~790K                                              │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  SIMPLE FUSER (2 CXBlocks)                                                │ │
│  │                                                                           │ │
│  │  Structure:                                                               │ │
│  │  ├── Optional input projection: Conv2d (256 → 256), k=1                   │ │
│  │  ├── CXBlock 1                                                            │ │
│  │  └── CXBlock 2                                                            │ │
│  │                                                                           │ │
│  │  Parameters: 2 × 790K ≈ 1.6M                                              │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  SIMPLE MASK ENCODER                                                      │ │
│  │                                                                           │ │
│  │  Configuration:                                                           │ │
│  │  • out_dim: 64 (for memory features)                                      │ │
│  │  • in_dim: 256 (pixel features)                                           │ │
│  │                                                                           │ │
│  │  Structure:                                                               │ │
│  │  1. Mask Downsampler: mask → (B, 256, H/16, W/16)                         │ │
│  │  2. Pixel Feat Projection: Conv2d (256 → 256), k=1                        │ │
│  │  3. Fusion: pix_feat_proj + mask_downsampled                              │ │
│  │  4. Fuser: 2× CXBlock                                                     │ │
│  │  5. Output Projection: Conv2d (256 → 64), k=1                             │ │
│  │  6. Position Encoding: Sine 2D (64 features)                              │ │
│  │                                                                           │ │
│  │  Output: {"vision_features": (B, 64, H', W'),                              │ │
│  │           "vision_pos_enc": [(B, 64, H', W')]}                             │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Parameter Summary

| Component | Parameters | Description |
|-----------|------------|-------------|
| **ViT Backbone** | ~445M | 32-layer Vision Transformer (1024 dim) |
| **Neck (FPN)** | ~5M | Multi-scale feature pyramid |
| **Text Encoder** | ~260M | 24-layer Text Transformer (1024 dim) |
| **Geometry Encoder** | ~8M | 3-layer Transformer + projections |
| **Transformer Encoder** | ~10M | 6-layer fusion encoder (256 dim) |
| **Transformer Decoder** | ~25M | 6-layer decoder with box refinement |
| **Segmentation Head** | ~3M | Pixel decoder + mask predictor |
| **Scoring Module** | ~1M | Dot product scoring MLP |
| **Memory Components** | ~3M | For video tracking (optional) |
| **Total (Approximate)** | **~760M** | Full SAM3 Image Model |

---

## Data Flow Summary

```
                                    INPUT
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
            ┌───────────┐     ┌───────────┐     ┌───────────────┐
            │   Image   │     │   Text    │     │   Geometry    │
            │ 1008×1008 │     │  Prompt   │     │ Boxes/Points  │
            └─────┬─────┘     └─────┬─────┘     └───────┬───────┘
                  │                 │                   │
                  ▼                 ▼                   ▼
            ┌───────────┐     ┌───────────┐     ┌───────────────┐
            │    ViT    │     │   Text    │     │   Geometry    │
            │ Backbone  │     │  Encoder  │     │    Encoder    │
            └─────┬─────┘     └─────┬─────┘     └───────┬───────┘
                  │                 │                   │
                  ▼                 │                   │
            ┌───────────┐           │                   │
            │   Neck    │           │                   │
            │   (FPN)   │           │                   │
            └─────┬─────┘           │                   │
                  │                 │                   │
                  ▼                 ▼                   ▼
            ┌─────────────────────────────────────────────┐
            │           TRANSFORMER ENCODER               │
            │      (Vision-Language-Geometry Fusion)      │
            └─────────────────────┬───────────────────────┘
                                  │
                                  ▼
            ┌─────────────────────────────────────────────┐
            │           TRANSFORMER DECODER               │
            │       (200 Object Queries × 6 Layers)       │
            └─────────────────────┬───────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
            ┌───────────┐  ┌───────────┐  ┌───────────┐
            │   Mask    │  │   Score   │  │    Box    │
            │   Head    │  │   Head    │  │   Head    │
            └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
                  │              │              │
                  ▼              ▼              ▼
            ┌───────────┐  ┌───────────┐  ┌───────────┐
            │pred_masks │  │pred_logits│  │pred_boxes │
            │(B,200,H,W)│  │ (B,200,1) │  │ (B,200,4) │
            └───────────┘  └───────────┘  └───────────┘
```

---

## Key Design Choices

1. **Unified Vision-Language Architecture**: Single backbone processes both modalities, enabling grounded understanding.

2. **Multi-Scale Features**: FPN-style neck provides features at multiple resolutions for precise segmentation.

3. **Query-Based Detection**: 200 learnable object queries enable open-vocabulary detection and segmentation.

4. **Iterative Box Refinement**: Each decoder layer refines box predictions progressively.

5. **DAC (Divide-and-Conquer)**: Training strategy that handles one-to-one and one-to-many matching.

6. **Box Relative Position Bias**: Spatial attention is guided by predicted box locations.

7. **Presence Token**: Global scene-level signal indicating if target objects exist.

8. **Dot Product Scoring**: Efficient query-text similarity computation for open-vocabulary classification.

---

*Generated from SAM3 source code analysis*

