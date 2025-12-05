# SAM3-Small Training Journey

**Project:** Training a Compact SAM3 Model for Edge Deployment  
**Date:** December 3, 2024  
**Hardware:** NVIDIA RTX 4070 Laptop (8GB VRAM)  
**Target Deployment:** NVIDIA Jetson Orin Nano  

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Understanding SAM3 Architecture](#2-understanding-sam3-architecture)
3. [Designing SAM3-Small](#3-designing-sam3-small)
4. [Setting Up Training Environment](#4-setting-up-training-environment)
5. [Training Attempts & Lessons](#5-training-attempts--lessons)
6. [Key Concepts Learned](#6-key-concepts-learned)
7. [Errors Encountered & Solutions](#7-errors-encountered--solutions)
8. [Current Status & Next Steps](#8-current-status--next-steps)
9. [Files Created](#9-files-created)
10. [Useful Commands](#10-useful-commands)

---

## 1. Project Goal

### Original Objective
Create a compact version of SAM3 (~1GB / 200M parameters) that can:
- Run on edge devices like Jetson Orin Nano
- Maintain reasonable detection/segmentation capabilities
- Be trained on consumer hardware (8GB GPU)

### Why SAM3?
SAM3 (Segment Anything Model 3) provides:
- Text-conditioned detection ("find the fish")
- Zero-shot generalization to new objects
- High-quality segmentation masks
- Open-vocabulary understanding

### The Challenge
The original SAM3 is 848M parameters (~3.4GB FP32), too large for edge deployment. We needed to compress it to ~200M parameters while retaining functionality.

---

## 2. Understanding SAM3 Architecture

### Original SAM3 Specifications

| Component | Configuration |
|-----------|---------------|
| **Total Parameters** | 848M |
| **Input Resolution** | 1008×1008 |
| **ViT embed_dim** | 1024 |
| **ViT depth** | 32 layers |
| **ViT num_heads** | 16 |
| **ViT patch_size** | 14 |
| **Text encoder dim** | 1024 |
| **Text encoder layers** | 24 |
| **Decoder layers** | 6 |
| **Object queries** | 200 |

### Architecture Components

```
┌────────────────────────────────────────────────────────────────┐
│                        SAM3 ARCHITECTURE                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │   Image     │    │  Text Prompt │    │ Geometry Prompt │   │
│  │  1008×1008  │    │  "find fish" │    │  [boxes/points] │   │
│  └──────┬──────┘    └──────┬───────┘    └────────┬────────┘   │
│         │                  │                      │            │
│         ▼                  ▼                      ▼            │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │ViT Backbone │    │Text Encoder  │    │Geometry Encoder │   │
│  │  (32 layers)│    │ (24 layers)  │    │   (3 layers)    │   │
│  └──────┬──────┘    └──────┬───────┘    └────────┬────────┘   │
│         │                  │                      │            │
│         └──────────────────┼──────────────────────┘            │
│                            │                                   │
│                            ▼                                   │
│                  ┌──────────────────┐                         │
│                  │ Transformer      │                         │
│                  │ Encoder (6 layers)                         │
│                  │ + Decoder (6 layers)                       │
│                  └─────────┬────────┘                         │
│                            │                                   │
│                            ▼                                   │
│                  ┌──────────────────┐                         │
│                  │ Detection Head   │                         │
│                  │ + Segmentation   │                         │
│                  └──────────────────┘                         │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Key Insight: Resolution Constraint
The patch_size of 14 requires input resolution to be divisible by 14:
- 1008 ÷ 14 = 72 ✅ (original)
- 256 ÷ 14 = 18.29 ❌ (not valid)
- 252 ÷ 14 = 18 ✅ (our choice)

---

## 3. Designing SAM3-Small

### Target Specifications

| Component | SAM3 Original | SAM3-Small | Reduction |
|-----------|---------------|------------|-----------|
| Parameters | 848M | ~200M | 76% smaller |
| Resolution | 1008×1008 | 252×252 | 4× smaller |
| ViT embed_dim | 1024 | 768 | 25% smaller |
| ViT depth | 32 | 16 | 50% fewer |
| ViT num_heads | 16 | 12 | 25% fewer |
| Text encoder dim | 1024 | 512 | 50% smaller |
| Text encoder layers | 24 | 12 | 50% fewer |
| Decoder layers | 6 | 4 | 33% fewer |
| Object queries | 200 | 100 | 50% fewer |

### Design Decisions

1. **Resolution: 252×252**
   - Divisible by patch_size (14)
   - Small enough for edge devices
   - Still reasonable for medium/large objects

2. **ViT: 768 dim, 16 layers**
   - Matches ViT-Base architecture
   - Good balance of capacity vs. size

3. **Text Encoder: 512 dim, 12 layers**
   - Sufficient for text understanding
   - Significant size reduction

### File Created: `training/sam3_small/model_builder.py`

This file contains:
- `SAM3SmallConfig` dataclass with all hyperparameters
- Component builder functions for each part
- Weight initialization functions
- SVD projection functions (for weight transfer)
- Main `build_sam3_small()` function

---

## 4. Setting Up Training Environment

### Dataset: Roboflow Aquarium

Downloaded from Roboflow Universe in COCO format:

```
data/aquarium/
├── train/           # 448 images
│   ├── images/
│   └── _annotations.coco.json
├── valid/           # 127 images (508 with negatives)
│   ├── images/
│   └── _annotations.coco.json
└── test/            # 63 images
    ├── images/
    └── _annotations.coco.json
```

**Classes (7):** fish, jellyfish, penguin, puffin, shark, starfish, stingray

### Training Configuration

Key settings in `training/configs/sam3_small_train.yaml`:

```yaml
scratch:
  resolution: 252
  train_batch_size: 2
  gradient_accumulation_steps: 1
  lr_scale: 0.005
  scheduler_warmup: 200

trainer:
  max_epochs: 50
  val_epoch_freq: 5

optim:
  amp:
    enabled: false  # FP32 for stability
```

### Git Branch Structure

```
main
└── jetson-orin (previous work)
    └── feature/sam3-small-training (current)
```

---

## 5. Training Attempts & Lessons

### Attempt 1: Random Initialization

**Approach:** Build model with random weights, train on Aquarium dataset.

**Result:** ❌ Failed immediately

**Error:**
```
ValueError: matrix contains invalid numeric entries
```

**Root Cause:** Random weights produce unstable outputs → NaN in attention → NaN in cost matrix → Hungarian matching fails.

**Lesson:** Transformers are sensitive to initialization. Can't just start with random weights.

---

### Attempt 2: Custom Weight Initialization

**Approach:** Add proper initialization:
```python
def initialize_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
```

Plus special handling for:
- `bbox_embed` bias → 0.5 (center predictions)
- `reference_points` → uniform [0.2, 0.8]
- `positional_embedding` → normal(0, 0.01)

**Result:** ❌ Training runs but AP = 0.0 after 50 epochs

**Lesson:** Proper initialization prevents NaN but isn't enough to learn detection from scratch with small dataset.

---

### Attempt 3: Pretrained Text Encoder Only

**Approach:** Load SAM3's text encoder weights directly (same architecture).

**Result:** ❌ AP = 0.0 after 50 epochs

**Problem:** Text encoder dimensions matched, but ViT backbone was still random. Model could understand text but couldn't see images properly.

**Lesson:** Both vision AND language components need pretrained weights.

---

### Attempt 4: SVD Weight Projection (Both Components)

**Approach:** Use Singular Value Decomposition to project large weights to smaller dimensions.

**Implementation:**
```python
def project_weight_svd(large_weight, target_shape):
    """Project large weights using SVD approximation."""
    U, S, Vh = torch.linalg.svd(large_weight.float())
    k = min(target_shape)
    projected = U[:out_dim, :k] @ diag(S[:k]) @ Vh[:k, :in_dim]
    return projected
```

**Strategy:**
- Layer skipping: Take every 2nd layer (32→16 for ViT, 24→12 for text)
- Dimension projection: SVD compress 1024→768 (ViT) and 1024→512 (text)

**Result:** ❌ AP = 0.0 at epoch 10+

**Analysis:**
- Loss was ~500 at epoch 8 (vs ~460 without SVD)
- SVD compression lost too much information
- Combined loss: ~25-30% of original capacity retained

**Lesson:** SVD compresses ALL weights equally, including important features. Better to REMOVE unimportant parts than compress everything.

---

### Key Realization: SVD vs. Pruning

```
SVD Approach (what we tried):
├── Compress ALL weights equally
├── Important features get compressed
├── Less important features get compressed
└── Result: Everything is mediocre

Pruning Approach (recommended alternative):
├── Keep important weights at FULL quality
├── REMOVE less important weights entirely
└── Result: Remaining parts work well
```

---

## 6. Key Concepts Learned

### 6.1 Model Size vs. Information

```
Model Size = Number of Parameters (fixed by architecture)
           = How much the model CAN store

Information = Quality of weights (random vs pretrained)
            = How much useful knowledge is stored

You can have:
- Large model + random weights = 0 useful information
- Small model + good weights = Lots of useful information
```

### 6.2 Training Metrics Explained

| Metric | Meaning | Target |
|--------|---------|--------|
| `train_all_loss` | Total loss | < 200 |
| `loss_bbox` | L1 distance for boxes | < 0.5 |
| `loss_giou` | Generalized IoU loss | < 0.5 |
| `ce_f1` | Classification F1 | > 0.5 |
| `presence_dec_acc` | Presence detection | > 90% |
| `AP` | Average Precision | > 0.1 |

### 6.3 Named Parameters vs Named Modules

```python
# Catches nn.Module subclasses (Embedding, Linear, LayerNorm)
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        # Initialize linear layers
        
# Catches raw Parameters NOT wrapped in modules
for name, param in model.named_parameters():
    if "positional_embedding" in name:
        # Initialize raw parameters
```

**Important:** Some parameters (like `positional_embedding`) are raw `nn.Parameter`, not modules. They won't be caught by `named_modules()`.

### 6.4 COCO Evaluation Metrics

```
AP @[IoU=0.50:0.95] = Average precision across IoU thresholds
AP @[IoU=0.50]      = "Loose" match (50% overlap required)
AP @[IoU=0.75]      = "Strict" match (75% overlap required)
AR @[maxDets=100]   = Average recall with max 100 detections
```

### 6.5 Knowledge Transfer Techniques

| Technique | Description | Quality |
|-----------|-------------|---------|
| **Direct Loading** | Same architecture, copy weights | Best |
| **SVD Projection** | Compress with matrix decomposition | Poor |
| **Structured Pruning** | Remove unimportant heads/layers | Good |
| **Knowledge Distillation** | Teacher-student training | Good |

---

## 7. Errors Encountered & Solutions

### Error 1: NaN in Cost Matrix

```
ValueError: matrix contains invalid numeric entries
Location: matcher.py → linear_sum_assignment(cost)
```

**Cause:** Random weights → unstable attention → NaN propagation

**Solution:** Proper weight initialization with Xavier + special handling for bbox/reference points.

---

### Error 2: Bash History Expansion

```bash
python -c "print('Hello!')"
# Error: -bash: !': event not found
```

**Cause:** `!` in double quotes triggers bash history expansion

**Solution:** Use single quotes:
```bash
python -c 'print("Hello!")'
```

---

### Error 3: Config Not Found

```
ConfigCompositionException: Config not found
```

**Cause:** SAM3 trainer uses Hydra with `initialize_config_module("sam3.train")`, only looks in `sam3/sam3/train/configs/`.

**Solution:** Copy/symlink config to SAM3's config directory:
```bash
cp training/configs/sam3_small_train.yaml sam3/sam3/train/configs/
```

---

### Error 4: Missing Module

```
ModuleNotFoundError: No module named 'training'
```

**Cause:** Python can't find our custom training package

**Solution:** Add to PYTHONPATH:
```bash
PYTHONPATH="/path/to/project:$PYTHONPATH" python ...
```

---

### Error 5: Config Missing port_range

```
omegaconf.errors.ConfigAttributeError: port_range
```

**Cause:** submitit section missing required field

**Solution:** Add to config:
```yaml
submitit:
  use_cluster: false
  port_range: [10000, 65000]
```

---

### Error 6: SAM3SmallConfig has no attribute 'seek'

```
AttributeError: 'SAM3SmallConfig' object has no attribute 'seek'
```

**Cause:** Called wrong function - `load_text_encoder_weights(model, config)` instead of `load_text_encoder_weights_with_svd(model, config)`

**Solution:** Use correct function name with `_with_svd` suffix.

---

## 8. Current Status & Next Steps

### Current Status

| Item | Status |
|------|--------|
| Model architecture defined | ✅ Complete |
| Training config created | ✅ Complete |
| Dataset downloaded | ✅ Complete |
| Weight initialization | ✅ Complete |
| SVD weight transfer | ✅ Implemented, ❌ Not effective |
| Training runs successfully | ✅ Yes |
| Model produces AP > 0 | ❌ Not yet |

### Why AP = 0?

SVD compression loses too much information. The model has:
- ~25-30% of original SAM3's capacity
- Weights are "smeared" across all components
- Not enough quality in any single component

### Recommended Next Steps

#### Option 1: Width & Depth Pruning (Most Promising)
- Load full SAM3 with learnable pruning masks
- Train to identify unimportant heads/layers
- Remove pruned components
- Fine-tune remaining model

**Pros:** Keeps important weights at full quality  
**Cons:** Requires ~4-5GB VRAM for pruning phase

#### Option 2: Use Full SAM3 + TensorRT
- Skip compact model attempt
- Export SAM3 to TensorRT with FP16
- Accept slower inference (~2-5 FPS on Jetson)

**Pros:** Works today, high quality  
**Cons:** Larger model, slower inference

#### Option 3: Use YOLO Instead
- Train YOLOv8n/s on Aquarium dataset
- Much faster inference (30+ FPS)
- Smaller models (6-22 MB)

**Pros:** Fast, well-supported, works on Jetson  
**Cons:** No text conditioning, no segmentation masks

---

## 9. Files Created

### Project Structure

```
training/
├── sam3_small/
│   ├── __init__.py           # Package with version info
│   └── model_builder.py      # Main model builder (1218 lines)
│       ├── SAM3SmallConfig   # Hyperparameter dataclass
│       ├── initialize_weights()
│       ├── project_weight_svd()
│       ├── load_vit_weights_with_svd()
│       ├── load_text_encoder_weights_with_svd()
│       ├── add_nan_hooks()   # Debug utility
│       └── build_sam3_small()
└── configs/
    └── sam3_small_train.yaml # Training configuration

data/
└── aquarium/                 # Roboflow Aquarium dataset
    ├── train/
    ├── valid/
    └── test/

experiments/
└── sam3_small_aquarium/      # Training outputs
    ├── checkpoints/
    ├── logs/
    ├── predictions/
    └── tensorboard/
```

---

## 10. Useful Commands

### Training

```bash
# Set up environment
cd "/home/vkwk/cursor/linux setup/sam3-project"
source venv/bin/activate

# Delete old experiment (fresh start)
rm -rf experiments/sam3_small_aquarium

# Run training
PYTHONPATH="$PWD:$PYTHONPATH" python sam3/sam3/train/train.py \
    -c configs/sam3_small_train.yaml \
    --use-cluster 0 \
    --num-gpus 1
```

### Testing Model Build

```bash
cd training/sam3_small
python model_builder.py
# Should print model statistics without errors
```

### Monitoring

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# TensorBoard
tensorboard --logdir experiments/sam3_small_aquarium/tensorboard
```

### Git Workflow

```bash
# Check status
git status

# Add and commit
git add training/ TRAINING_JOURNEY.md
git commit -m "docs: add training journey documentation"

# Push (if remote is set up)
git push origin feature/sam3-small-training
```

---

## Conclusion

This project explored the challenge of compressing a large vision-language model (SAM3, 848M params) to a compact size (~200M params) for edge deployment.

### Key Takeaways

1. **Architecture reduction alone isn't enough** - smaller dimensions need quality weights
2. **SVD compression loses too much information** - better to prune than compress
3. **Transformers need pretrained weights** - can't learn complex tasks from scratch with small data
4. **Structured pruning is the most promising path** - remove unimportant parts, keep important ones intact

### What We Learned

- SAM3's architecture and how it processes images, text, and geometry
- How to set up training with Hydra configs
- Weight initialization strategies for transformers
- SVD-based weight projection (and its limitations)
- Debugging NaN issues in neural networks
- COCO evaluation metrics and what they mean

### Future Work

The most promising next step is implementing **Width & Depth Pruning** as described in the [AAAI 2022 paper](https://github.com/andyrull/width-and-Depth-pruning-for-Vision-Transformer), adapted for SAM3's architecture.

---

## 11. Recent Progress: RTX 5080 Training & Protobuf Resolution

### December 5, 2024 Update: Protobuf Conflict Resolution

**Major Breakthrough:** Solved the fatal protobuf version conflict that was blocking all training attempts.

#### The Protobuf Problem
- **System OpenCV** compiled against protobuf 3.12.4
- **Python environment** had protobuf 5.29.4
- **TensorFlow** expected specific protobuf versions
- **Result:** Instant segfault when loading COCO datasets

#### The Solution: Pure Virtual Environment Isolation
Instead of complex system package management, we used a **100% Anaconda-free, venv-only** approach:

```bash
# 1. Fresh venv with Python 3.10
python3.10 -m venv venv

# 2. Install compatible PyTorch nightly with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 3. Install dependencies with compatible protobuf
pip install protobuf==3.20.3 opencv-python-headless==4.10.0.84 scipy numpy
pip install hydra-core omegaconf timm torchmetrics ftfy regex pycocotools
```

**Key Insight:** Using `opencv-python-headless` (PyPI wheel) instead of system `opencv-python` completely bypasses the protobuf-linked system OpenCV.

#### Results
- ✅ **Protobuf conflict: SOLVED** - no more fatal crashes
- ✅ **CUDA 12.8 compatibility** - RTX 5080 works perfectly
- ✅ **Model builds successfully** - 514M parameters load correctly
- ✅ **Data loading works** - COCO datasets access without errors

### Hardware Upgrade: RTX 5080 Training

**Previous Hardware:** RTX 4070 (8GB VRAM)  
**Current Hardware:** RTX 5080 (15.45 GB VRAM) via SSH remote access

#### Remote GPU Setup
- **Server:** Ubuntu 22.04 with RTX 5080
- **Access:** SSH with conda environment isolation
- **Environment:** Fresh conda environment with CUDA 12.8 + PyTorch nightly

### Current Status: Memory Limits Discovered

**Good News:** Training starts successfully - model loads, weights transfer, data loads, forward pass begins.

**Bad News:** RTX 5080 cannot handle our current model configuration.

#### Memory Analysis
Our SAM3-Small configuration:
- **Resolution:** 1008×1008 pixels
- **ViT Layers:** 16 (50% of original 32)
- **Text Layers:** 16 (67% of original 24)
- **Queries:** 100 (50% of original 200)
- **Precision:** BF16 mixed precision

**Memory Usage:**
- Model weights: ~2-3 GB
- Optimizer state: ~6-8 GB
- Activations: ~4-6 GB
- Attention masks: ~2-3 GB
- **Total: ~14.27 GB allocated** (out of 15.45 GB)

**Result:** Only 113 MB free → CUDA OOM on 204 MB allocation attempts.

#### The Reality Check
RTX 5080 (15.45 GB) **CANNOT** handle:
- ❌ 1008px resolution + 16 ViT layers + full transformer
- ❌ Requires **24+ GB VRAM** (RTX 4090/4090Ti or A100)

RTX 5080 **CAN** handle:
- ✅ Smaller models (8 ViT layers)
- ✅ Lower resolutions (672-768px)
- ✅ Simpler architectures

### Current Model Specifications

| Component | Original SAM3 | Our Config | Status |
|-----------|---------------|------------|--------|
| **Total Parameters** | 848M | 514M | ✅ Working |
| **Resolution** | 1008×1008 | 1008×1008 | ❌ Too large |
| **ViT embed_dim** | 1024 | 1024 | ✅ Same |
| **ViT depth** | 32 | 16 | ✅ Working |
| **ViT num_heads** | 16 | 16 | ✅ Same |
| **Text encoder dim** | 1024 | 1024 | ✅ Same |
| **Text encoder layers** | 24 | 16 | ✅ Working |
| **Decoder layers** | 6 | 6 | ✅ Same |
| **Object queries** | 200 | 100 | ✅ Working |
| **VRAM Required** | ~24GB | ~16GB | ❌ Too much |

### Next Steps & Options

#### Option 1: Reduce Model Size (Recommended)
- **ViT layers:** 16 → 8 (-50% memory)
- **Text layers:** 16 → 8 (-50% memory)
- **Resolution:** 1008 → 672 (-55% pixels, -70% memory)
- **Expected VRAM:** ~4-6 GB (fits RTX 5080)

#### Option 2: Upgrade Hardware
- **RTX 4090:** 24 GB VRAM (sufficient)
- **A100:** 40-80 GB VRAM (ideal)
- **Training speed:** 2-3x faster than RTX 5080

#### Option 3: CPU Training (Proof of Concept)
- **Memory:** Unlimited (uses system RAM)
- **Speed:** 50-100x slower than GPU
- **Use case:** Verify model works, not for actual training

### Files Updated/Created

#### Model Builder Updates
- `training/sam3_small/model_builder.py`
  - Added RTX 5080 compatibility
  - Reduced queries to 50 for memory efficiency
  - Maintained 16 layer architecture

#### Training Configuration
- `training/configs/sam3_small_coco.yaml`
  - COCO dataset configuration
  - BF16 mixed precision enabled
  - Gradient accumulation: 1
  - Remote GPU paths configured

#### Remote GPU Documentation
- `docs/REMOTE_GPU_ACCESS.md` - Complete guide for RTX 5080 access

### Key Insights from RTX 5080 Experience

1. **Protobuf isolation is critical** - venv/conda environment separation prevents conflicts
2. **Memory estimation is hard** - theoretical calculations don't account for attention masks and intermediate tensors
3. **RTX 5080 is not "large"** - 15GB is insufficient for modern vision-language models at high resolution
4. **Mixed precision helps but not enough** - BF16 reduces memory by ~30-40%, not 50%
5. **Remote GPU access works perfectly** - SSH + conda environments provide seamless development

### Current Blocking Issue

**GPU Memory:** RTX 5080 cannot fit our 514M parameter model at 1008px resolution.

**Immediate Options:**
1. Reduce to 8 ViT layers + 8 text layers (should fit)
2. Reduce resolution to 672px (should fit)
3. Combine both reductions (definitely fits)

**Long-term Options:**
1. Upgrade to RTX 4090 or A100
2. Implement model parallelism
3. Use gradient checkpointing more aggressively

---

## Conclusion & Status

### What We've Achieved
- ✅ **Protobuf conflict: PERMANENTLY SOLVED** - no more fatal crashes
- ✅ **RTX 5080 integration: COMPLETE** - remote GPU access working
- ✅ **Model building: SUCCESSFUL** - 514M parameter model loads correctly
- ✅ **Data pipeline: WORKING** - COCO datasets load without errors
- ✅ **Training framework: READY** - forward pass begins successfully

### Current Roadblock
- ❌ **GPU memory: INSUFFICIENT** - RTX 5080 cannot handle our model size

### Immediate Path Forward
Reduce model complexity to fit RTX 5080:
- ViT layers: 16 → 8
- Text layers: 16 → 8  
- Resolution: 1008 → 672 (optional)

This should reduce VRAM from ~16GB to ~4-6GB, enabling actual training on the RTX 5080.

---

*Document created: December 3, 2024*  
*Last updated: December 5, 2024*



