# SAM3 Project

Training and deploying SAM3 (Segment Anything Model 3) for edge devices.

---

## Quick Links

| Document | Description |
|----------|-------------|
| [TRAINING_JOURNEY.md](TRAINING_JOURNEY.md) | Full learning journey with concepts, errors, and solutions |
| [SAM3_ARCHITECTURE.md](SAM3_ARCHITECTURE.md) | Detailed SAM3 architecture breakdown |
| [PACKAGING_WEIGHTS.md](PACKAGING_WEIGHTS.md) | Guide for packaging model weights |

---

## Project Structure

```
sam3-project/
│
├── sam3/                          # Original SAM3 from Meta (submodule)
│   ├── sam3/                      # SAM3 Python package
│   │   ├── model/                 # Model architecture files
│   │   ├── train/                 # Training scripts and configs
│   │   └── eval/                  # Evaluation scripts
│   ├── examples/                  # Jupyter notebooks
│   └── assets/                    # Images, videos, BPE vocab
│
├── training/                      # Our custom compact model work
│   ├── sam3_small/                # SAM3-Small model builder
│   │   ├── __init__.py            # Package definition
│   │   └── model_builder.py       # Model construction (1200+ lines)
│   └── configs/
│       └── sam3_small_train.yaml  # Training configuration
│
├── data/                          # Datasets (gitignored)
│   └── aquarium/                  # Roboflow Aquarium dataset
│       ├── train/                 # 448 images
│       ├── valid/                 # 127 images
│       └── test/                  # 63 images
│
├── experiments/                   # Training outputs (gitignored)
│   └── sam3_small_aquarium/       # Experiment results
│       ├── checkpoints/           # Model checkpoints
│       ├── logs/                  # Training logs
│       └── tensorboard/           # TensorBoard events
│
├── venv/                          # Python virtual environment
│
├── TRAINING_JOURNEY.md            # Learning journey documentation
├── SAM3_ARCHITECTURE.md           # Architecture breakdown
├── PACKAGING_WEIGHTS.md           # Weight packaging guide
└── README.md                      # This file
```

---

## Setup

### 1. Clone the Repository

```bash
git clone git@github.com:vickhung/sam3-project.git
cd sam3-project
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install SAM3 with training dependencies
pip install -e "sam3/[train]"

# Or just core dependencies
pip install -e sam3/
```

### 4. Download Dataset

Download from [Roboflow Aquarium](https://universe.roboflow.com/brad-dwyer/aquarium-combined) in COCO format and extract to `data/aquarium/`.

---

## Training SAM3-Small

### Quick Start

```bash
# Activate environment
source venv/bin/activate

# Set Python path
export PYTHONPATH="$PWD:$PYTHONPATH"

# Copy config to SAM3's config directory
cp training/configs/sam3_small_train.yaml sam3/sam3/train/configs/

# Run training
python sam3/sam3/train/train.py \
    -c sam3_small_train.yaml \
    --use-cluster 0 \
    --num-gpus 1
```

### Monitor Training

```bash
# TensorBoard
tensorboard --logdir experiments/sam3_small_aquarium/tensorboard

# GPU usage
watch -n 1 nvidia-smi
```

### Training Logs

Logs are saved to `experiments/sam3_small_aquarium/logs/`:
- `log.txt` - Full training log
- `train_stats.json` - Training metrics
- `val_stats.json` - Validation metrics

---

## Key Files

### Model Builder

**Location:** `training/sam3_small/model_builder.py`

Contains:
- `SAM3SmallConfig` - Hyperparameter dataclass
- `build_sam3_small()` - Main model construction
- `initialize_weights()` - Weight initialization
- `project_weight_svd()` - SVD weight projection
- `add_nan_hooks()` - Debug utility for NaN detection

### Training Config

**Location:** `training/configs/sam3_small_train.yaml`

Key settings:
```yaml
scratch:
  resolution: 252          # Input resolution
  train_batch_size: 2      # Batch size
  lr_scale: 0.005          # Learning rate multiplier

trainer:
  max_epochs: 50           # Training epochs
  val_epoch_freq: 5        # Validation frequency
```

---

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Stable release |
| `jetson-orin` | Jetson deployment work |
| `feature/sam3-small-training` | SAM3-Small development |

---

## Hardware Requirements

### Training (RTX 4070 Laptop)
- GPU: 8GB VRAM minimum
- RAM: 16GB recommended
- Storage: 10GB for dataset + models

### Deployment (Jetson Orin Nano)
- GPU: Ampere with 8GB shared memory
- Recommended: TensorRT FP16 optimization

---

## Current Status

| Component | Status |
|-----------|--------|
| SAM3-Small architecture | ✅ Complete |
| Training pipeline | ✅ Working |
| SVD weight transfer | ⚠️ Implemented, not effective |
| Model performance (AP > 0) | ❌ In progress |

**Next Steps:** Width & Depth Pruning or use full SAM3 with TensorRT

See [TRAINING_JOURNEY.md](TRAINING_JOURNEY.md) for full details.

---

## License

SAM3 is licensed under Apache 2.0. See `sam3/LICENSE` for details.
