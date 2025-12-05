# Remote GPU Access Guide

## Remote Machine Details

| Property | Value |
|----------|-------|
| **Hostname** | bu10d2-laptop |
| **IP Address** | 172.16.0.53 |
| **SSH User** | bu10d2 |
| **GPU** | NVIDIA GeForce RTX 5080 Laptop GPU |
| **VRAM** | 16 GB |
| **CUDA Version** | 12.8 |
| **Python Environment** | ~/sam3-env |
| **Project Path** | ~/vk/sam3-project |

---

## Quick Connection

```bash
# SSH to remote machine
ssh bu10d2@172.16.0.53

# Check GPU status
ssh bu10d2@172.16.0.53 "nvidia-smi"
```

---

## Environment Setup

### Activate Python Environment
```bash
ssh bu10d2@172.16.0.53 "source ~/sam3-env/bin/activate && python --version"
```

### PyTorch Version (Blackwell-compatible nightly)
```bash
ssh bu10d2@172.16.0.53 "source ~/sam3-env/bin/activate && python -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}\")'"
```

Expected output:
```
PyTorch: 2.10.0.dev20251203+cu128, CUDA: 12.8
```

---

## File Transfer

### Sync files TO remote
```bash
# Sync entire project (excluding venv, cache, data)
rsync -avz --progress \
  --exclude 'venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude 'experiments' \
  --exclude 'data' \
  ./ bu10d2@172.16.0.53:~/vk/sam3-project/

# Sync single file
scp local_file.py bu10d2@172.16.0.53:~/vk/sam3-project/
```

### Sync files FROM remote
```bash
# Copy results back
scp bu10d2@172.16.0.53:~/vk/sam3-project/experiments/results.json ./

# Sync entire experiments folder
rsync -avz bu10d2@172.16.0.53:~/vk/sam3-project/experiments/ ./experiments/
```

---

## Running Training

### Start Training in Background
```bash
ssh bu10d2@172.16.0.53 "source ~/sam3-env/bin/activate && cd ~/vk/sam3-project && nohup python train_remote.py > training.log 2>&1 &"
```

### Monitor Training Log
```bash
# Tail the log
ssh bu10d2@172.16.0.53 "tail -f ~/vk/sam3-project/training.log"

# Check last 100 lines
ssh bu10d2@172.16.0.53 "tail -100 ~/vk/sam3-project/training.log"
```

### Check GPU Utilization
```bash
ssh bu10d2@172.16.0.53 "watch -n 1 nvidia-smi"
```

### Kill Training Process
```bash
ssh bu10d2@172.16.0.53 "pkill -f train_remote.py"
```

---

## Running Commands Remotely

### One-liner execution
```bash
ssh bu10d2@172.16.0.53 "source ~/sam3-env/bin/activate && cd ~/vk/sam3-project && python -c 'print(\"Hello from 5080!\")'"
```

### Interactive session
```bash
ssh bu10d2@172.16.0.53
# Then manually:
source ~/sam3-env/bin/activate
cd ~/vk/sam3-project
python your_script.py
```

---

## Data Locations on Remote

| Dataset | Path |
|---------|------|
| COCO val2017 | ~/vk/sam3-project/data/coco/val2017 |
| COCO annotations | ~/vk/sam3-project/data/coco/annotations |
| Aquarium | ~/vk/sam3-project/data/aquarium |

---

## Installed Packages

The `sam3-env` has these key packages:
- PyTorch 2.10.0.dev (nightly with CUDA 12.8 for Blackwell support)
- torchvision 0.25.0.dev
- hydra-core, omegaconf
- scipy, opencv-python-headless
- decord, pycocotools
- tensorboard

### Install additional packages
```bash
ssh bu10d2@172.16.0.53 "source ~/sam3-env/bin/activate && pip install package_name"
```

---

## Troubleshooting

### Check if training is running
```bash
ssh bu10d2@172.16.0.53 "ps aux | grep python | grep -v grep"
```

### Check GPU memory usage
```bash
ssh bu10d2@172.16.0.53 "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"
```

### Clear Python cache
```bash
ssh bu10d2@172.16.0.53 "cd ~/vk/sam3-project && find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null"
```

### Check disk space
```bash
ssh bu10d2@172.16.0.53 "df -h ~"
```

---

## Network Information

- Local machine: 172.16.0.23
- Remote machine: 172.16.0.53
- SSH port: 22 (default)

Both machines are on the same local network (172.16.0.x).

