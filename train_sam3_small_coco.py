#!/usr/bin/env python3
"""
Train SAM3-Small V2 on COCO dataset.

Usage:
    # Single GPU training:
    torchrun --nproc_per_node=1 train_sam3_small_coco.py
    
    # Or for quick test without distributed:
    python train_sam3_small_coco.py --single-gpu

This script trains the SAM3-Small model (16 ViT layers, full 1008 resolution)
on COCO val2017 (5K images, 80 categories).
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-gpu", action="store_true", 
                       help="Run without torchrun (single GPU)")
    args = parser.parse_args()
    
    import torch
    
    # Set environment for single GPU if not using torchrun
    if args.single_gpu or "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
    
    rank = int(os.environ.get("RANK", 0))
    
    if rank == 0:
        print("=" * 70)
        print("SAM3-Small V2 Training on COCO")
        print("=" * 70)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print()
    
    # Import hydra and run training
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    
    # Config path
    config_dir = os.path.join(os.path.dirname(__file__), "training/configs")
    config_name = "sam3_small_coco"
    
    if rank == 0:
        print(f"Config: {config_dir}/{config_name}.yaml")
        print()
    
    # Initialize hydra
    with initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base=None):
        cfg = compose(config_name=config_name)
    
    if rank == 0:
        # Print key settings
        print("Training Configuration:")
        print(f"  Dataset: COCO val2017")
        print(f"  Resolution: {cfg.scratch.resolution}")
        print(f"  Batch size: {cfg.scratch.train_batch_size} × {cfg.scratch.gradient_accumulation_steps} = {cfg.scratch.train_batch_size * cfg.scratch.gradient_accumulation_steps} effective")
        print(f"  Epochs: {cfg.trainer.max_epochs}")
        print(f"  AMP: {cfg.trainer.optim.amp.enabled} ({cfg.trainer.optim.amp.amp_dtype})")
        print(f"  Output: {cfg.launcher.experiment_log_dir}")
        print()
        
        # Create output directory
        os.makedirs(cfg.launcher.experiment_log_dir, exist_ok=True)
        
        # Save resolved config
        config_path = os.path.join(cfg.launcher.experiment_log_dir, "config_resolved.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(cfg, f)
        print(f"Config saved to: {config_path}")
        print()
    
    # Instantiate trainer
    if rank == 0:
        print("Initializing trainer...")
    trainer = instantiate(cfg.trainer)
    
    if rank == 0:
        print()
        print("=" * 70)
        print("Starting training...")
        print("=" * 70)
    
    # Run training
    trainer.train()
    
    if rank == 0:
        print()
        print("=" * 70)
        print("Training complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()

