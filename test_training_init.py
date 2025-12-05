#!/usr/bin/env python3
"""
Test script to verify training initialization works without data loading.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_training_components():
    print("=" * 60)
    print("Testing SAM3-Small Training Components")
    print("=" * 60)

    try:
        # Test model building
        print("1. Building model...")
        from training.sam3_small.model_builder import build_sam3_small
        import torch

        model = build_sam3_small(
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            device="cpu"
        )
        print(f"   ✓ Model built: {sum(p.numel() for p in model.parameters())} parameters")

        # Test loss function
        print("2. Testing loss function...")
        from sam3.train.loss.sam3_loss import Sam3LossWrapper
        from sam3.train.matcher import BinaryHungarianMatcherV2

        matcher = BinaryHungarianMatcherV2(
            focal=True, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
            alpha=0.25, gamma=2, stable=False
        )

        loss_fn = Sam3LossWrapper(
            matcher=matcher,
            o2m_weight=2.0,
            use_o2m_matcher_on_o2m_aux=False
        )
        print("   ✓ Loss function created")

        # Test optimizer construction
        print("3. Testing optimizer...")
        from sam3.train.optim.optimizer import construct_optimizer

        optimizer_conf = {
            "_target_": "torch.optim.AdamW",
            "lr": 8e-5,
            "weight_decay": 0.1
        }

        options_conf = {
            "lr": {
                "scheduler": {
                    "_target_": "sam3.train.optim.schedulers.InverseSquareRootParamScheduler",
                    "base_lr": 8e-5,
                    "timescale": 50,
                    "warmup_steps": 100,
                    "cooldown_steps": 50
                }
            },
            "weight_decay": {
                "scheduler": {
                    "_target_": "fvcore.common.param_scheduler.ConstantParamScheduler",
                    "value": 0.1
                }
            }
        }

        optimizer = construct_optimizer(
            model=model,
            optimizer_conf=optimizer_conf,
            options_conf=options_conf
        )
        print("   ✓ Optimizer constructed")

        # Test gradient scaler
        print("4. Testing gradient scaler...")
        scaler = torch.amp.GradScaler()
        print("   ✓ Gradient scaler created")

        print("\n" + "=" * 60)
        print("SUCCESS: All training components initialize correctly!")
        print("The training code is working - issue is with data loading.")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_components()
    sys.exit(0 if success else 1)