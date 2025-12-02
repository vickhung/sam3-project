"""
SAM3-Small: A compact version of SAM3 for training and edge deployment.

Model Specifications:
- Resolution: 252×252 (vs 1008×1008 original)
- Parameters: ~200M (vs ~760M original)
- Size: ~0.8GB FP32, ~0.4GB FP16

Usage:
    from training.sam3_small import build_sam3_small, SAM3SmallConfig
    
    model = build_sam3_small(
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        device="cuda",
    )
"""

__version__ = "0.1.0"

from .model_builder import build_sam3_small, SAM3SmallConfig, SAM3_SMALL_CONFIG

__all__ = [
    "__version__",
    "build_sam3_small",
    "SAM3SmallConfig",
    "SAM3_SMALL_CONFIG",
]