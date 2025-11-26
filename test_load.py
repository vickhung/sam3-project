from sam3.model_builder import build_sam3_image_model
import torch

print("Import successful")
print(f"CUDA available: {torch.cuda.is_available()}")
try:
    model = build_sam3_image_model(enable_segmentation=True)
    print("Model built successfully")
except Exception as e:
    print(f"Error building model: {e}")

