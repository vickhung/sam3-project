import cv2
import torch
import numpy as np
from PIL import Image
import os
import sys

# Add the sam3 directory to the python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sam3_dir = os.path.join(current_dir, "sam3")
sys.path.append(sam3_dir)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def test_minimal():
    """Minimal test to isolate segfault"""
    print("Step 1: CUDA check")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    print("Step 2: Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(enable_segmentation=True, device=device)
    processor = Sam3Processor(model, device=device)
    print("Model loaded")
    
    print("Step 3: Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam")
        return
    
    print("Step 4: Reading frame...")
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        cap.release()
        return
    
    print(f"Frame shape: {frame.shape}")
    cap.release()
    
    print("Step 5: Converting to PIL...")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    print(f"PIL image size: {pil_image.size}")
    
    print("Step 6: Setting image...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_state = processor.set_image(pil_image)
    print("Image set")
    
    print("Step 7: Setting text prompt...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    output = processor.set_text_prompt("person", inference_state)
    print("Text prompt set")
    
    print("Step 8: Accessing masks...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    masks = output.get("masks")
    print(f"Masks type: {type(masks)}")
    
    if masks is not None:
        print(f"Masks is tensor: {isinstance(masks, torch.Tensor)}")
        if isinstance(masks, torch.Tensor):
            print(f"Masks device: {masks.device}, dtype: {masks.dtype}, shape: {masks.shape}")
            
            print("Step 9: Converting to CPU...")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            if masks.dtype == torch.bool:
                masks_cpu = masks.float().detach().cpu()
            else:
                masks_cpu = masks.detach().cpu()
            
            print("Step 10: Converting to numpy...")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            masks_np = masks_cpu.numpy()
            print(f"Numpy shape: {masks_np.shape}")
    
    print("Step 11: Cleanup...")
    del inference_state
    del output
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    print("SUCCESS: All steps completed!")

if __name__ == "__main__":
    try:
        test_minimal()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

