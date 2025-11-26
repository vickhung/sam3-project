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

def test_static_image():
    """Test SAM 3 with a static image first"""
    print("Loading SAM 3 model...")
    try:
        model = build_sam3_image_model(enable_segmentation=True)
        processor = Sam3Processor(model)
        print("SAM 3 model loaded successfully.")
    except Exception as e:
        print(f"Error loading SAM 3 model: {e}")
        return False

    # Try to capture one frame from webcam
    print("Capturing test frame from webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read frame.")
        return False
    
    print(f"Frame captured: {frame.shape}")
    
    # Convert to PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    print(f"PIL image size: {pil_image.size}")
    
    # Process with SAM 3
    print("Processing with SAM 3...")
    try:
        inference_state = processor.set_image(pil_image)
        print("Image set successfully")
        
        output = processor.set_text_prompt("person", inference_state)
        print("Text prompt processed successfully")
        
        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")
        
        print(f"Found {len(masks) if masks is not None else 0} masks")
        print(f"Found {len(boxes) if boxes is not None else 0} boxes")
        print(f"Found {len(scores) if scores is not None else 0} scores")
        
        if masks is not None and len(masks) > 0:
            print("Success! Model is working.")
            return True
        else:
            print("No masks found, but no crash either.")
            return True
            
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_static_image()
    if success:
        print("\n✓ Static image test passed! The issue might be with the real-time loop.")
    else:
        print("\n✗ Static image test failed. There's a deeper issue.")

