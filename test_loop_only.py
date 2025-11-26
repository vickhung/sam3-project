import cv2
import torch
import numpy as np
from PIL import Image
import os
import sys
import gc

# Add the sam3 directory to the python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sam3_dir = os.path.join(current_dir, "sam3")
sys.path.append(sam3_dir)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def main():
    # Verify CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.cuda.set_device(0)
    else:
        print("WARNING: CUDA not available, using CPU (will be very slow!)")
    
    print("Loading SAM 3 model...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_sam3_image_model(enable_segmentation=True, device=device)
        processor = Sam3Processor(model, device=device)
        
        if torch.cuda.is_available():
            first_param = next(model.parameters())
            print(f"Model device: {first_param.device}")
            print(f"Processor device: {processor.device}")
        
        print("SAM 3 model loaded successfully.")
    except Exception as e:
        print(f"Error loading SAM 3 model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Webcam opened. Processing frames (no display). Press Ctrl+C to stop.")
    print("Processing every 10th frame for stability.")

    prompt_text = "person"
    frame_count = 0
    process_every_n = 10
    success_count = 0
    
    try:
        while frame_count < 100:  # Limit to 100 frames for testing
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            if frame_count % process_every_n == 0:
                try:
                    if frame is None or frame.size == 0:
                        continue
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    if pil_image.size[0] == 0 or pil_image.size[1] == 0:
                        continue

                    # Process with SAM 3
                    inference_state = processor.set_image(pil_image)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    output = processor.set_text_prompt(prompt_text, inference_state)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    # Extract masks safely
                    masks = output.get("masks") if "masks" in output else None
                    
                    if masks is not None:
                        try:
                            mask_len = len(masks) if hasattr(masks, '__len__') else 0
                        except:
                            mask_len = 0
                        
                        if mask_len > 0 and isinstance(masks, torch.Tensor):
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            
                            if masks.device.type == 'cuda':
                                if masks.dtype == torch.bool:
                                    masks_cpu = masks.float().detach().cpu()
                                else:
                                    masks_cpu = masks.detach().cpu()
                                
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                
                                masks_np = masks_cpu.numpy()
                                success_count += 1
                                
                                if frame_count % (process_every_n * 5) == 0:
                                    print(f"Frame {frame_count}: Processed successfully, found {mask_len} masks")
                    
                    # Cleanup
                    if "backbone_out" in inference_state:
                        backbone_out = inference_state["backbone_out"]
                        for key, value in list(backbone_out.items()):
                            if isinstance(value, torch.Tensor) and value.device.type == 'cuda':
                                try:
                                    _ = value.detach().cpu()
                                except:
                                    pass
                    
                    del inference_state
                    del output
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    
                    gc.collect()
                    
                except Exception as e:
                    print(f"Inference error on frame {frame_count}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    gc.collect()

        print(f"\nCompleted {frame_count} frames, {success_count} successful inferences")
        print("No segfault detected!")

    except KeyboardInterrupt:
        print(f"\nInterrupted. Processed {frame_count} frames, {success_count} successful")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        print("Cleaned up. Goodbye!")

if __name__ == "__main__":
    main()

