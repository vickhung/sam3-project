import cv2
import torch
import numpy as np
from PIL import Image
import os
import sys
import gc
import threading
import queue
import time

# Add the sam3 directory to the python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sam3_dir = os.path.join(current_dir, "sam3")
sys.path.append(sam3_dir)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class CameraSAM3:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = True
        self.last_mask = None
        self.prompt_text = "person"
        self.frame_count = 0
        
    def inference_thread(self, processor, cap, process_every_n=10):
        """Thread for running inference - keeps CUDA operations separate"""
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            if self.frame_count % process_every_n == 0:
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
                    
                    output = processor.set_text_prompt(self.prompt_text, inference_state)
                    
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
                                
                                # Process mask shape
                                if len(masks_np.shape) == 4:
                                    masks_np = masks_np.squeeze(1)
                                
                                if len(masks_np.shape) == 3:
                                    combined_mask = np.any(masks_np > 0.5, axis=0).astype(np.uint8)
                                elif len(masks_np.shape) == 2:
                                    combined_mask = (masks_np > 0.5).astype(np.uint8)
                                else:
                                    combined_mask = None
                                
                                if combined_mask is not None:
                                    h, w = frame.shape[:2]
                                    if combined_mask.shape != (h, w):
                                        combined_mask = cv2.resize(combined_mask, (w, h), 
                                                                  interpolation=cv2.INTER_NEAREST)
                                    self.last_mask = combined_mask
                    
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
                    print(f"Inference error on frame {self.frame_count}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    gc.collect()
            
            # Put frame in queue for display (non-blocking)
            try:
                if not self.frame_queue.full():
                    self.frame_queue.put((frame.copy(), self.last_mask), block=False)
            except queue.Full:
                pass  # Skip if queue is full
        
        cap.release()
    
    def display_thread(self):
        """Thread for OpenCV display - completely separate from CUDA"""
        while self.running:
            try:
                frame, mask = self.frame_queue.get(timeout=0.1)
                
                display_frame = frame.copy()
                
                # Draw mask overlay
                if mask is not None and isinstance(mask, np.ndarray):
                    try:
                        overlay = display_frame.copy()
                        mask_indices = mask == 1
                        if mask_indices.any():
                            overlay[mask_indices] = [0, 255, 0]
                        display_frame = cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0)
                    except:
                        pass
                
                # Draw text
                cv2.putText(display_frame, f"Prompt: {self.prompt_text}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('SAM 3 Live Camera', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Display error: {e}")
                continue
        
        cv2.destroyAllWindows()

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

    print("Webcam opened. Press 'q' to quit.")
    print("Using threaded approach to separate CUDA from OpenCV.")

    camera = CameraSAM3()
    
    # Start inference thread
    inference_thread = threading.Thread(target=camera.inference_thread, args=(processor, cap, 10), daemon=True)
    inference_thread.start()
    
    # Start display thread
    display_thread = threading.Thread(target=camera.display_thread, daemon=True)
    display_thread.start()
    
    try:
        # Wait for threads
        inference_thread.join()
        display_thread.join()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        camera.running = False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        camera.running = False
    finally:
        camera.running = False
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        print("Cleaned up. Goodbye!")

if __name__ == "__main__":
    main()

