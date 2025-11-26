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
    else:
        print("WARNING: CUDA not available, using CPU (will be very slow!)")
    
    print("Loading SAM 3 model...")
    try:
        # Explicitly set device to cuda
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set CUDA device explicitly for RTX 4070
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        
        model = build_sam3_image_model(enable_segmentation=True, device=device)
        
        # Fix processor device initialization - explicitly pass device
        processor = Sam3Processor(model, device=device)
        
        # Verify model is on GPU
        if torch.cuda.is_available():
            # Check if model parameters are on GPU
            first_param = next(model.parameters())
            print(f"Model device: {first_param.device}")
            if first_param.device.type != 'cuda':
                print("WARNING: Model is not on GPU! Moving to GPU...")
                model = model.cuda()
            
            # Verify processor device matches
            print(f"Processor device: {processor.device}")
            if processor.device != device:
                print(f"WARNING: Processor device mismatch! Expected {device}, got {processor.device}")
        
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

    # Set lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Webcam opened. Press 'q' to quit.")
    print("Processing every 10th frame for stability.")
    print("Note: If you experience crashes, the issue may be OpenCV/CUDA interaction.")

    prompt_text = "person"
    frame_count = 0
    process_every_n = 10  # Process less frequently
    last_mask = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            display_frame = frame.copy()
            
            # Only process every Nth frame
            if frame_count % process_every_n == 0:
                try:
                    # Safety checks: Verify frame is valid
                    if frame is None or frame.size == 0:
                        print(f"Warning: Invalid frame at {frame_count}, skipping...")
                        continue
                    
                    # Show GPU memory before inference
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        mem_before = torch.cuda.memory_allocated(0) / 1024**2
                    
                    # Convert frame to RGB with error handling
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        
                        # Validate image dimensions
                        if pil_image.size[0] == 0 or pil_image.size[1] == 0:
                            print(f"Warning: Invalid PIL image dimensions at {frame_count}, skipping...")
                            continue
                    except Exception as e:
                        print(f"Error converting frame to PIL at {frame_count}: {e}")
                        continue

                    # Verify CUDA availability before inference
                    if torch.cuda.is_available() and not torch.cuda.is_initialized():
                        print("Warning: CUDA not initialized, reinitializing...")
                        torch.cuda.init()

                    # Process with SAM 3
                    inference_state = processor.set_image(pil_image)
                    
                    # CRITICAL: Synchronize before text prompt to ensure image processing is complete
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    output = processor.set_text_prompt(prompt_text, inference_state)
                    
                    # CRITICAL: Synchronize immediately after inference
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    # Show GPU memory after inference
                    if torch.cuda.is_available():
                        mem_after = torch.cuda.memory_allocated(0) / 1024**2
                        if frame_count % (process_every_n * 10) == 0:  # Print every 100 frames
                            print(f"GPU Memory: {mem_after:.1f} MB (used {mem_after - mem_before:.1f} MB)")
                    
                    # Extract and process masks with proper GPU tensor handling
                    # Note: output is the state dict, masks are stored in state["masks"]
                    masks = None
                    if "masks" in output:
                        masks = output["masks"]
                    
                    if masks is not None:
                        # Check if masks is empty
                        try:
                            mask_len = len(masks) if hasattr(masks, '__len__') else 0
                        except:
                            mask_len = 0
                        
                        if mask_len > 0:
                            # CRITICAL FIX: Synchronize CUDA operations before accessing tensors
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            
                            # CRITICAL FIX: Properly handle GPU tensors
                            # Move to CPU immediately and detach from computation graph
                            try:
                                if isinstance(masks, torch.Tensor):
                                    # Check tensor device before operations
                                    if masks.device.type == 'cuda':
                                        # Handle boolean tensors: convert to float first
                                        if masks.dtype == torch.bool:
                                            masks_float = masks.float()
                                            masks_cpu = masks_float.detach().cpu()
                                        else:
                                            masks_cpu = masks.detach().cpu()
                                        
                                        # Synchronize again before numpy conversion
                                        torch.cuda.synchronize()
                                        masks_np = masks_cpu.numpy()
                                    else:
                                        # Already on CPU
                                        if masks.dtype == torch.bool:
                                            masks_np = masks.float().numpy()
                                        else:
                                            masks_np = masks.numpy()
                                else:
                                    masks_np = np.array(masks)
                            except Exception as tensor_error:
                                print(f"Error converting masks tensor at frame {frame_count}: {tensor_error}")
                                masks_np = None
                            
                            if masks_np is not None:
                                # Handle shape
                                if len(masks_np.shape) == 4:
                                    masks_np = masks_np.squeeze(1)
                                
                                if len(masks_np.shape) == 3:
                                    combined_mask = np.any(masks_np > 0.5, axis=0).astype(np.uint8)
                                elif len(masks_np.shape) == 2:
                                    combined_mask = (masks_np > 0.5).astype(np.uint8)
                                else:
                                    combined_mask = None

                                if combined_mask is not None:
                                    # Resize to match frame
                                    h, w = frame.shape[:2]
                                    if combined_mask.shape != (h, w):
                                        combined_mask = cv2.resize(combined_mask, (w, h), 
                                                                  interpolation=cv2.INTER_NEAREST)
                                    
                                    last_mask = combined_mask
                    
                    # CRITICAL FIX: Proper memory cleanup
                    # Move all GPU tensors to CPU before deletion
                    if torch.cuda.is_available():
                        # Extract any remaining GPU tensors to CPU
                        if "masks" in output and isinstance(output["masks"], torch.Tensor):
                            if output["masks"].device.type == 'cuda':
                                _ = output["masks"].detach().cpu()
                        if "boxes" in output and isinstance(output["boxes"], torch.Tensor):
                            if output["boxes"].device.type == 'cuda':
                                _ = output["boxes"].detach().cpu()
                        if "scores" in output and isinstance(output["scores"], torch.Tensor):
                            if output["scores"].device.type == 'cuda':
                                _ = output["scores"].detach().cpu()
                        
                        # Clear inference state GPU references
                        if "backbone_out" in inference_state:
                            backbone_out = inference_state["backbone_out"]
                            for key, value in backbone_out.items():
                                if isinstance(value, torch.Tensor) and value.device.type == 'cuda':
                                    _ = value.detach().cpu()
                    
                    # Delete references
                    del inference_state
                    del output
                    
                    # Explicit CUDA synchronization before cache clear
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    
                    # Force garbage collection
                    gc.collect()
                    
                except Exception as e:
                    print(f"Inference error on frame {frame_count}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Cleanup on error
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    gc.collect()
                    # Continue with last mask if available
            
            # CRITICAL: Ensure ALL CUDA operations are complete before ANY OpenCV operations
            # This prevents segfaults from CUDA/OpenCV thread conflicts
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                # Small delay to ensure GPU operations fully complete
                import time
                time.sleep(0.001)  # 1ms delay
            
            # Draw mask overlay if available (CPU operations only)
            if last_mask is not None:
                try:
                    # Ensure mask is valid numpy array on CPU
                    if isinstance(last_mask, np.ndarray):
                        overlay = display_frame.copy()
                        mask_indices = last_mask == 1
                        if mask_indices.any():
                            overlay[mask_indices] = [0, 255, 0]
                        display_frame = cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0)
                except Exception as e:
                    print(f"Error drawing overlay at frame {frame_count}: {e}")
            
            # Draw text (CPU operations only)
            try:
                cv2.putText(display_frame, f"Prompt: {prompt_text}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_frame, f"Frame: {frame_count}", (10, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except Exception as e:
                print(f"Error drawing text at frame {frame_count}: {e}")

            # CRITICAL: Final CUDA sync before OpenCV window operations
            # OpenCV window operations can cause segfaults if CUDA operations are still pending
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # OpenCV display operations - these can conflict with CUDA
            try:
                # Use a try-except to catch any OpenCV errors
                cv2.imshow('SAM 3 Live Camera', display_frame)
                
                # Use non-blocking waitKey with proper error handling
                key = cv2.waitKey(1)
                if key != -1:
                    key = key & 0xFF
                    if key == ord('q'):
                        break
            except cv2.error as e:
                print(f"OpenCV error at frame {frame_count}: {e}")
                # Continue without display if OpenCV fails
                pass
            except Exception as e:
                print(f"Error displaying frame at {frame_count}: {e}")
                # Don't break, just skip display for this frame
                pass

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Clear GPU memory with proper synchronization
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        print("Cleaned up. Goodbye!")

if __name__ == "__main__":
    main()
