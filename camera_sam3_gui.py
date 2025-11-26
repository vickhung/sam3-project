"""
Live SAM3 video segmentation demo with prompt text input.

This script avoids the OpenCV/CUDA crash by running SAM3 inference in a
separate process. The main process handles the GUI (Tkinter) and camera,
while the worker process performs all CUDA work and returns overlaid frames.
"""

import multiprocessing as mp
import queue
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

# Ensure the SAM3 package is discoverable
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SAM3_DIR = ROOT_DIR / "sam3"
sys.path.append(str(SAM3_DIR))

import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


@dataclass
class WorkerConfig:
    # Run SAM3 on every Nth frame; lower = more responsive, higher = less GPU load.
    process_every_n: int = 2
    # Maximum resolution used for SAM3 inference (longer image side).
    # The GUI will still display full-resolution frames; masks are resized.
    max_infer_resolution: int = 720
    default_prompt: str = "person"


def overlay_masks_on_frame(frame_bgr: np.ndarray, masks_tensor: torch.Tensor) -> np.ndarray:
    """Return a BGR frame with SAM3 masks overlaid."""
    if masks_tensor is None or masks_tensor.numel() == 0:
        return frame_bgr

    if masks_tensor.dtype == torch.bool:
        masks_float = masks_tensor.float()
    else:
        masks_float = masks_tensor

    masks_cpu = masks_float.detach().cpu()
    masks_np = masks_cpu.numpy()

    if masks_np.ndim == 4:
        masks_np = masks_np.squeeze(1)

    if masks_np.ndim == 3:
        combined_mask = np.any(masks_np > 0.5, axis=0).astype(np.uint8)
    elif masks_np.ndim == 2:
        combined_mask = (masks_np > 0.5).astype(np.uint8)
    else:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    if combined_mask.shape != (h, w):
        combined_mask = cv2.resize(combined_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = frame_bgr.copy()
    overlay[combined_mask == 1] = [0, 255, 0]
    blended = cv2.addWeighted(overlay, 0.4, frame_bgr, 0.6, 0)
    return blended


def sam3_worker(frame_q: mp.Queue, result_q: mp.Queue, prompt_q: mp.Queue, config: WorkerConfig):
    """Worker process: receives frames, runs SAM3, sends back overlaid frames."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.cuda.set_device(0)
            # Let cuDNN pick the fastest algorithms for this GPU / input size.
            torch.backends.cudnn.benchmark = True
        model = build_sam3_image_model(enable_segmentation=True, device=device)
        processor = Sam3Processor(model, device=device)
        print("[Worker] SAM3 model loaded on", device)
    except Exception as exc:
        result_q.put(("error", f"Failed to load SAM3: {exc}"))
        return

    current_prompt = config.default_prompt
    frames_since_last = 0

    while True:
        try:
            frame = frame_q.get()
        except (EOFError, KeyboardInterrupt):
            break

        if frame is None:
            break

        # Check for prompt updates
        try:
            while True:
                current_prompt = prompt_q.get_nowait()
        except queue.Empty:
            pass

        frames_since_last += 1
        if frames_since_last % config.process_every_n != 0:
            continue

        try:
            # Optionally downscale for faster inference while keeping display full-res.
            frame_for_infer = frame
            if config.max_infer_resolution is not None:
                h, w = frame.shape[:2]
                longer_side = max(h, w)
                if longer_side > config.max_infer_resolution:
                    scale = config.max_infer_resolution / float(longer_side)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame_for_infer = cv2.resize(
                        frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
                    )

            frame_rgb = cv2.cvtColor(frame_for_infer, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            inference_state = processor.set_image(pil_image)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            output_state = processor.set_text_prompt(current_prompt, inference_state)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            masks = output_state.get("masks")
            overlaid = overlay_masks_on_frame(frame, masks)

            # Encode to JPEG bytes for transfer
            success, buffer = cv2.imencode(".jpg", overlaid)
            if success:
                result_q.put(("frame", buffer.tobytes()))
            else:
                result_q.put(("error", "Failed to encode frame"))

            # Cleanup GPU memory
            del inference_state
            del output_state
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as exc:
            result_q.put(("error", f"Inference error: {exc}"))

    print("[Worker] shutting down")


class LiveSegmentationGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SAM3 Live Segmentation")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.video_label = tk.Label(self.root)
        self.video_label.grid(row=0, column=0, padx=10, pady=10)

        self.segmented_label = tk.Label(self.root)
        self.segmented_label.grid(row=0, column=1, padx=10, pady=10)

        self.prompt_entry = tk.Entry(self.root, width=40)
        self.prompt_entry.insert(0, "person")
        self.prompt_entry.grid(row=1, column=0, padx=10, pady=5, sticky="we")

        self.prompt_button = tk.Button(self.root, text="Apply Prompt", command=self.send_prompt)
        self.prompt_button.grid(row=1, column=1, padx=10, pady=5, sticky="e")

        self.status_var = tk.StringVar(value="Status: Initializing...")
        self.status_label = tk.Label(self.root, textvariable=self.status_var)
        self.status_label.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_var.set("Status: Failed to open camera")
            raise RuntimeError("Could not open camera")

        self.frame_q = mp.Queue(maxsize=2)
        self.result_q = mp.Queue(maxsize=2)
        self.prompt_q = mp.Queue()
        self.worker_config = WorkerConfig()
        self.worker = mp.Process(
            target=sam3_worker,
            args=(self.frame_q, self.result_q, self.prompt_q, self.worker_config),
            daemon=True,
        )
        self.worker.start()
        self.status_var.set("Status: Worker started")

        self.update_loop()

    def send_prompt(self):
        prompt = self.prompt_entry.get().strip()
        if prompt:
            self.prompt_q.put(prompt)
            self.status_var.set(f"Status: Prompt queued -> {prompt}")

    def update_loop(self):
        ret, frame = self.cap.read()
        if ret:
            self.show_frame(self.video_label, frame)
            if self.frame_q.qsize() < 2:
                try:
                    self.frame_q.put_nowait(frame.copy())
                except queue.Full:
                    pass

        # Show latest segmented frame if available
        try:
            while True:
                tag, payload = self.result_q.get_nowait()
                if tag == "frame":
                    np_bytes = np.frombuffer(payload, dtype=np.uint8)
                    decoded = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
                    if decoded is not None:
                        self.show_frame(self.segmented_label, decoded)
                elif tag == "error":
                    self.status_var.set(f"Status: {payload}")
        except queue.Empty:
            pass

        self.root.after(10, self.update_loop)

    def show_frame(self, label: tk.Label, frame_bgr: np.ndarray):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=image)
        label.configure(image=imgtk)
        label.image = imgtk

    def on_close(self):
        self.status_var.set("Status: Shutting down...")
        self.cap.release()
        try:
            self.frame_q.put_nowait(None)
        except queue.Full:
            pass
        time.sleep(0.2)
        self.worker.terminate()
        self.worker.join(timeout=1)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    mp.set_start_method("spawn", force=True)
    gui = LiveSegmentationGUI()
    gui.run()


if __name__ == "__main__":
    main()

