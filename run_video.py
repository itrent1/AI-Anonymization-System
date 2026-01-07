import os
import cv2
import warnings
import logging
import threading
import time
import torch
import onnxruntime as ort
from queue import Queue
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# --- SYSTEM CONFIGURATION ---
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ORT_LOGGING_LEVEL'] = '3'

# Path Definitions
ROOT = Path(__file__).resolve().parent
INPUT_VIDEO = ROOT / "inputs" / "input3.mp4"
SOURCE_FACE = ROOT / "inputs" / "unn.jpg" 
OUTPUT_VIDEO = ROOT / "outputs" / "output3.mp4"

# Performance Tuning
BATCH_SIZE = 4 

class VideoStreamer:
    """
    Handles asynchronous video frame retrieval to optimize I/O performance.
    """
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.queue = Queue(maxsize=128)
        self.running = True

    def read_frames(self):
        while self.running:
            if not self.queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.running = False
                    break
                self.queue.put(frame)
            else:
                time.sleep(0.001)

def main():
    # --- HARDWARE ACCELERATION INITIALIZATION ---
    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        execution_device = 0
        print("Hardware Acceleration: CUDA enabled.")
    else:
        providers = ['CPUExecutionProvider']
        execution_device = 'cpu'
        print("Hardware Acceleration: CPU mode.")

    # 1. InsightFace Engine (Modules enabled: Detection, Recognition, Landmarks)
    app = FaceAnalysis(
        name="buffalo_l", 
        providers=providers, 
        allowed_modules=['detection', 'recognition', 'landmark_2d_106']
    )
    app.prepare(ctx_id=0 if execution_device == 0 else -1, det_size=(320, 320)) 
    
    # 2. Inswapper Model
    model_path = ROOT / "weights" / "inswapper_128.onnx"
    swapper = get_model(str(model_path), providers=providers)
    
    # 3. YOLOv8 Object Detection
    yolo = YOLO("yolov8m.pt") 

    def get_face_embedding(path):
        """
        Extracts facial features from the donor image.
        """
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Source image not found: {path}")
        faces = app.get(img)
        if not faces:
            # Attempt detection with upscaling if initial pass fails
            faces = app.get(cv2.resize(img, (0,0), fx=2, fy=2))
        if not faces:
            raise ValueError("No facial features detected in source image.")
        return sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0]

    print("Status: Initializing donor profile...")
    universal_donor = get_face_embedding(SOURCE_FACE)

    # Video Processing Setup
    streamer = VideoStreamer(INPUT_VIDEO)
    total_frames = int(streamer.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = streamer.cap.get(cv2.CAP_PROP_FPS)
    w = int(streamer.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(streamer.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    threading.Thread(target=streamer.read_frames, daemon=True).start()
    out = cv2.VideoWriter(str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # --- MAIN PROCESSING LOOP ---
    pbar = tqdm(total=total_frames, desc="Anonymization Progress")
    
    while streamer.running or not streamer.queue.empty():
        batch_frames = []
        for _ in range(BATCH_SIZE):
            if not streamer.queue.empty():
                batch_frames.append(streamer.queue.get())
            else: 
                break
        
        if not batch_frames: 
            continue

        # Batch Inference (YOLO)
        results = yolo.predict(batch_frames, conf=0.3, device=execution_device, verbose=False)
        
        for i, frame in enumerate(batch_frames):
            for box in results[i].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Region of Interest (ROI) with 15% padding
                padding = int((y2 - y1) * 0.15)
                y1_crop = max(0, y1 - padding)
                roi = frame[y1_crop:y2, x1:x2]
                
                if roi.size > 0:
                    faces = app.get(roi)
                    if faces:
                        # Process the primary face detected in ROI
                        face = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0]
                        roi = swapper.get(roi, face, universal_donor, paste_back=True)
                    
                    frame[y1_crop:y2, x1:x2] = roi

            out.write(frame)
            pbar.update(1)

    # Resource Cleanup
    streamer.cap.release()
    out.release()
    print(f"Process Complete. Output saved to: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()