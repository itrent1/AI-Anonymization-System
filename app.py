import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import time
import subprocess
import imageio_ffmpeg as ffmpeg_pkg
from pathlib import Path
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Video Anonymization System", layout="wide")

# Directory and Path Definitions
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "weights" / "inswapper_128.onnx"
INPUTS_DIR = BASE_DIR / "inputs"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Ensure required directories exist
OUTPUTS_DIR.mkdir(exist_ok=True)
INPUTS_DIR.mkdir(exist_ok=True)

# --- MODEL INITIALIZATION AND CACHING ---
@st.cache_resource
def load_ai_models():
    """
    Initializes AI models and selects the optimal computation provider (CUDA or CPU).
    """
    if not WEIGHTS_PATH.exists():
        st.error(f"Critical Error: Model weights not found at {WEIGHTS_PATH}")
        st.stop()

    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    
    # Hardware acceleration check
    if 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        execution_device = 0
    else:
        providers = ['CPUExecutionProvider']
        execution_device = 'cpu'
    
    # 1. InsightFace: Face Analysis Module
    face_app = FaceAnalysis(
        name="buffalo_l", 
        providers=providers, 
        allowed_modules=['detection', 'recognition', 'landmark_2d_106']
    )
    face_app.prepare(ctx_id=0 if execution_device == 0 else -1, det_size=(320, 320))
    
    # 2. Inswapper: Face Swapping Module
    swapper_model = get_model(str(WEIGHTS_PATH), providers=providers)
    
    # 3. YOLOv8: Human Detection Module
    yolo_path = BASE_DIR / "weights" / "yolov8n.pt"
    yolo_model = YOLO(str(yolo_path)) 
    
    return face_app, swapper_model, yolo_model, execution_device

# --- USER INTERFACE ---
st.title("AI-Based Video Anonymization System")
st.markdown("---")

st.sidebar.header("Configuration")
uploaded_face = st.sidebar.file_uploader("1. Upload Donor Portrait", type=["jpg", "png", "jpeg"])
yolo_conf = st.sidebar.slider("Detection Confidence Threshold", 0.1, 0.9, 0.3)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Stream")
    uploaded_video = st.file_uploader("2. Upload Source Video", type=["mp4", "avi", "mov"])

# --- CORE LOGIC ---
if uploaded_video:
    face_app, swapper, yolo, compute_device = load_ai_models()
    
    # Temporary storage for uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    input_video_path = tfile.name

    target_face = None
    donor_img = None

    # Handle Donor Image Selection
    if uploaded_face:
        img_array = np.frombuffer(uploaded_face.read(), np.uint8)
        donor_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        default_face = INPUTS_DIR / "universal.jpg"
        if default_face.exists():
            donor_img = cv2.imread(str(default_face))

    if donor_img is not None:
        detected_faces = face_app.get(donor_img)
        if detected_faces:
            # Select the largest face detected in the donor image
            target_face = sorted(
                detected_faces, 
                key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), 
                reverse=True
            )[0]
            st.sidebar.success(f"Donor profile initialized. Execution Device: {compute_device}")
        else:
            st.sidebar.error("Error: No facial features detected in donor image.")
    
    if target_face is not None:
        if st.button("Execute Anonymization"):
            cap = cv2.VideoCapture(input_video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            timestamp = int(time.time())
            output_raw = OUTPUTS_DIR / f"raw_output_{timestamp}.mp4"
            final_output_path = OUTPUTS_DIR / f"processed_output_{timestamp}.mp4"
            
            # Initial frame writing using standard codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_raw), fourcc, fps, (width, height))

            progress_bar = st.progress(0)
            status_text = st.empty()
            curr_frame = 0
            start_time = time.time()

            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break

                    # Human Detection Phase
                    results = yolo.predict(frame, conf=yolo_conf, device=compute_device, verbose=False)
                    
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Define Region of Interest (ROI) with padding
                            padding = int((y2 - y1) * 0.1)
                            y1_padded = max(0, y1 - padding)
                            roi = frame[y1_padded:y2, x1:x2]
                            
                            if roi.size > 0:
                                # Facial Analysis Phase
                                roi_faces = face_app.get(roi)
                                if roi_faces:
                                    # Select primary face in ROI
                                    best_roi_face = sorted(
                                        roi_faces, 
                                        key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), 
                                        reverse=True
                                    )[0]
                                    # Face Replacement Phase
                                    roi = swapper.get(roi, best_roi_face, target_face, paste_back=True)
                                    frame[y1_padded:y2, x1:x2] = roi

                    out.write(frame)
                    curr_frame += 1
                    
                    if curr_frame % 5 == 0:
                        elapsed = time.time() - start_time
                        processing_fps = curr_frame / elapsed
                        progress_bar.progress(curr_frame / total_frames)
                        status_text.text(f"Progress: {curr_frame}/{total_frames} frames | Velocity: {processing_fps:.1f} FPS")

                cap.release()
                out.release()

                # --- VIDEO TRANSCODING VIA FFmpeg ---
                with st.spinner('Optimizing video stream for web compatibility...'):
                    ffmpeg_exe = ffmpeg_pkg.get_ffmpeg_exe()
                    cmd = [
                        ffmpeg_exe, '-y', '-i', str(output_raw),
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        str(final_output_path)
                    ]
                    # Execute transcoding as a subprocess for system stability
                    subprocess.run(cmd, check=True, capture_output=True)

                with col2:
                    st.subheader("Output Result")
                    if final_output_path.exists():
                        st.video(str(final_output_path))
                        with open(final_output_path, "rb") as f:
                            st.download_button("Download Processed Video", f, file_name="anonymized_output.mp4")
                        st.success("Processing successfully completed.")
                        # Cleanup intermediate files
                        if output_raw.exists(): os.remove(str(output_raw))

            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
            finally:
                # Cleanup temporary source files
                if os.path.exists(input_video_path): os.remove(input_video_path)
    else:
        st.warning("Action Required: Please upload a donor portrait image.")