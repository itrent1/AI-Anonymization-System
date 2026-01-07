# Automated System for Video Stream Anonymization Using Deep Learning

## Executive Summary
This project implements a high-performance automated system for facial anonymization in digital video streams. By integrating state-of-the-art deep learning architectures, the system provides a robust solution for identity protection while maintaining the visual context, lighting conditions, and temporal consistency of the source material.

The system is designed for application in privacy-sensitive environments, including video surveillance and research dataset anonymization.

## System Architecture
The software utilizes a strictly modular, sequential execution pipeline:
1. **Preprocessing:** Asynchronous video stream decoding and frame buffering via FFmpeg and OpenCV.
2. **Object Localization:** YOLOv8-based identification of human silhouettes and Regions of Interest (ROI).
3. **Biometric Analysis:** InsightFace (Buffalo_L) models for 106-point landmark detection and feature extraction.
4. **Identity Synthesis:** Latent space identity transformation using the InSwapper-128 model.
5. **Bitstream Finalization:** H.264 (YUV420p) transcoding for universal compatibility.

## Project Assets and Weights
| Asset Type | File Name | Target Directory | Access Link |
| :--- | :--- | :--- | :--- |
| **Face Swap Model** | inswapper_128.onnx | /weights | [Download via Google Drive](https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view) |
| **Detection Weights** | yolov8n.pt | /weights | [Download via Ultralytics](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt) |

## Installation & Deployment
1. **Environment:** `python -m venv venv`
2. **Activation:** `venv\Scripts\activate` (Windows)
3. **Libraries:** `pip install -r requirements.txt`

## Execution
* **Web Interface:** `streamlit run app.py`
* **CLI Mode:** `python run_video.py`

## Ethical Statement
This software is developed strictly for research, academic, and privacy protection purposes. Unauthorized use for identity manipulation is discouraged.
