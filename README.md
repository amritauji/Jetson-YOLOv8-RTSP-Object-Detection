# 🚀 Jetson YOLOv8 RTSP Object Detection with Alerts

A **real-time Jetson-optimized object detection and alert system** built with **YOLOv8**.  
It connects to **RTSP camera feeds**, performs **real-time inference** using the Jetson’s **GPU (CUDA)**, and generates **alerts with snapshots and logs** when specific objects are detected.

---

## 📘 Overview

This project demonstrates how to deploy **Ultralytics YOLOv8** on **NVIDIA Jetson Orin Nano** for **real-time surveillance**.  
It includes automatic RTSP reconnection, threaded video capture, background alert logging, and optional sound notifications for detections of interest.

---

## 🧠 Features

- 🔄 Real-time object detection from RTSP stream  
- ⚙️ Optimized for NVIDIA Jetson (CUDA, FP16, TensorRT-ready)  
- 💾 Logs detections with timestamps and bounding boxes  
- 📸 Saves alert snapshots for specified object classes  
- 🔔 Optional system beep when alert objects appear  
- 🔧 Auto RTSP reconnect and threaded frame handling  
- 🧰 Environment-configurable parameters  

---


---

## 🧰 Requirements

### Hardware
- NVIDIA Jetson Orin Nano / Xavier / Nano
- JetPack with CUDA, cuDNN, TensorRT installed
- RTSP-enabled IP camera

### Software
- Ubuntu 20.04 / 22.04
- Python ≥ 3.8

---

## 📦 Dataset Used

The project uses **Ultralytics YOLOv8 pretrained weights**, trained on the **COCO 2017 dataset** (80 object classes).  
Default model used: `yolov8n.pt` (lightweight version).  

You can download custom YOLOv8 weights from [Ultralytics Models](https://github.com/ultralytics/ultralytics).

---

## ⚙️ Installation

### 1️⃣ Update & install dependencies
```bash
sudo apt update && sudo apt install python3-pip -y
pip3 install ultralytics opencv-python torch torchvision numpy

🔧 Configuration

Set your RTSP camera credentials and environment variables:
export RTSP_USERNAME="your_username"
export RTSP_PASSWORD="your_password"
export RTSP_IP="192.168.1.100"
export RTSP_PORT="554"
export RTSP_PATH="stream1"

# Optional parameters
export MODEL_PATH="models/yolov8n.pt"
export CONF_THRESHOLD=0.35
export ALERT_CLASSES="person,cell phone,laptop,backpack"
export ENABLE_BEEP=1

▶️ Run the Project

🧾 Working Procedure
1️⃣ RTSP Connection

Connects to your camera via RTSP using GStreamer/FFmpeg.

Auto-reconnects if connection fails.

2️⃣ Model Initialization

Loads YOLOv8 model on GPU (CUDA).

Uses half precision (FP16) for improved inference speed.

3️⃣ Threaded Capture

Captures frames continuously using a background thread.

Allows inference without blocking video stream.

4️⃣ Detection & Alerts

Every Nth frame (default skip = 3) is passed to YOLOv8.

Detected objects are annotated with bounding boxes.



If object class matches an alert type:

📸 Saves frame to logs/alerts/

🧾 Logs detection to logs/detections.log

🔊 Optional beep sound for alert

5️⃣ Logging & Display

Displays FPS, number of detected objects, and system stats.

Keeps background logs for post-analysis.

6️⃣ Graceful Shutdown

Cleans up threads, camera stream, and logs on exit.

Shows session summary (duration, FPS, total frames, etc).

🧰 Troubleshooting
Issue	Fix
RTSP stream not connecting	Check IP, port, credentials; ensure camera is reachable
Low FPS	Increase SKIP_FRAMES, reduce IMGSZ
GPU not used	Ensure CUDA & JetPack installed (nvcc --version)
Beep sound missing	Check /usr/share/sounds/ path or disable beep

