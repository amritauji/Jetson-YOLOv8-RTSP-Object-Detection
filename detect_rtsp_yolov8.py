#!/usr/bin/env python3
"""
Improved Jetson-compatible YOLOv8 RTSP detector with alerts.
- Use env vars for RTSP creds
- Background writer for alerts/logs
- Robust RTSP connect & reconnect
- Uses device properly and half precision on CUDA
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os, time, datetime, subprocess
from threading import Thread, Lock, Event
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from urllib.parse import quote
from pathlib import Path
import torch
import sys
import signal

# ---------------- CONFIG ----------------
RTSP_USERNAME = os.getenv("RTSP_USERNAME", "mit@07")
RTSP_PASSWORD = os.getenv("RTSP_PASSWORD", "Cse@t^06&B7")
RTSP_IP = os.getenv("RTSP_IP", "172.27.166.143")
RTSP_PORT = os.getenv("RTSP_PORT", "8094")
RTSP_PATH = os.getenv("RTSP_PATH", "h264_ulaw.sdp")

encoded_username = quote(RTSP_USERNAME, safe='')
encoded_password = quote(RTSP_PASSWORD, safe='')
VIDEO_SOURCE = f"rtsp://{encoded_username}:{encoded_password}@{RTSP_IP}:{RTSP_PORT}/{RTSP_PATH}"

MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n.pt")
LOG_PATH = Path(os.getenv("LOG_PATH", "logs/detections.log"))
ALERT_DIR = Path(os.getenv("ALERT_DIR", "logs/alerts"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.35))
ALERT_CLASSES = set(os.getenv("ALERT_CLASSES", "person,cell phone,laptop,backpack").split(","))
ENABLE_BEEP = os.getenv("ENABLE_BEEP", "1") == "1"

# Perf & display
IMGSZ = int(os.getenv("IMGSZ", 320))
SKIP_FRAMES = int(os.getenv("SKIP_FRAMES", 3))
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", 1))
USE_TENSORRT = os.getenv("USE_TENSORRT", "0") == "1"
STREAM_RESOLUTION = (int(os.getenv("STREAM_W", 640)), int(os.getenv("STREAM_H", 480)))
RESIZE_DISPLAY = os.getenv("RESIZE_DISPLAY", "1") == "1"
DISPLAY_WIDTH = int(os.getenv("DISPLAY_WIDTH", 960))

# make dirs
ALERT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE} | PyTorch: {torch.__version__}")

if DEVICE.startswith("cuda"):
    try:
        print(f"[INFO] CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

# load model
print("[INFO] Loading model:", MODEL_PATH)
model = YOLO(MODEL_PATH)
try:
    model.to(DEVICE)
except Exception:
    # ultralytics may handle device in call; ignore if unsupported
    pass

# use half when cuda is available and model supports it
USE_HALF = DEVICE.startswith("cuda")
if USE_HALF:
    try:
        model.model.half()
        print("[INFO] Model set to half precision")
    except Exception:
        pass

# Threaded camera (robust connect + fallback)
class ThreadedCamera:
    def __init__(self, src, resolution=None, buffer_size=1, reconnect=True):
        self.src = src
        self.resolution = resolution
        self.buffer_size = buffer_size
        self.reconnect = reconnect
        self.capture = None
        self.frame = None
        self.grabbed = False
        self.running = True
        self.lock = Lock()
        self._connected = Event()
        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _check_gstreamer(self):
        try:
            result = subprocess.run(['gst-inspect-1.0', '--version'], capture_output=True, timeout=2)
            return result.returncode == 0
        except Exception:
            return False

    def _open_capture(self):
        # try GStreamer hardware -> GStreamer software -> FFMPEG TCP -> FFMPEG UDP -> plain
        use_gst = self._check_gstreamer() and self.src.startswith("rtsp://")
        tried = []
        if use_gst:
            gst_hw = (
                f'rtspsrc location="{self.src}" latency=100 protocols=tcp ! '
                "rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! "
                "video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1"
            )
            tried.append(("GSTREAMER_HW", gst_hw))
            gst_sw = (
                f'rtspsrc location="{self.src}" latency=100 protocols=tcp ! rtph264depay ! h264parse ! avdec_h264 ! "
                "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1"
            )
            tried.append(("GSTREAMER_SW", gst_sw))

        tried.append(("FFMPEG_TCP", self.src))
        # try open
        for name, pipeline in tried:
            try:
                if name.startswith("GSTREAMER"):
                    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                else:
                    # set ffmpeg options via env
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
                    cap = cv2.VideoCapture(pipeline, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    print(f"[SUCCESS] Connected using: {name}")
                    self.capture = cap
                    return True
                else:
                    try:
                        cap.release()
                    except:
                        pass
                except Exception as e:
                print(f"[WARN] {name} open failed: {e}")
        return False

    def _worker(self):
        backoff = 1
        while self.running:
            if not self.capture or not self.capture.isOpened():
                ok = self._open_capture()
                if not ok:
                    print(f"[WARN] Could not open stream. Reattempt in {backoff}s")
                    time.sleep(backoff)
                    backoff = min(10, backoff * 2)
                    continue
                backoff = 1
                # set resolution if desired
                if self.resolution and self.capture:
                    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                    try:
                        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
                    except Exception:
                        pass
                self._connected.set()

            # read loop
            try:
                grabbed, frame = self.capture.read()
            except Exception:
                grabbed, frame = False, None

            if grabbed and frame is not None:
                with self.lock:
                    self.grabbed = True
                    self.frame = frame.copy()
            else:
                # small sleep to avoid busy loop
                time.sleep(0.01)
        # cleanup
        try:
            if self.capture:
                self.capture.release()
        except:
            pass

    def read(self):
        with self.lock:
            return (self.grabbed, self.frame.copy() if self.frame is not None else None)

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        try:
            if self.capture and self.capture.isOpened():
                self.capture.release()
        except:
            pass

# background writer for alerts and log lines
executor = ThreadPoolExecutor(max_workers=2)
log_lock = Lock()

def write_log_line(line: str):
    with log_lock:
        with LOG_PATH.open("a") as f:
            f.write(line)

def save_alert_image_async(frame):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = ALERT_DIR / f"alert_{ts}.jpg"
    def _save():
        cv2.imwrite(str(fname), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    executor.submit(_save)
    return str(fname)

def play_beep():
    if not ENABLE_BEEP:
        return
    try:
        subprocess.run(["paplay", "/usr/share/sounds/ubuntu/stereo/bell.ogg"], check=False, timeout=0.5,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        try:
            subprocess.run(["aplay", "/usr/share/sounds/alsa/Front_Center.wav"], check=False, timeout=0.5,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            try:
                os.system('printf "\\a"')
            except:
                pass

def log_detection(ts, cls, conf, bbox):
    line = f"{ts}\t{cls}\t{conf:.3f}\t{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}\n"
    executor.submit(write_log_line, line)

# Connect camera
print("[INFO] Connecting to RTSP stream (credentials masked in print)...")
masked_url = f"rtsp://{RTSP_USERNAME}:****@{RTSP_IP}:{RTSP_PORT}/{RTSP_PATH}"
print(f"[INFO] URL: {masked_url}")

cap = ThreadedCamera(VIDEO_SOURCE, resolution=STREAM_RESOLUTION, buffer_size=BUFFER_SIZE)

# wait for first frame
print("[INFO] Waiting for stream connection...")
for _ in range(80):
    ok, frame = cap.read()
    if ok and frame is not None:
        break
    time.sleep(0.1)
else:
    print("❌ Failed to get initial frame. Exiting.")
    cap.release()
    sys.exit(1)

print(f"[INFO] Stream connected. Resolution: {frame.shape[1]}x{frame.shape[0]}")

# warm model (single call)
print("[INFO] Warming up model...")
try:
    dummy = np.zeros((STREAM_RESOLUTION[1], STREAM_RESOLUTION[0], 3), dtype=np.uint8)
    _ = model(dummy, imgsz=IMGSZ, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)
except Exception:
    pass
print("[INFO] Model ready")

# main loop
frame_count = 0
processed_count = 0
start_time = time.time()
fps_q = deque(maxlen=15)
display_frame = None

running = True
def handle_sigint(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

try:
    while running:
        loop_start = time.time()
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue
        frame_count += 1

        if frame_count % SKIP_FRAMES == 0:
            processed_count += 1
            inf_start = time.time()
            # run inference; ultralytics accepts device param
            results = model(frame, imgsz=IMGSZ, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)
            inference_time = time.time() - inf_start

            r = results[0]
            boxes = []
            confs = []
            cls_ids = []
            if hasattr(r, "boxes") and getattr(r, "boxes") is not None and len(r.boxes) > 0:
                # depending on device, boxes may be tensors; convert carefully
                try:
                    boxes_np = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                except Exception:
                    # fallback if already numpy
                    boxes_np = np.array(r.boxes.xyxy) if hasattr(r.boxes, "xyxy") else np.array([])
                    confs = np.array(r.boxes.conf) if hasattr(r.boxes, "conf") else np.array([])
                    cls_ids = np.array(r.boxes.cls).astype(int) if hasattr(r.boxes, "cls") else np.array([])
                boxes = boxes_np if len(boxes_np) > 0 else []

            annotated = frame.copy()
            alert_triggered = False
            alert_items = []

            for bbox, conf, cid in zip(boxes, confs, cls_ids):
                if conf < CONF_THRESHOLD:
                    continue
                cls_name = r.names[int(cid)] if hasattr(r, "names") else str(int(cid))
                x1, y1, x2, y2 = map(int, bbox[:4])
                label = f"{cls_name} {conf:.2f}"
                color = (0, 255, 0)
                if cls_name in ALERT_CLASSES:
                    color = (0, 0, 255)
                    alert_triggered = True
                    alert_items.append((cls_name, float(conf)))
                    ts_iso = datetime.datetime.now().isoformat()
                    log_detection(ts_iso, cls_name, float(conf), (x1, y1, x2, y2))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(annotated, (x1, max(0, y1 - label_size[1] - 6)),
                              (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if alert_triggered:
                fname = save_alert_image_async(annotated.copy())
                play_beep()
                cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 50), (0, 0, 0), -1)
                cv2.putText(annotated, "⚠️ ALERT: Object of Interest Detected",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                print(f"[ALERT] {datetime.datetime.now().strftime('%H:%M:%S')} | {alert_items} -> {fname}")

            loop_time = time.time() - loop_start
            fps_q.append(1.0 / loop_time if loop_time > 0 else 0)
            avg_fps = float(np.mean(fps_q)) if len(fps_q) > 0 else 0.0

            info_bg_h = 70
            cv2.rectangle(annotated, (0, annotated.shape[0] - info_bg_h),
                          (annotated.shape[1], annotated.shape[0]), (0, 0, 0), -1)
            info_lines = [
                f"FPS: {avg_fps:.1f} | Device: {DEVICE.upper()} | Model: {IMGSZ}px",
                f"Frame: {frame_count} | Processed: {processed_count} | Objects: {len(boxes)}"
            ]
            for i, line in enumerate(info_lines):
                y_pos = annotated.shape[0] - info_bg_h + 20 + (i * 25)
                cv2.putText(annotated, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            display_frame = annotated

        # show
        if display_frame is not None:
            show = display_frame.copy()
            if RESIZE_DISPLAY:
                h, w = show.shape[:2]
                scale = DISPLAY_WIDTH / w
                show = cv2.resize(show, (DISPLAY_WIDTH, int(h * scale)), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Jetson Object Detection + Alert", show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

finally:
    print("[INFO] Cleaning up...")
    running = False
    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=True)
    elapsed = time.time() - start_time if start_time else 0.0001
    print("\n=== SESSION SUMMARY ===")
    print(f" Duration: {elapsed:.1f}s")
    print(f" Total frames: {frame_count}")
    print(f" Processed frames: {processed_count}")
    print(f" Average FPS: {processed_count/elapsed:.2f}")
    print(f" Skip ratio: 1/{SKIP_FRAMES}")
    print(f" Log file: {LOG_PATH}")
    print(f" Alert directory: {ALERT_DIR}")
    print("========================\n")
