from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import asyncio
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for now (dev only)
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Locate model file dynamically
model_path = "yolov8n.pt"
if not os.path.exists(model_path) and os.path.exists("backend/" + model_path):
    model_path = "backend/" + model_path
model = YOLO(model_path)


@app.get("/")
def home():
    return {"message": "CrowdPhysics API running"}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # run heavy CV/YOLO work in a background thread so server stays responsive
    result = await asyncio.to_thread(analyze_video, save_path)
    return result


def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_area_m2 = 50  # assumed area, adjust later

    ret, prev_frame = cap.read()
    if not ret:
        return {"error": "Could not read video"}

    # resize for speed (divergence math is scale-invariant, still accurate)
    resize_width = 480

    def resize_frame(f):
        h, w = f.shape[:2]
        scale = resize_width / w
        return cv2.resize(f, (resize_width, int(h * scale)))

    prev_frame = resize_frame(prev_frame)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_num = 0
    people_count = 0
    risk_history = []
    max_risk = 0.0
    danger_frames = 0
    total_checked_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_frame(frame)
        frame_num += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        dx = flow[..., 0]
        dy = flow[..., 1]
        d_dx_dx = np.gradient(dx, axis=1)
        d_dy_dy = np.gradient(dy, axis=0)
        divergence = np.mean(d_dx_dx + d_dy_dy)

        if frame_num % 5 == 0:
            results = model(frame, verbose=False, conf=0.15)
            people_count = sum(1 for box in results[0].boxes if int(box.cls[0]) == 0)

        density = people_count / frame_area_m2
        raw_risk = density * (-divergence) * 100000
        raw_risk = max(raw_risk, 0)

        risk_history.append(raw_risk)
        if len(risk_history) > 10:
            risk_history.pop(0)
        risk_score = sum(risk_history) / len(risk_history)

        max_risk = max(max_risk, risk_score)
        total_checked_frames += 1
        if risk_score >= 30:
            danger_frames += 1

        prev_gray = gray

    cap.release()

    danger_percent = (danger_frames / total_checked_frames * 100) if total_checked_frames else 0

    if max_risk < 10:
        overall_status = "SAFE"
    elif max_risk < 30:
        overall_status = "CAUTION"
    else:
        overall_status = "DANGER"

    # cast numpy types (np.float64 / np.int64) to plain Python types so FastAPI can JSON-serialize
    return {
        "status": overall_status,
        "max_risk_score": round(float(max_risk), 2),
        "danger_percent_of_video": round(float(danger_percent), 1),
        "total_frames_checked": int(total_checked_frames),
        "last_people_count": int(people_count)
    }