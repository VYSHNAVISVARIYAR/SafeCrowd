from fastapi import FastAPI, UploadFile, File
import shutil
import os
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")


@app.get("/")
def home():
    return {"message": "SafeCrowd API running"}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_video(save_path)
    return result


def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_area_m2 = 50  # assumed area, adjust later

    ret, prev_frame = cap.read()
    if not ret:
        return {"error": "Could not read video"}

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_num = 0
    people_count = 0
    risk_history = []
    max_risk = 0
    danger_frames = 0
    total_checked_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

    return {
        "status": overall_status,
        "max_risk_score": round(max_risk, 2),
        "danger_percent_of_video": round(danger_percent, 1),
        "total_frames_checked": total_checked_frames,
        "last_people_count": people_count
    }