from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
import asyncio
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# serve processed videos so frontend can play them back
app.mount("/outputs", StaticFiles(directory=OUTPUT_FOLDER), name="outputs")

# Locate model file dynamically
model_path = "yolov8n.pt"
if not os.path.exists(model_path) and os.path.exists("backend/" + model_path):
    model_path = "backend/" + model_path
model = YOLO(model_path)


@app.get("/")
def home():
    return {"message": "SafeCrowd API running"}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = await asyncio.to_thread(analyze_video, save_path, file.filename)
    return result


def analyze_video(video_path, original_filename):
    cap = cv2.VideoCapture(video_path)
    frame_area_m2 = 50  # assumed area, adjust later

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    resize_width = 480

    ret, prev_frame = cap.read()
    if not ret:
        return {"error": "Could not read video"}

    def resize_frame(f):
        h, w = f.shape[:2]
        scale = resize_width / w
        return cv2.resize(f, (resize_width, int(h * scale)))

    prev_frame = resize_frame(prev_frame)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    out_h, out_w = prev_frame.shape[:2]

    output_filename = f"annotated_{original_filename.rsplit('.', 1)[0]}.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    frame_num = 0
    people_count = 0
    risk_history = []
    max_risk = 0.0
    danger_frames = 0
    total_checked_frames = 0

    timeline = []  # [{time_sec, risk_score, label}]
    last_boxes = []

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
            last_boxes = [
                tuple(map(int, box.xyxy[0]))
                for box in results[0].boxes
                if int(box.cls[0]) == 0
            ]

        density = people_count / frame_area_m2
        raw_risk = density * (-divergence) * 100000
        raw_risk = max(raw_risk, 0)

        risk_history.append(raw_risk)
        if len(risk_history) > 10:
            risk_history.pop(0)
        risk_score = sum(risk_history) / len(risk_history)

        if risk_score < 10:
            label = "SAFE"
            color = (0, 255, 0)
        elif risk_score < 30:
            label = "CAUTION"
            color = (0, 255, 255)
        else:
            label = "DANGER"
            color = (0, 0, 255)

        max_risk = max(max_risk, risk_score)
        total_checked_frames += 1
        if risk_score >= 30:
            danger_frames += 1

        # --- draw overlay on frame ---
        for (x1, y1, x2, y2) in last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        step = 20
        for y in range(0, gray.shape[0], step):
            for x in range(0, gray.shape[1], step):
                fx, fy = flow[y, x]
                end_x, end_y = int(x + fx * 3), int(y + fy * 3)
                cv2.arrowedLine(frame, (x, y), (end_x, end_y), (0, 0, 255), 1, tipLength=0.4)

        cv2.putText(frame, f"{label} | Risk: {risk_score:.1f} | People: {people_count}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        writer.write(frame)

        timeline.append({
            "time_sec": round(frame_num / fps, 2),
            "risk_score": round(float(risk_score), 2),
            "label": label
        })

        prev_gray = gray

    cap.release()
    writer.release()

    danger_percent = (danger_frames / total_checked_frames * 100) if total_checked_frames else 0

    if max_risk < 10:
        overall_status = "SAFE"
    elif max_risk < 30:
        overall_status = "CAUTION"
    else:
        overall_status = "DANGER"

    return {
        "status": overall_status,
        "max_risk_score": round(float(max_risk), 2),
        "danger_percent_of_video": round(float(danger_percent), 1),
        "total_frames_checked": int(total_checked_frames),
        "last_people_count": int(people_count),
        "annotated_video_url": f"http://127.0.0.1:8000/outputs/{output_filename}",
        "timeline": timeline
    }