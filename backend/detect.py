# pyrefly: ignore [missing-import]
from ultralytics import YOLO
import cv2
import numpy as np
import winsound
import time

import os

# Locate files dynamically since we moved into the backend folder
model_path = "yolov8n.pt"
if not os.path.exists(model_path) and os.path.exists("backend/" + model_path):
    model_path = "backend/" + model_path

video_path = "A_busy_urban_pedestrian_street.mp4"
if not os.path.exists(video_path) and os.path.exists("../" + video_path):
    video_path = "../" + video_path
elif not os.path.exists(video_path) and os.path.exists("backend/" + video_path):
    video_path = "backend/" + video_path

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
frame_area_m2 = 50  # guess: assume video shows ~50 sqm area (adjust later)

frame_num = 0
people_count = 0
risk_history = []

danger_start_time = None
alert_triggered = False
cooldown_until = 0  # timestamp until which new alarm blocked

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

    # divergence: how much arrows spreading (+) or squishing (-)
    dx = flow[..., 0]
    dy = flow[..., 1]
    d_dx_dx = np.gradient(dx, axis=1)
    d_dy_dy = np.gradient(dy, axis=0)
    divergence = np.mean(d_dx_dx + d_dy_dy)

    if frame_num % 5 == 0:
        results = model(frame, verbose=False)
        people_count = sum(1 for box in results[0].boxes if int(box.cls[0]) == 0)
        for box in results[0].boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    density = people_count / frame_area_m2

    # negative divergence (squishing) = danger. Flip sign so danger = positive.
    raw_risk = density * (-divergence) * 100000
    raw_risk = max(raw_risk, 0)  # negative (spreading) = 0 risk, not danger

    risk_history.append(raw_risk)
    if len(risk_history) > 10:
        risk_history.pop(0)
    risk_score = sum(risk_history) / len(risk_history)  # smoothed average

    # decide color/label
    if risk_score < 10:
        color = (0, 255, 0)   # green
        label = "SAFE"
    elif risk_score < 30:
        color = (0, 255, 255) # yellow
        label = "CAUTION"
    else:
        color = (0, 0, 255)   # red
        label = "DANGER"

    print(f"Frame {frame_num}: {label} | risk={risk_score:.2f} | people={people_count}")

    cv2.putText(frame, f"{label} | Risk: {risk_score:.2f} | People: {people_count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    now = time.time()

    if label == "DANGER":
        if danger_start_time is None:
            danger_start_time = now

        sustained = now - danger_start_time >= 10  # danger held 10+ sec

        if sustained and now >= cooldown_until:
            if not alert_triggered:
                winsound.Beep(1000, 500)
                alert_triggered = True
                cooldown_until = now + 20  # block new alarm for 20 sec after this one
            cv2.putText(frame, "!! EVACUATE ZONE !!", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        danger_start_time = None
        alert_triggered = False

    cv2.imshow("CrowdPhysics - press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_gray = gray

cap.release()
cv2.destroyAllWindows()