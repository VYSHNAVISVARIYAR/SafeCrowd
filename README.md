# 🚦 SafeCrowd

**Physics-informed crowd crush risk detection from video.**

SafeCrowd analyzes crowd footage and estimates real-time crush/stampede risk — not by simply counting people, but by combining crowd **density** with **flow convergence**, the same physics signature that precedes real-world crowd disasters.

---

## The Problem

Crowd crush disasters — Kumbh Mela, Sabarimala, Itaewon — kill hundreds of people every year. These deaths happen when local crowd density crosses a critical threshold *and* people can no longer move freely, forming a compression wave similar to a traffic jam.

No affordable, deployable tool exists for small event organizers (temple committees, local police, mid-size event managers) to get early warning. Enterprise crowd-analytics systems cost lakhs of rupees and need custom camera infrastructure.

**Most crowd-monitoring tools just count people.** A packed but freely-moving crowd and a packed, stuck crowd can have the *same headcount* — but only one of them is dangerous. Headcount alone can't tell them apart.

## The Approach

SafeCrowd is grounded in two established models from crowd-dynamics research, not an arbitrary rule:

- **Social Force Model** (Helbing & Molnár, 1995) — models pedestrians as particles under attractive/repulsive forces; the foundational paper in crowd-disaster analysis.
- **LWR Traffic Shockwave Model** (Lighthill–Whitham–Richards) — the same math used to explain how traffic jams form applies to crowds: when density rises *and* velocity drops at the same time, a compression wave forms. This is the crush precursor.

### Risk formula

```
risk_score = density(zone) × velocity_convergence(zone)
```

- `density` = people detected ÷ zone area
- `velocity_convergence` = how much crowd movement vectors are converging/opposing (stuck, compressing) vs. spreading (flowing freely), computed from optical-flow divergence

High density **alone** isn't flagged as dangerous if the crowd is still flowing. High density **combined with** stalled/converging movement is what triggers risk — this is the actual mechanism behind real crush events, not just a headcount threshold.

## How It Works

1. Video uploaded through the browser
2. **YOLOv8-nano** detects and counts people frame by frame
3. **OpenCV Farneback optical flow** tracks how the crowd is moving between frames
4. Flow divergence is computed — negative divergence (converging vectors) signals compression
5. Risk score = density × convergence, smoothed over a rolling window to reduce frame-to-frame noise
6. Result classified as **SAFE / CAUTION / DANGER** and returned to the frontend

## Tech Stack

| Layer | Tech |
|---|---|
| Detection | YOLOv8-nano (`ultralytics`) |
| Motion tracking | OpenCV (Farneback optical flow) |
| Backend | FastAPI + Uvicorn |
| Frontend | React (Vite) |
| Cost | ₹0 — fully free/open-source, no paid APIs |

## Screenshots

*(Add screenshot of the upload UI and a DANGER-result screenshot here)*

## Running Locally

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn main:app --reload
```
Backend runs at `http://127.0.0.1:8000`

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Frontend runs at `http://localhost:5173`

Upload a crowd video from the browser UI and click **Analyze video**.

## Known Limitations

Stated honestly, as any real safety tool should be:

- **Zone area is currently a fixed estimate**, not measured per-video — risk scores are directionally correct but not calibrated to a real venue yet.
- **YOLOv8-nano undercounts in very dense crowds** due to occlusion (people hidden behind others) — a known limitation of general-purpose object detectors above high density.
- **Optical flow is noisy in dense scenes** — texture and clothing patterns can be mistaken for motion; smoothing helps but doesn't fully eliminate this.
- **Single-camera only** — no 3D depth or multi-camera fusion in this version.
- **Analysis runs on uploaded footage**, not a live camera feed yet — though the same `cv2.VideoCapture` call works identically with an RTSP camera URL in place of a file path, so live deployment is a small extension, not a rebuild.
- Risk score is an **estimate**, not a guaranteed prediction.

## Future Work

- Live RTSP/webcam camera support
- Manual zone-area input for accurate density calibration
- Multi-zone monitoring per venue
- SMS/WhatsApp alerts to on-site organizers
- History logging + trend graphs for post-event review

## Why This Matters

Enterprise crowd-analytics systems are priced out of reach for the events that need them most — temple committees, local festivals, small stadiums. SafeCrowd applies the same physics real crowd-disaster researchers use, in a free, self-hosted tool, aiming to bring early crush-warning capability to gatherings that currently have none.

---

*Built as a self-hosted, zero-paid-API system — no external service dependency, no recurring cost.*
