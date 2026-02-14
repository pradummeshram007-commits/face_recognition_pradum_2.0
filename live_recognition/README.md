# Live Face Recognition System

Real-time webcam-based face recognition powered by **DeepFace + Facenet**.

## Features
- 📷 Live webcam capture via browser
- 📝 Register face with one click
- 🔍 Verify identity in real-time
- ⚡ CPU-optimized (~1s per verification)

## Tech Stack
- **Backend**: FastAPI + DeepFace (Facenet model)
- **Frontend**: Vanilla HTML + JavaScript (getUserMedia API)
- **Distance Metric**: Cosine similarity (threshold: 0.40)

## Setup

```bash
pip install fastapi uvicorn python-multipart deepface tf-keras opencv-python
```

## Run

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open **http://127.0.0.1:8000** in your browser.

## How It Works
1. **Register** — Capture your face from webcam → extract Facenet embedding → store in memory
2. **Verify** — Capture frame → extract embedding → compare cosine distance against registered face
3. If distance < 0.40 → ✅ Match, otherwise → ❌ No match

## File Structure
```
├── main.py        # FastAPI backend (DeepFace + Facenet)
├── index.html     # Webcam frontend (vanilla HTML/JS)
└── README.md
```

## Author
Pradum Meshram
