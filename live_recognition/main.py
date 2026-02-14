"""
Live Face Recognition Backend — FIXED
FastAPI + DeepFace (Facenet) — CPU-optimized, single-device demo.

Key fix: enforce_detection=True so we ONLY compare actual face crops,
not full frames (which would match any two webcam images from same room).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(title="Live Face Recognition")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_NAME = "Facenet"
DETECTOR = "opencv"
DISTANCE_METRIC = "cosine"
THRESHOLD = 0.40  # official Facenet cosine threshold

# ─────────────────────────────────────────────
# In-memory store
# ─────────────────────────────────────────────
registered_embedding: list | None = None
registered_name: str = ""


# ─────────────────────────────────────────────
# Preload model ONCE at startup
# ─────────────────────────────────────────────
@app.on_event("startup")
def load_model():
    print("[*] Preloading Facenet model ...")
    t = time.time()
    DeepFace.build_model(task="facial_recognition", model_name=MODEL_NAME)
    DeepFace.build_model(task="face_detector", model_name=DETECTOR)
    print(f"[OK] Model loaded in {time.time() - t:.1f}s")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def bytes_to_cv2(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def get_embedding(img: np.ndarray) -> list:
    """
    Extract face embedding. enforce_detection=True ensures we ONLY
    embed an actual detected face, NOT the full background frame.
    If no face is found, DeepFace raises ValueError — we let it bubble up.
    """
    results = DeepFace.represent(
        img_path=img,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR,
        enforce_detection=True,   # <-- THIS IS CRITICAL
    )
    if not results:
        raise ValueError("No face detected in the frame")

    # Pick the face with the largest area (most prominent)
    best = max(results, key=lambda r: r["facial_area"].get("w", 0) * r["facial_area"].get("h", 0))
    face_area = best["facial_area"]
    face_w = face_area.get("w", 0)
    face_h = face_area.get("h", 0)

    # Sanity check: face must be at least 40x40 to be reliable
    if face_w < 40 or face_h < 40:
        raise ValueError(
            f"Detected face is too small ({face_w}x{face_h}). "
            "Move closer to the camera."
        )

    print(f"    [face] area={face_w}x{face_h}, confidence={best.get('face_confidence', 'N/A')}")
    return best["embedding"]


def cosine_distance(a: list, b: list) -> float:
    va, vb = np.array(a), np.array(b)
    dot = np.dot(va, vb)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0:
        return 1.0
    return 1.0 - float(dot / norm)


# ─────────────────────────────────────────────
# Serve frontend
# ─────────────────────────────────────────────
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "index.html")


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open(FRONTEND_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ─────────────────────────────────────────────
# POST /register — store face embedding
# ─────────────────────────────────────────────
@app.post("/register")
async def register(
    image: UploadFile = File(...),
    name: str = Form("User"),
):
    global registered_embedding, registered_name

    try:
        data = await image.read()
        img = bytes_to_cv2(data)
        print(f"[register] frame={img.shape}, name={name}")

        t = time.time()
        emb = get_embedding(img)
        elapsed = time.time() - t

        registered_embedding = emb
        registered_name = name

        print(f"[register] OK — {name} registered in {elapsed:.2f}s")
        return JSONResponse({
            "success": True,
            "name": name,
            "message": f"Face registered in {elapsed:.2f}s",
            "embedding_size": len(emb),
        })

    except ValueError as e:
        print(f"[register] FAIL — {e}")
        return JSONResponse({"success": False, "message": str(e)}, status_code=400)
    except Exception as e:
        print(f"[register] ERROR — {e}")
        return JSONResponse({"success": False, "message": f"Error: {e}"}, status_code=500)


# ─────────────────────────────────────────────
# POST /verify — compare against stored embedding
# ─────────────────────────────────────────────
@app.post("/verify")
async def verify(image: UploadFile = File(...)):
    global registered_embedding

    if registered_embedding is None:
        return JSONResponse(
            {"verified": False, "message": "No face registered yet. Register first."},
            status_code=400,
        )

    try:
        data = await image.read()
        img = bytes_to_cv2(data)
        print(f"[verify] frame={img.shape}")

        t = time.time()
        current_emb = get_embedding(img)
        dist = cosine_distance(registered_embedding, current_emb)
        elapsed = time.time() - t

        verified = dist < THRESHOLD
        name = registered_name if verified else "Unknown"

        print(f"[verify] distance={dist:.4f}, threshold={THRESHOLD}, verified={verified}, time={elapsed:.2f}s")

        return JSONResponse({
            "verified": verified,
            "distance": round(dist, 4),
            "threshold": THRESHOLD,
            "name": name,
            "time": round(elapsed, 2),
        })

    except ValueError as e:
        print(f"[verify] FAIL — {e}")
        return JSONResponse({"verified": False, "message": str(e)}, status_code=400)
    except Exception as e:
        print(f"[verify] ERROR — {e}")
        return JSONResponse({"verified": False, "message": f"Error: {e}"}, status_code=500)


# ─────────────────────────────────────────────
# GET /status
# ─────────────────────────────────────────────
@app.get("/status")
def status():
    return {
        "model": MODEL_NAME,
        "detector": DETECTOR,
        "distance_metric": DISTANCE_METRIC,
        "threshold": THRESHOLD,
        "registered": registered_embedding is not None,
        "registered_name": registered_name if registered_embedding else None,
    }
