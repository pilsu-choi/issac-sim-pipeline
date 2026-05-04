"""FastAPI inference service for the trained YOLO detector."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response
from ultralytics import YOLO


MODEL_PATH = Path(os.getenv("MODEL_PATH", "runs/detect/train/weights/best.pt"))

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total number of inference requests.",
)
REQUEST_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds.",
)

app = FastAPI(title="Humanoid Egocentric Object Detector")
model: YOLO | None = None


@app.get("/")
def root() -> dict[str, object]:
    return {
        "service": "Humanoid Egocentric Object Detector",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "predict": {
            "method": "POST",
            "path": "/predict",
            "field": "file",
        },
    }


@app.on_event("startup")
def load_model() -> None:
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file does not exist: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    REQUEST_COUNT.inc()
    start_time = time.perf_counter()

    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as image_file:
        image_file.write(await file.read())
        image_file.flush()
        results = model.predict(image_file.name, verbose=False)

    REQUEST_LATENCY.observe(time.perf_counter() - start_time)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            xyxy = [float(value) for value in box.xyxy[0]]
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": result.names.get(class_id, str(class_id)),
                    "confidence": confidence,
                    "bbox_xyxy": xyxy,
                }
            )

    return {
        "filename": file.filename,
        "num_detections": len(detections),
        "detections": detections,
    }
