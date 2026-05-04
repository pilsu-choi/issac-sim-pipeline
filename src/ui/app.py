"""Gradio demo UI for the humanoid egocentric object detector."""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image
from prometheus_client import Counter, Histogram, start_http_server
from ultralytics import YOLO


MODEL_PATH = Path(os.getenv("MODEL_PATH", "runs/detect/train/weights/best.pt"))
UI_PORT = int(os.getenv("UI_PORT", "7860"))
UI_METRICS_PORT = int(os.getenv("UI_METRICS_PORT", "7861"))

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total number of inference requests.",
)
REQUEST_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds.",
)


@lru_cache(maxsize=1)
def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file does not exist: {MODEL_PATH}. "
            "Set MODEL_PATH or train the model before launching the UI."
        )
    return YOLO(str(MODEL_PATH))


def predict(image: Image.Image, confidence: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    REQUEST_COUNT.inc()
    start_time = time.perf_counter()
    try:
        model = load_model()
        results = model.predict(image, conf=confidence, verbose=False)

        if not results:
            return np.asarray(image), []

        result = results[0]
        # Ultralytics returns an annotated BGR array; Gradio expects RGB.
        annotated_image = result.plot()[:, :, ::-1]

        detections: list[dict[str, Any]] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": result.names.get(class_id, str(class_id)),
                        "confidence": round(float(box.conf[0]), 4),
                        "bbox_xyxy": [round(float(value), 2) for value in box.xyxy[0]],
                    }
                )

        return annotated_image, detections
    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - start_time)


with gr.Blocks(title="Humanoid Egocentric Perception Demo") as demo:
    gr.Markdown(
        """
        # Humanoid Egocentric Perception Demo

        Isaac Sim synthetic dataset으로 학습한 YOLO 모델의 객체 탐지 결과를 확인하는 데모 UI입니다.
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            confidence = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=0.25,
                step=0.05,
                label="Confidence Threshold",
            )
            run_button = gr.Button("Run Detection", variant="primary")

        with gr.Column():
            output_image = gr.Image(type="numpy", label="Detection Result")
            output_json = gr.JSON(label="Detections")

    gr.Markdown(
        f"""
        **Model path:** `{MODEL_PATH}`

        Tip: synthetic dataset sample images are under
        `data/yolo/humanoid_egocentric_perception/images/val`.
        """
    )

    run_button.click(
        fn=predict,
        inputs=[input_image, confidence],
        outputs=[output_image, output_json],
    )


if __name__ == "__main__":
    start_http_server(UI_METRICS_PORT)
    demo.launch(server_name="0.0.0.0", server_port=UI_PORT)
