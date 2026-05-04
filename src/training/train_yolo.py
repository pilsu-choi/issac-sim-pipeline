"""Train a YOLO detector and log the run to MLflow."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import mlflow
import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="Training config path.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str | None:
    return None if device == "auto" else device


def load_yolo_model(model_name: str):
    # We log metrics/artifacts explicitly below. Disable Ultralytics' built-in
    # MLflow callback to avoid a second run writing artifacts to the wrong URI.
    from ultralytics.utils import SETTINGS

    SETTINGS["mlflow"] = False
    from ultralytics import YOLO

    return YOLO(model_name)


def normalize_metric_name(metric_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:/ -]", "", metric_name)


def safe_log_artifact(local_path: Path, artifact_path: str) -> None:
    try:
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
    except Exception as exc:
        print(
            "MLflow artifact upload skipped. "
            f"Local artifact remains at {local_path}. Reason: {exc}"
        )


def log_epoch_metrics_from_results_csv(results_csv_path: Path) -> None:
    if not results_csv_path.exists():
        print(f"MLflow epoch metric logging skipped. Missing file: {results_csv_path}")
        return

    with results_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            row = {metric_name.strip(): value for metric_name, value in row.items()}
            epoch = int(float(row["epoch"]))
            train_loss = 0.0
            val_loss = 0.0
            has_train_loss = False
            has_val_loss = False

            for metric_name, raw_value in row.items():
                if metric_name in {"epoch", "time"} or raw_value in {None, ""}:
                    continue

                metric_value = float(raw_value)
                normalized_name = normalize_metric_name(metric_name)
                mlflow.log_metric(normalized_name, metric_value, step=epoch)

                if normalized_name.startswith("train/") and normalized_name.endswith("_loss"):
                    train_loss += metric_value
                    has_train_loss = True
                elif normalized_name.startswith("val/") and normalized_name.endswith("_loss"):
                    val_loss += metric_value
                    has_val_loss = True

            if has_train_loss:
                mlflow.log_metric("train/loss", train_loss, step=epoch)
            if has_val_loss:
                mlflow.log_metric("val/loss", val_loss, step=epoch)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training_config = config["training"]

    mlflow.set_tracking_uri(config["tracking"]["mlflow_tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run() as run:
        output_project_dir = Path(config["artifacts"]["output_dir"]).resolve()
        run_name = config["artifacts"].get("run_name", "train")

        mlflow.log_params(
            {
                "model_name": config["model_name"],
                "dataset_yaml": config["dataset_yaml"],
                "epochs": training_config["epochs"],
                "image_size": training_config["image_size"],
                "batch_size": training_config["batch_size"],
                "run_name": run_name,
            }
        )

        model = load_yolo_model(config["model_name"])
        results = model.train(
            data=config["dataset_yaml"],
            epochs=training_config["epochs"],
            imgsz=training_config["image_size"],
            batch=training_config["batch_size"],
            workers=training_config["workers"],
            device=resolve_device(training_config["device"]),
            project=str(output_project_dir),
            name=run_name,
            exist_ok=True,
        )

        metrics = getattr(results, "results_dict", {}) or {}
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(normalize_metric_name(metric_name), float(metric_value))

        log_epoch_metrics_from_results_csv(Path(results.save_dir) / "results.csv")

        best_model_path = Path(results.save_dir) / "weights" / "best.pt"
        if best_model_path.exists():
            safe_log_artifact(best_model_path, artifact_path="model")

        print(f"MLflow run id: {run.info.run_id}")
        print(f"Best model: {best_model_path}")


if __name__ == "__main__":
    main()
