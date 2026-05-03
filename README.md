# Humanoid Robot Perception Synthetic Data MLOps

Isaac Sim으로 휴머노이드 로봇의 head-mounted camera 시점 합성 데이터를 생성하고, 객체 탐지 모델을 학습한 뒤 API로 배포하고 모니터링하는 MLOps 프로젝트입니다.

## 목표

이 프로젝트는 휴머노이드 로봇이 작업 환경에서 바라보는 데이터를 synthetic data pipeline으로 생성해, 실제 라벨링 데이터 부족 문제를 해결하는 흐름을 보여줍니다.

- Isaac Sim Replicator 기반 humanoid egocentric 이미지/라벨 생성
- 머리 장착 카메라 시점의 depth, segmentation, bounding box 생성
- YOLO 객체 탐지 모델 학습
- MLflow 기반 실험 추적
- FastAPI 기반 추론 API
- Prometheus/Grafana 기반 서빙 모니터링
- Docker Compose 기반 재현 가능한 실행 환경

## Architecture

```text
Isaac Sim Replicator
  -> humanoid egocentric perception dataset
  -> YOLO training
  -> MLflow experiment tracking
  -> FastAPI inference service
  -> Prometheus/Grafana monitoring
```

## Project Structure

```text
.
├── configs/
│   ├── dataset.yaml
│   └── training.yaml
├── docs/
│   └── ROADMAP.md
├── monitoring/
│   └── prometheus.yml
├── scripts/
│   └── generate_synthetic_dataset.py
├── src/
│   ├── api/
│   │   └── app.py
│   └── training/
│       └── train_yolo.py
├── docker-compose.yml
├── Dockerfile.api
├── Makefile
└── pyproject.toml
```

## 0. Python 환경

이 프로젝트는 `uv`를 기준으로 의존성을 관리합니다.

```bash
cd issac_sim_tutorial
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## 1. Isaac Sim 데이터 생성

Isaac Sim Python으로 synthetic dataset을 생성합니다. 외장하드에 설치된 Isaac Sim의 Python 실행 파일을 환경변수로 지정해서 사용합니다.

```bash
export ISAAC_SIM_PYTHON="/path/to/isaac-sim/python.sh" # ex)/media/pilsu/ced47498-a1cc-49bb-9d28-030ba0e417ee/isaac-sim/python.sh
make generate-data
```

Isaac Sim GUI를 띄우려면 아래 명령을 사용합니다.

```bash
make generate-data-gui
```

출력 예시는 `data/synthetic/humanoid_egocentric_perception` 아래에 저장됩니다.

## 2. 모델 학습

```bash
uv run python src/training/train_yolo.py --config configs/training.yaml
```

학습 설정은 `configs/training.yaml`에서 관리합니다. 학습 결과와 주요 metric은 MLflow에 기록됩니다.

## 3. 추론 API 실행

```bash
export MODEL_PATH="runs/detect/train/weights/best.pt"
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

헬스 체크:

```bash
curl http://localhost:8000/health
```

이미지 추론:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sample.jpg"
```

## 4. MLOps Stack 실행

```bash
docker compose up --build
```

- FastAPI: http://localhost:8000
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Project Message

> Built an end-to-end humanoid egocentric perception MLOps pipeline using NVIDIA Isaac Sim, YOLO, MLflow, FastAPI, Docker, and Prometheus/Grafana to address labeled data scarcity in robotics perception systems.

