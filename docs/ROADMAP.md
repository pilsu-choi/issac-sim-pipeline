# Roadmap

## v0.1 - MLOps Skeleton

- [x] Isaac Sim synthetic data generation script
- [x] YOLO training entrypoint
- [x] MLflow experiment tracking
- [x] FastAPI inference service
- [x] Prometheus metrics endpoint
- [x] Docker Compose stack

## v0.2 - Dataset Quality

- [ ] Replace primitives with humanoid/workcell USD assets
- [ ] Add head-mounted camera calibration metadata
- [ ] Export annotations in YOLO format
- [ ] Add train/validation split script
- [ ] Add dataset card with sample images and class distribution
- [ ] Add DVC for dataset versioning

## v0.3 - Model Evaluation

- [ ] Add mAP, precision, recall report
- [ ] Compare synthetic-only and small real egocentric fine-tuning data
- [ ] Evaluate domain randomization for camera height, pitch, lighting, and object occlusion
- [ ] Register best model in MLflow Model Registry
- [ ] Add batch inference evaluation script

## v0.4 - project Polish

- [ ] Add architecture diagram
- [ ] Add demo GIF or screenshots
- [x] Add Grafana dashboard JSON
- [ ] Add Korean/English project summary

## Optional Extensions

- [ ] Cosmos-based synthetic video augmentation
- [ ] Isaac Lab reinforcement learning pipeline
- [ ] ROS2 inference node integration
