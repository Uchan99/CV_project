#!/bin/bash
# Phase 4 추가 — YOLOv8s 416 전용 학습 (E4-5)
#
# 목적: "640 학습→416 추론"(E4-2) vs "416 학습→416 추론"(E4-5) 비교.
#       train-test 해상도 일치가 성능과 day-night gap 역전에 주는 영향 검증.
#
# E3-4와 완전히 동일한 조건 (오버샘플링+증강, freeze=0), imgsz만 640→416.
#
# 사용법:
#   chmod +x scripts/run_phase4_s416.sh
#   nohup bash scripts/run_phase4_s416.sh > logs/phase4_s416.log 2>&1 &
#   tail -f logs/phase4_s416.log

set -u
cd ~/workspace/CV_project
source .venv/bin/activate      # nohup 환경에서도 yolo 인식
mkdir -p logs

echo "========================================"
echo "Phase 4 추가: YOLOv8s 416 학습 시작: $(date)"
echo "비교 목적: 640학습→416추론(E4-2) vs 416학습→416추론(E4-5)"
echo "========================================"

# E3-4와 동일 설정, imgsz만 416
yolo detect train \
  data=bdd_night_oversample_x2.yaml \
  model=yolov8s.pt \
  epochs=30 \
  imgsz=416 \
  batch=16 \
  seed=0 \
  workers=4 \
  freeze=0 \
  hsv_v=0.6 hsv_s=0.8 hsv_h=0.02 \
  name=E4-5_s416_trained \
  || echo "⚠️ 학습 실패"

echo ""
echo "========================================"
echo "학습 종료: $(date)"
echo "결과: runs/detect/E4-5_s416_trained/weights/best.pt"
echo "다음: 416으로 평가 후 E4-2와 비교"
echo "========================================"