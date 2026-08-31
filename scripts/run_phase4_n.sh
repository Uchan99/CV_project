#!/bin/bash
# Phase 4 — YOLOv8n 학습 (엣지 디바이스 가정)
#
# E3-4(YOLOv8s + 오버샘플링 + 증강)와 공정 비교를 위해,
# YOLOv8n을 "동일한 데이터 전략"으로 학습.
# 학습 후 640/416 두 해상도로 평가 → E4-3, E4-4 채움.
#
# 사용법:
#   chmod +x scripts/run_phase4_n.sh
#   nohup bash scripts/run_phase4_n.sh > logs/phase4_n.log 2>&1 &
#   tail -f logs/phase4_n.log

set -u
cd ~/workspace/CV_project
source .venv/bin/activate      # nohup 환경에서도 yolo 인식 (지난 교훈)
mkdir -p logs

echo "========================================"
echo "Phase 4 YOLOv8n 학습 시작: $(date)"
echo "데이터 전략: 야간 오버샘플링 x2 + 야간 증강 (E3-4와 동일)"
echo "========================================"

# E3-4와 동일한 설정: 오버샘플링 데이터 + 증강, 단 model만 n
yolo detect train \
  data=bdd_night_oversample_x2.yaml \
  model=yolov8n.pt \
  epochs=30 \
  imgsz=640 \
  batch=16 \
  seed=0 \
  workers=4 \
  freeze=0 \
  hsv_v=0.6 hsv_s=0.8 hsv_h=0.02 \
  name=E4_yolov8n_combined \
  || echo "⚠️ 학습 실패"

echo ""
echo "========================================"
echo "학습 종료: $(date)"
echo "결과: runs/detect/E4_yolov8n_combined/weights/best.pt"
echo "다음: 640/416 두 해상도로 평가 (E4-3, E4-4)"
echo "========================================"