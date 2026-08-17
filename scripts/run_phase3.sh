#!/bin/bash
# Phase 3 — 데이터 전략 실험 (freeze=0 고정, 데이터/증강만 변경)
#
# E3-1(기준)은 Phase 2의 E2-1_freeze0과 동일하므로 재실행 안 함.
# 여기서는 E3-2, E3-3, E3-4 세 개만 실행.
#
# 공통: YOLOv8s, 30 epoch, imgsz=640, batch=16, seed=0, freeze=0
#
# 사용법:
#   python3 scripts/prepare_phase3.py --night_mult 2   # 먼저 오버샘플링 목록 생성
#   chmod +x scripts/run_phase3.sh
#   nohup bash scripts/run_phase3.sh > logs/phase3.log 2>&1 &
#   tail -f logs/phase3.log

set -u
cd ~/workspace/CV_project
mkdir -p logs

MODEL=yolov8s.pt
EPOCHS=30
IMGSZ=640
BATCH=16
SEED=0
WORKERS=4

echo "========================================"
echo "Phase 3 데이터 전략 실험 시작: $(date)"
echo "========================================"

# ── E3-2: 야간 오버샘플링 x2 ────────────────────────
echo ""
echo "[E3-2_night_oversample] 시작: $(date)"
yolo detect train \
  data=bdd_night_oversample_x2.yaml \
  model=$MODEL epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH seed=$SEED workers=$WORKERS freeze=0 \
  name=E3-2_night_oversample \
  || echo "[E3-2] ⚠️ 실패 — 다음으로 진행"
echo "[E3-2] 종료: $(date)"

# ── E3-3: 야간 특화 증강 (밝기/대비/노이즈 강화) ────────
# hsv_v: 밝기 변화 폭 (기본 0.4 → 0.6으로 어둡게도 자주)
# hsv_h/hsv_s: 색조/채도 변화
# 기본 데이터는 원본(bdd.yaml), 증강 옵션만 강화
echo ""
echo "[E3-3_night_aug] 시작: $(date)"
yolo detect train \
  data=bdd.yaml \
  model=$MODEL epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH seed=$SEED workers=$WORKERS freeze=0 \
  hsv_v=0.6 hsv_s=0.8 hsv_h=0.02 \
  name=E3-3_night_aug \
  || echo "[E3-3] ⚠️ 실패 — 다음으로 진행"
echo "[E3-3] 종료: $(date)"

# ── E3-4: 오버샘플링 + 증강 조합 ────────────────────
echo ""
echo "[E3-4_combined] 시작: $(date)"
yolo detect train \
  data=bdd_night_oversample_x2.yaml \
  model=$MODEL epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH seed=$SEED workers=$WORKERS freeze=0 \
  hsv_v=0.6 hsv_s=0.8 hsv_h=0.02 \
  name=E3-4_combined \
  || echo "[E3-4] ⚠️ 실패 — 다음으로 진행"
echo "[E3-4] 종료: $(date)"

echo ""
echo "========================================"
echo "Phase 3 전체 종료: $(date)"
echo "결과 확인: ls runs/detect/E3-*"
echo "========================================"