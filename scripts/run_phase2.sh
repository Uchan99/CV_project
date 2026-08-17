#!/bin/bash
# Phase 2 — 전이학습 깊이(freeze) 실험 4개 순차 실행
#
# freeze만 바꾸고 나머지(데이터·epoch·seed·batch)는 전부 고정 → 공정 비교
# 각 학습은 독립적. 하나가 실패해도 다음으로 계속 진행됨.
#
# 사용법:
#   chmod +x scripts/run_phase2.sh
#   nohup bash scripts/run_phase2.sh > logs/phase2.log 2>&1 &
#   tail -f logs/phase2.log      # 진행 확인
#
# 결과: runs/detect/E2-{1,2,3,4}_* 각각에 저장

set -u  # 미정의 변수 사용 시 경고 (set -e는 안 씀: 하나 실패해도 계속 가야 하므로)

cd ~/workspace/CV_project
mkdir -p logs

# 공통 설정 (모든 실험 동일)
DATA=bdd.yaml
MODEL=yolov8s.pt
EPOCHS=30
IMGSZ=640
BATCH=16
SEED=0
WORKERS=4

echo "========================================"
echo "Phase 2 freeze 실험 시작: $(date)"
echo "공통: model=$MODEL epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH seed=$SEED"
echo "========================================"

# 실험 목록: "실험이름 freeze값"
declare -a EXPERIMENTS=(
  "E2-1_freeze0 0"
  "E2-2_freeze10 10"
  "E2-3_freeze15 15"
  "E2-4_freeze22 22"
)

for exp in "${EXPERIMENTS[@]}"; do
  # 이름과 freeze값 분리
  NAME=$(echo "$exp" | cut -d' ' -f1)
  FREEZE=$(echo "$exp" | cut -d' ' -f2)

  echo ""
  echo "----------------------------------------"
  echo "[$NAME] freeze=$FREEZE 학습 시작: $(date)"
  echo "----------------------------------------"

  # 학습 실행. || true 로 감싸서 실패해도 스크립트가 멈추지 않게 함
  yolo detect train \
    data=$DATA \
    model=$MODEL \
    epochs=$EPOCHS \
    imgsz=$IMGSZ \
    batch=$BATCH \
    seed=$SEED \
    workers=$WORKERS \
    freeze=$FREEZE \
    name=$NAME \
    || echo "[$NAME] ⚠️ 학습 실패 또는 중단 — 다음 실험으로 진행"

  echo "[$NAME] 종료: $(date)"
done

echo ""
echo "========================================"
echo "Phase 2 전체 종료: $(date)"
echo "결과 확인: ls runs/detect/E2-*"
echo "========================================"