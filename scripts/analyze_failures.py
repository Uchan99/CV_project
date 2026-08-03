"""
야간 실패 케이스 분석 스크립트 (Phase 1 마무리)
- baseline 모델을 야간 이미지에 돌려, 예측(초록) vs 정답(빨강)을 한 이미지에 겹쳐 그림
- 놓침/헛박스/위치오류를 눈으로 확인
- 주간 이미지도 몇 장 뽑아 대조하면 "야간이 왜 어려운지"가 보임

색 규칙:
  - 빨강(정답, GT): 실제로 있어야 할 박스
  - 초록(예측, Pred): 모델이 찾은 박스 (confidence 함께 표시)
  → 빨강만 있고 초록 없음 = 놓침(miss)
  → 초록만 있고 빨강 없음 = 헛박스(false positive)
  → 둘이 겹침 = 정답 검출

실행:
  python3 scripts/analyze_failures.py --model runs/detect/E1-1_baseline_s_50ep/weights/best.pt --cond night --n 8
  python3 scripts/analyze_failures.py --model runs/detect/E1-1_baseline_s_50ep/weights/best.pt --cond daytime --n 4
"""

import argparse
import random
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
SPLITS = DATA / "splits"

CLASS_NAMES = [
    "pedestrian", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign",
]
IMG_W, IMG_H = 1280, 720

GT_COLOR = (0, 0, 255)      # 빨강 (BGR): 정답
PRED_COLOR = (0, 255, 0)    # 초록: 예측


def draw_gt(img, label_path):
    """정답 라벨(빨강) 그리기"""
    if not label_path.exists():
        return 0
    n = 0
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls, xc, yc, w, h = map(float, line.split())
        x1 = int((xc - w / 2) * IMG_W); y1 = int((yc - h / 2) * IMG_H)
        x2 = int((xc + w / 2) * IMG_W); y2 = int((yc + h / 2) * IMG_H)
        cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, 2)
        n += 1
    return n


def draw_pred(img, result, conf_th=0.25):
    """예측(초록) 그리기 + confidence 표시"""
    n = 0
    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < conf_th:
            continue
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), PRED_COLOR, 2)
        label = f"{CLASS_NAMES[cls]} {conf:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRED_COLOR, 1)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cond", default="night",
                    help="daytime / night / rainy / snowy 등 (splits 접미사)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.25, help="예측 신뢰도 임계값")
    args = ap.parse_args()

    # 조건별 목록에서 이미지 뽑기
    tod_conds = {"daytime", "night", "dawn_dusk"}
    prefix = "val_tod_" if args.cond in tod_conds else "val_weather_"
    list_file = SPLITS / f"{prefix}{args.cond}.txt"
    if not list_file.exists():
        print(f"목록 없음: {list_file}")
        return

    imgs = list_file.read_text().splitlines()
    random.seed(42)
    picks = random.sample(imgs, min(args.n, len(imgs)))

    model = YOLO(args.model)
    out_dir = DATA / "failure_analysis" / args.cond
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{args.cond}] {len(picks)}장 분석 → {out_dir}\n")
    print(f"{'파일':30s} {'정답(GT)':>8s} {'예측(Pred)':>10s}  차이")
    print("-" * 60)

    total_gt = total_pred = 0
    for img_path in picks:
        img_path = Path(img_path)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        label_path = DATA / "labels" / "val" / f"{img_path.stem}.txt"

        # 예측 실행
        result = model(str(img_path), verbose=False, conf=args.conf)[0]

        n_gt = draw_gt(img, label_path)
        n_pred = draw_pred(img, result, conf_th=args.conf)
        total_gt += n_gt
        total_pred += n_pred

        diff = n_pred - n_gt
        flag = "← 많이 놓침" if diff <= -5 else ("← 헛박스 많음" if diff >= 5 else "")
        print(f"{img_path.stem:30s} {n_gt:>8d} {n_pred:>10d}  {diff:+d} {flag}")

        out_path = out_dir / f"{img_path.stem}_cmp.jpg"
        cv2.imwrite(str(out_path), img)

    print("-" * 60)
    print(f"{'합계':30s} {total_gt:>8d} {total_pred:>10d}")
    print(f"\n초록=예측, 빨강=정답")
    print(f"→ 빨강만 있는 곳=놓침, 초록만=헛박스, 겹침=정답 검출")
    print(f"VS Code에서 {out_dir} 열어 확인하세요.")


if __name__ == "__main__":
    main()