"""
개선 효과 시각 비교 스크립트 (Phase 3)
- 같은 야간 이미지에 대해 두 모델(기준 vs 개선)의 예측을 나란히 비교
- 왼쪽: 모델 A 예측, 오른쪽: 모델 B 예측 (정답=빨강, 예측=초록)
- 개선 모델이 놓쳤던 것을 더 잡는지 눈으로 확인

실행:
  python3 scripts/compare_models.py \
    --model_a runs/detect/E2-1_freeze0/weights/best.pt --name_a "baseline" \
    --model_b runs/detect/E3-4_combined/weights/best.pt --name_b "night_strategy" \
    --cond night --n 6
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
SPLITS = DATA / "splits"

CLASS_NAMES = [
    "pedestrian", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign",
]
IMG_W, IMG_H = 1280, 720
GT_COLOR = (0, 0, 255)      # 빨강: 정답
PRED_COLOR = (0, 255, 0)    # 초록: 예측


def draw(img, label_path, result, conf_th, title):
    """한 장에 정답(빨강)+예측(초록) 그리고 상단에 제목/카운트"""
    vis = img.copy()
    # 정답
    n_gt = 0
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            cls, xc, yc, w, h = map(float, line.split())
            x1 = int((xc - w/2)*IMG_W); y1 = int((yc - h/2)*IMG_H)
            x2 = int((xc + w/2)*IMG_W); y2 = int((yc + h/2)*IMG_H)
            cv2.rectangle(vis, (x1, y1), (x2, y2), GT_COLOR, 2)
            n_gt += 1
    # 예측
    n_pred = 0
    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < conf_th:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(vis, (x1, y1), (x2, y2), PRED_COLOR, 2)
        n_pred += 1
    # 상단 제목 바
    bar = np.zeros((40, vis.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, f"{title}  GT(red)={n_gt}  Pred(green)={n_pred}",
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return np.vstack([bar, vis]), n_gt, n_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", required=True)
    ap.add_argument("--name_a", default="A")
    ap.add_argument("--model_b", required=True)
    ap.add_argument("--name_b", default="B")
    ap.add_argument("--cond", default="night")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    tod = {"daytime", "night", "dawn_dusk"}
    prefix = "val_tod_" if args.cond in tod else "val_weather_"
    imgs = (SPLITS / f"{prefix}{args.cond}.txt").read_text().splitlines()
    random.seed(42)
    picks = random.sample(imgs, min(args.n, len(imgs)))

    model_a = YOLO(args.model_a)
    model_b = YOLO(args.model_b)
    out_dir = DATA / "compare" / args.cond
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{args.cond}] {len(picks)}장 비교 → {out_dir}")
    print(f"왼쪽={args.name_a}, 오른쪽={args.name_b}  (빨강=정답, 초록=예측)\n")
    print(f"{'파일':22s} {args.name_a+' pred':>14s} {args.name_b+' pred':>16s}")
    print("-" * 56)

    for img_path in picks:
        img_path = Path(img_path)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        label_path = DATA / "labels" / "val" / f"{img_path.stem}.txt"

        ra = model_a(str(img_path), verbose=False, conf=args.conf)[0]
        rb = model_b(str(img_path), verbose=False, conf=args.conf)[0]

        va, gt_a, pred_a = draw(img, label_path, ra, args.conf, args.name_a)
        vb, gt_b, pred_b = draw(img, label_path, rb, args.conf, args.name_b)

        # 좌우로 붙이기 (사이 흰 구분선)
        sep = np.full((va.shape[0], 6, 3), 255, dtype=np.uint8)
        combined = np.hstack([va, sep, vb])

        out_path = out_dir / f"{img_path.stem}_compare.jpg"
        cv2.imwrite(str(out_path), combined)
        print(f"{img_path.stem:22s} {pred_a:>14d} {pred_b:>16d}")

    print("-" * 56)
    print(f"\nVS Code에서 {out_dir} 열어 좌우 비교하세요.")
    print("→ 오른쪽(개선)이 왼쪽(기준)보다 초록 박스를 더 많이/정확히 잡으면 개선 성공")


if __name__ == "__main__":
    main()