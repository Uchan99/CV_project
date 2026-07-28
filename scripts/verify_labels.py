"""
라벨 시각 검증 스크립트
- YOLO 포맷으로 변환된 라벨(labels_yolo/*.txt)을 실제 이미지에 그려서 저장
- 주간/야간 등 조건을 섞어 뽑아, 변환이 제대로 됐는지 눈으로 확인
- 결과 이미지를 VS Code에서 열어 박스가 물체에 잘 맞는지 보면 됨

실행:
  python3 scripts/verify_labels.py --root data/100k --split train --n 8
  python3 scripts/verify_labels.py --root data/100k --split val --n 6
"""

import argparse
import random
from pathlib import Path

import cv2  # OpenCV: 이미지 읽기/그리기/저장

# 클래스 번호 → 이름 (prepare_bdd.py의 CLASS_MAP과 순서 일치)
CLASS_NAMES = [
    "pedestrian", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign",
]

# 클래스별 박스 색상 (BGR — OpenCV는 RGB가 아니라 BGR 순서)
COLORS = [
    (0, 0, 255), (0, 128, 255), (0, 255, 0), (255, 128, 0), (255, 0, 0),
    (255, 0, 255), (128, 0, 255), (0, 255, 255), (255, 255, 0), (128, 255, 0),
]

IMG_W, IMG_H = 1280, 720


def draw_one(img_path, label_path, out_path):
    """이미지 한 장에 YOLO 라벨 박스를 그려서 저장"""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ⚠️ 이미지 못 읽음: {img_path}")
        return False

    if not label_path.exists():
        print(f"  ⚠️ 라벨 없음: {label_path}")
        return False

    n_boxes = 0
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls, xc, yc, w, h = line.split()
        cls = int(cls)
        xc, yc, w, h = float(xc), float(yc), float(w), float(h)
        # 정규화 좌표 → 픽셀 좌표로 역변환 (그리기 위해)
        x1 = int((xc - w / 2) * IMG_W)
        y1 = int((yc - h / 2) * IMG_H)
        x2 = int((xc + w / 2) * IMG_W)
        y2 = int((yc + h / 2) * IMG_H)
        color = COLORS[cls % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, CLASS_NAMES[cls], (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        n_boxes += 1

    cv2.imwrite(str(out_path), img)
    print(f"  저장: {out_path.name}  (박스 {n_boxes}개)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="예: data/100k")
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--n", type=int, default=8, help="검증할 이미지 수")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    img_dir = root / args.split
    label_dir = root.parent / "labels_yolo" / args.split
    out_dir = root.parent / "verify" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = root.parent / "splits"

    # 주간/야간을 섞어서 뽑기 (조건별로 변환이 잘 됐는지 함께 확인)
    picks = []
    for cond in ["daytime", "night"]:
        list_file = splits_dir / f"{args.split}_tod_{cond}.txt"
        if list_file.exists():
            imgs = list_file.read_text().splitlines()
            random.seed(42)
            picks += random.sample(imgs, min(args.n // 2, len(imgs)))

    # 목록이 없으면 폴더에서 그냥 뽑기
    if not picks:
        picks = [str(p) for p in list(img_dir.glob("*.jpg"))[:args.n]]

    print(f"[{args.split}] {len(picks)}장 검증 → {out_dir}")
    ok = 0
    for img_path in picks:
        img_path = Path(img_path)
        label_path = label_dir / f"{img_path.stem}.txt"
        out_path = out_dir / f"{img_path.stem}_annotated.jpg"
        if draw_one(img_path, label_path, out_path):
            ok += 1

    print(f"\n완료: {ok}/{len(picks)}장 저장됨")
    print(f"VS Code에서 {out_dir} 폴더를 열어 이미지를 확인하세요.")
    print("→ 박스가 실제 차/사람/표지판 위치에 잘 맞으면 변환 성공입니다.")


if __name__ == "__main__":
    main()