"""
BDD100K (이미지별 개별 json 버전) 전처리 스크립트
- 각 json → YOLO 포맷 .txt (정규화 좌표)로 변환
- weather / timeofday 속성별 이미지 목록(txt) 생성
- 조건별 통계 출력

전제 폴더 구조:
  data/100k/train/xxxx.jpg + xxxx.json  (짝으로 존재)
  data/100k/val/xxxx.jpg   + xxxx.json

실행:
  python3 prepare_bdd.py --root ~/workspace/CV_project/data/100k --split train
  python3 prepare_bdd.py --root ~/workspace/CV_project/data/100k --split val
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

# BDD100K detection 10개 클래스 → YOLO 클래스 인덱스
# (train/val 전 이미지에서 등장하는 표준 카테고리)
CLASS_MAP = {
    "pedestrian": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "traffic light": 8,
    "traffic sign": 9,
    # 2018 라벨 호환용 별칭 (혹시 다른 이름으로 들어온 경우 대비)
    "person": 0,
    "motor": 6,
    "bike": 7,
}

IMG_W, IMG_H = 1280, 720  # BDD100K 고정 해상도


def convert_box(b):
    """box2d(x1,y1,x2,y2 픽셀) → YOLO(x_center,y_center,w,h 정규화)"""
    x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
    # 좌표 순서 보정 및 경계 클리핑
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(x1, IMG_W)); x2 = max(0, min(x2, IMG_W))
    y1 = max(0, min(y1, IMG_H)); y2 = max(0, min(y2, IMG_H))
    xc = (x1 + x2) / 2 / IMG_W
    yc = (y1 + y2) / 2 / IMG_H
    w = (x2 - x1) / IMG_W
    h = (y2 - y1) / IMG_H
    return xc, yc, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="예: ~/workspace/CV_project/data/100k")
    ap.add_argument("--split", required=True, choices=["train", "val"])
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    split_dir = root / args.split
    label_out = root.parent / "labels_yolo" / args.split   # data/labels_yolo/train/*.txt
    list_out = root.parent / "splits"                       # data/splits/*.txt
    label_out.mkdir(parents=True, exist_ok=True)
    list_out.mkdir(parents=True, exist_ok=True)

    jsons = sorted(split_dir.glob("*.json"))
    print(f"[{args.split}] json {len(jsons)}개 발견")

    # 속성별 이미지 목록
    by_weather = defaultdict(list)
    by_timeofday = defaultdict(list)
    weather_ct, tod_ct, scene_ct = Counter(), Counter(), Counter()
    unknown_cats = Counter()
    n_boxes = 0
    n_no_attr = 0

    for jp in jsons:
        stem = jp.stem
        img_path = split_dir / f"{stem}.jpg"
        if not img_path.exists():
            continue  # 이미지 없는 라벨은 건너뜀

        d = json.load(open(jp))
        attrs = d.get("attributes") or {}
        weather = attrs.get("weather", "undefined")
        tod = attrs.get("timeofday", "undefined")
        scene = attrs.get("scene", "undefined")
        if not attrs:
            n_no_attr += 1

        weather_ct[weather] += 1
        tod_ct[tod] += 1
        scene_ct[scene] += 1

        # 이미지 절대경로를 목록에 기록 (YOLO는 이미지 경로 기준)
        by_weather[weather].append(str(img_path))
        by_timeofday[tod].append(str(img_path))

        # 박스 변환
        lines = []
        frames = d.get("frames", [])
        objects = frames[0].get("objects", []) if frames else d.get("labels", [])
        for obj in objects:
            cat = obj.get("category")
            if "box2d" not in obj or obj["box2d"] is None:
                continue  # 차선/영역 등 박스 없는 항목 스킵
            if cat not in CLASS_MAP:
                unknown_cats[cat] += 1
                continue
            cls = CLASS_MAP[cat]
            xc, yc, w, h = convert_box(obj["box2d"])
            if w <= 0 or h <= 0:
                continue
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            n_boxes += 1

        (label_out / f"{stem}.txt").write_text("\n".join(lines))

    # 속성별 이미지 목록 저장
    for weather, imgs in by_weather.items():
        safe = weather.replace(" ", "_").replace("/", "_")
        (list_out / f"{args.split}_weather_{safe}.txt").write_text("\n".join(imgs))
    for tod, imgs in by_timeofday.items():
        safe = tod.replace(" ", "_").replace("/", "_")
        (list_out / f"{args.split}_tod_{safe}.txt").write_text("\n".join(imgs))

    # 전체 목록도 저장
    all_imgs = [str(split_dir / f"{jp.stem}.jpg") for jp in jsons
                if (split_dir / f"{jp.stem}.jpg").exists()]
    (list_out / f"{args.split}_all.txt").write_text("\n".join(all_imgs))

    # 통계 출력
    print(f"\n=== [{args.split}] 통계 ===")
    print(f"총 박스 수: {n_boxes}")
    print(f"attributes 없는 이미지: {n_no_attr}")
    print(f"\n[timeofday]")
    for k, v in tod_ct.most_common():
        print(f"  {k:12s}: {v:6d} ({v/len(jsons)*100:.1f}%)")
    print(f"\n[weather]")
    for k, v in weather_ct.most_common():
        print(f"  {k:12s}: {v:6d} ({v/len(jsons)*100:.1f}%)")
    print(f"\n[scene]")
    for k, v in scene_ct.most_common():
        print(f"  {k:12s}: {v:6d}")
    if unknown_cats:
        print(f"\n[⚠️ 매핑 안 된 category] {dict(unknown_cats)}")
    print(f"\n라벨 저장: {label_out}")
    print(f"목록 저장: {list_out}")


if __name__ == "__main__":
    main()