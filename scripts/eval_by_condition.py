"""
조건별 평가 스크립트
- splits/ 의 조건별 이미지 목록을 이용해, timeofday/weather 조건마다 mAP를 따로 측정
- day-night gap(주간 mAP - 야간 mAP)을 산출 → 이 프로젝트의 핵심 지표
- 결과를 화면 출력 + CSV 한 줄로 누적 저장

전제:
- 학습/사전학습된 모델 가중치(.pt) 경로
- data/splits/val_tod_*.txt, val_weather_*.txt 존재
- data/bdd_base.yaml (클래스 정의용, 아래 참조)

실행 예:
  # BDD 학습 모델 평가
  python3 scripts/eval_by_condition.py --model runs/detect/train-2/weights/best.pt --tag E1-1_bdd_1ep

  # COCO 사전학습 모델 zero-shot 평가
  python3 scripts/eval_by_condition.py --model yolov8n.pt --tag E1-0_zeroshot --coco
"""

import argparse
import csv
import tempfile
from pathlib import Path
from datetime import datetime

import yaml
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent.parent   # CV_project/
DATA = PROJECT / "data"
SPLITS = DATA / "splits"

# 평가할 조건 (splits 파일명의 접미사와 일치)
CONDITIONS = {
    "all":        "val_all.txt",
    "daytime":    "val_tod_daytime.txt",
    "night":      "val_tod_night.txt",
    "dawn_dusk":  "val_tod_dawn_dusk.txt",
    "rainy":      "val_weather_rainy.txt",
    "snowy":      "val_weather_snowy.txt",
    "clear":      "val_weather_clear.txt",
}

# BDD 클래스 (우리 학습 모델용)
BDD_NAMES = {
    0: "pedestrian", 1: "rider", 2: "car", 3: "truck", 4: "bus",
    5: "train", 6: "motorcycle", 7: "bicycle", 8: "traffic light", 9: "traffic sign",
}

# COCO 클래스 중 BDD와 대응되는 것 (zero-shot 평가 시 사용)
# COCO 인덱스: person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7, traffic light=9
# 주의: zero-shot은 클래스 체계가 달라 근사 비교임 (아래 설명 참조)
COCO_NAMES_FULL = None  # YOLO가 자체 COCO 이름을 사용


def make_temp_yaml(list_file, names, coco=False):
    """조건별 이미지 목록을 val로 지정하는 임시 yaml 생성"""
    cfg = {
        "path": str(DATA),
        "train": str(list_file),  # 형식 통과용 (평가엔 안 쓰임)
        "val": str(list_file),
    }
    if coco:
        # COCO 사전학습 모델은 자체 80클래스 이름을 씀 → names 생략
        pass
    else:
        cfg["names"] = names
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="가중치 .pt 경로")
    ap.add_argument("--tag", required=True, help="실험 ID (예: E1-1_bdd)")
    ap.add_argument("--coco", action="store_true", help="COCO 사전학습 zero-shot 평가")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = YOLO(args.model)
    names = None if args.coco else BDD_NAMES

    results = {}
    print(f"\n{'='*50}\n[{args.tag}] 조건별 평가 시작\n{'='*50}")

    for cond, fname in CONDITIONS.items():
        list_file = SPLITS / fname
        if not list_file.exists():
            print(f"  [{cond}] 목록 없음 → 건너뜀 ({fname})")
            continue

        n_imgs = len(list_file.read_text().splitlines())
        tmp_yaml = make_temp_yaml(list_file, names, coco=args.coco)

        # split='val' 지정, verbose 최소화
        metrics = model.val(data=tmp_yaml, imgsz=args.imgsz, verbose=False,
                            split="val", plots=False)
        map50 = metrics.box.map50
        map5095 = metrics.box.map
        results[cond] = (n_imgs, map50, map5095)
        print(f"  [{cond:10s}] n={n_imgs:5d}  mAP50={map50:.4f}  mAP50-95={map5095:.4f}")

    # day-night gap 계산
    gap50 = gap5095 = None
    if "daytime" in results and "night" in results:
        gap50 = results["daytime"][1] - results["night"][1]
        gap5095 = results["daytime"][2] - results["night"][2]
        print(f"\n  >>> day-night gap (mAP50)    = {gap50:.4f}")
        print(f"  >>> day-night gap (mAP50-95) = {gap5095:.4f}")

    # CSV 누적 저장
    csv_path = PROJECT / "experiments.csv"
    header = ["tag", "date", "model", "imgsz"]
    for c in CONDITIONS:
        header += [f"{c}_map50", f"{c}_map5095", f"{c}_n"]
    header += ["gap_map50", "gap_map5095"]

    row = {"tag": args.tag, "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "model": args.model, "imgsz": args.imgsz}
    for c in CONDITIONS:
        if c in results:
            n, m50, m5095 = results[c]
            row[f"{c}_map50"] = f"{m50:.4f}"
            row[f"{c}_map5095"] = f"{m5095:.4f}"
            row[f"{c}_n"] = n
    row["gap_map50"] = f"{gap50:.4f}" if gap50 is not None else ""
    row["gap_map5095"] = f"{gap5095:.4f}" if gap5095 is not None else ""

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(f"\n결과 저장: {csv_path}")
    print("="*50)


if __name__ == "__main__":
    main()