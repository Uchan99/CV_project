"""
Zero-shot 평가 스크립트 (E1-0)
- COCO 사전학습 YOLO를 BDD로 파인튜닝하지 않고 그대로 평가
- COCO↔BDD 클래스 체계가 달라, 겹치는 7개 클래스만 매핑해 근사 비교 (방법 A)
- 조건별(주간/야간/우천 등) mAP + day-night gap 산출, experiments.csv에 누적

핵심 아이디어 (방법 A):
- BDD 라벨(.txt)에서 매핑 대상 7개 클래스만 남기고 COCO 번호로 변환한
  "임시 라벨 세트"를 만든다.
- COCO 모델은 그 7개만 평가하도록 classes= 로 제한한다.
- 이렇게 하면 예측·정답의 클래스 번호 체계가 COCO로 통일되어 비교 가능.

실행:
  python3 scripts/eval_zeroshot.py --model yolov8s.pt --tag E1-0_zeroshot_s
"""

import argparse
import csv
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import yaml
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
SPLITS = DATA / "splits"

# ── COCO ↔ BDD 매핑 (겹치는 7개만) ────────────────────────────
# key: BDD 클래스 번호, value: COCO 클래스 번호
BDD_TO_COCO = {
    0: 0,   # pedestrian → person
    2: 2,   # car        → car
    3: 7,   # truck      → truck
    4: 5,   # bus        → bus
    6: 3,   # motorcycle → motorcycle
    7: 1,   # bicycle    → bicycle
    8: 9,   # traffic light → traffic light
}
# 평가 대상 COCO 클래스 번호 목록
COCO_EVAL_CLASSES = sorted(set(BDD_TO_COCO.values()))
# COCO 번호 → 표시 이름 (출력용)
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 9: "traffic light",
}

CONDITIONS = {
    "all":        "val_all.txt",
    "daytime":    "val_tod_daytime.txt",
    "night":      "val_tod_night.txt",
    "dawn_dusk":  "val_tod_dawn_dusk.txt",
    "rainy":      "val_weather_rainy.txt",
    "snowy":      "val_weather_snowy.txt",
    "clear":      "val_weather_clear.txt",
}


def build_coco_labels():
    """
    BDD 라벨(labels/val/*.txt)을 COCO 번호로 변환한 임시 라벨 폴더 생성.
    - 매핑 대상 7개 클래스만 남기고, 클래스 번호를 COCO 번호로 치환.
    - 좌표는 그대로 (정규화 형식 동일).
    반환: 임시 이미지 심볼릭 링크 루트 (images/val, labels/val 구조)
    """
    src_img = DATA / "images" / "val"
    src_lbl = DATA / "labels" / "val"
    tmp_root = DATA / "_zeroshot_tmp"
    tmp_img = tmp_root / "images" / "val"
    tmp_lbl = tmp_root / "labels" / "val"

    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_img.mkdir(parents=True)
    tmp_lbl.mkdir(parents=True)

    # 이미지는 심볼릭 링크로 (복사 안 함 → 빠르고 용량 0)
    for img in src_img.glob("*.jpg"):
        (tmp_img / img.name).symlink_to(img.resolve())

    # 라벨은 COCO 번호로 변환해 새로 씀
    n_converted = 0
    for lbl in src_lbl.glob("*.txt"):
        new_lines = []
        for line in lbl.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            bdd_cls = int(parts[0])
            if bdd_cls not in BDD_TO_COCO:
                continue  # 매핑 안 되는 클래스(rider, train, traffic sign) 제외
            coco_cls = BDD_TO_COCO[bdd_cls]
            new_lines.append(f"{coco_cls} {' '.join(parts[1:])}")
        (tmp_lbl / lbl.name).write_text("\n".join(new_lines))
        n_converted += 1

    print(f"임시 COCO 라벨 생성: {n_converted}개 → {tmp_lbl}")
    return tmp_root


def make_yaml(list_file, tmp_root):
    """조건별 목록을 val로 지정 + COCO 80클래스 이름 사용하는 yaml"""
    # 목록 파일의 경로를 임시 루트 기준으로 재작성
    imgs = list_file.read_text().splitlines()
    remapped = []
    for p in imgs:
        name = Path(p).name
        remapped.append(str(tmp_root / "images" / "val" / name))
    tmp_list = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    tmp_list.write("\n".join(remapped))
    tmp_list.close()

    cfg = {
        "path": str(tmp_root),
        "train": tmp_list.name,   # 형식 통과용
        "val": tmp_list.name,
        "nc": 80,                 # COCO 80클래스 (형식 통과용)
    }
    tmp_yaml = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(cfg, tmp_yaml)
    tmp_yaml.close()
    return tmp_yaml.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt", help="COCO 사전학습 가중치")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    print(f"\n{'='*50}\n[{args.tag}] Zero-shot 평가 (COCO→BDD 7클래스 근사)\n{'='*50}")
    print(f"평가 클래스(COCO 번호): {COCO_EVAL_CLASSES}")
    print("제외: rider, train, traffic sign (COCO 대응 없음)\n")

    tmp_root = build_coco_labels()
    model = YOLO(args.model)

    results = {}
    for cond, fname in CONDITIONS.items():
        list_file = SPLITS / fname
        if not list_file.exists():
            print(f"  [{cond}] 목록 없음 → 건너뜀")
            continue
        n_imgs = len(list_file.read_text().splitlines())
        tmp_yaml = make_yaml(list_file, tmp_root)

        # classes= 로 평가 대상을 7개 COCO 클래스로 제한
        metrics = model.val(data=tmp_yaml, imgsz=args.imgsz, verbose=False,
                            split="val", plots=False, classes=COCO_EVAL_CLASSES)
        map50 = metrics.box.map50
        map5095 = metrics.box.map
        results[cond] = (n_imgs, map50, map5095)
        print(f"  [{cond:10s}] n={n_imgs:5d}  mAP50={map50:.4f}  mAP50-95={map5095:.4f}")

    gap50 = gap5095 = None
    if "daytime" in results and "night" in results:
        gap50 = results["daytime"][1] - results["night"][1]
        gap5095 = results["daytime"][2] - results["night"][2]
        print(f"\n  >>> day-night gap (mAP50)    = {gap50:.4f}")
        print(f"  >>> day-night gap (mAP50-95) = {gap5095:.4f}")

    # CSV 누적 (eval_by_condition.py와 동일한 헤더 구조)
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

    # 임시 폴더 정리
    shutil.rmtree(tmp_root)
    print(f"\n결과 저장: {csv_path}")
    print("임시 폴더 정리 완료")
    print("="*50)


if __name__ == "__main__":
    main()