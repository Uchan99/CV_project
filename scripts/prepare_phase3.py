"""
Phase 3 데이터 전략 준비 스크립트
- 야간 오버샘플링용 train 이미지 목록 생성 (야간 이미지를 N배 중복 포함)
- 그 목록을 train으로 쓰는 yaml 생성

원리:
- YOLO는 train에 "이미지 경로 목록(txt)"을 받을 수 있음
- 이 목록에 야간 이미지 경로를 여러 번 써넣으면, 학습 시 야간을 그만큼 자주 봄 (=오버샘플링)
- val은 기존과 동일하게 고정 (평가 조건 불변)

실행:
  python3 scripts/prepare_phase3.py --night_mult 2
  → data/splits/train_night_oversample_x2.txt 생성
  → bdd_night_oversample_x2.yaml 생성
"""

import argparse
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
SPLITS = DATA / "splits"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--night_mult", type=int, default=2,
                    help="야간 이미지를 몇 배로 넣을지 (2 = 야간을 2번 포함)")
    args = ap.parse_args()

    # 전체 train 목록과 야간 train 목록 읽기
    all_list = (SPLITS / "train_all.txt").read_text().splitlines()
    night_list = (SPLITS / "train_tod_night.txt").read_text().splitlines()

    all_list = [x for x in all_list if x.strip()]
    night_list = [x for x in night_list if x.strip()]

    print(f"전체 train: {len(all_list)}장")
    print(f"야간 train: {len(night_list)}장 ({len(night_list)/len(all_list)*100:.1f}%)")

    # 오버샘플링: 전체 목록 + 야간을 (night_mult - 1)번 추가
    # night_mult=2 → 야간이 원래 1번 + 추가 1번 = 총 2번 등장
    oversampled = list(all_list)
    for _ in range(args.night_mult - 1):
        oversampled += night_list

    # 새 야간 비중 계산
    total = len(oversampled)
    night_count = len(night_list) * args.night_mult
    print(f"\n오버샘플링 후:")
    print(f"  총 이미지(중복 포함): {total}장")
    print(f"  야간 등장 횟수: {night_count}회 ({night_count/total*100:.1f}%)")

    # 목록 파일 저장
    out_list = SPLITS / f"train_night_oversample_x{args.night_mult}.txt"
    out_list.write_text("\n".join(oversampled))
    print(f"\n목록 저장: {out_list}")

    # yaml 생성
    yaml_content = f"""# Phase 3 - 야간 오버샘플링 x{args.night_mult}
path: {DATA}
train: {out_list}
val: images/val

names:
  0: pedestrian
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motorcycle
  7: bicycle
  8: traffic light
  9: traffic sign
"""
    out_yaml = PROJECT / f"bdd_night_oversample_x{args.night_mult}.yaml"
    out_yaml.write_text(yaml_content)
    print(f"yaml 저장: {out_yaml}")


if __name__ == "__main__":
    main()