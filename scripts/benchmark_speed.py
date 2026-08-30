"""
속도 벤치마크 스크립트 (Phase 4)
- 신뢰할 수 있는 추론 속도 측정: warmup + GPU synchronize + 분위수(p50/p95/p99)
- 모델 추론만 vs end-to-end(전처리+추론+후처리) 둘 다 측정
- 결과를 화면 출력 + speed_benchmark.csv에 누적

측정 원칙:
  1) warmup으로 초기 느린 구간 제거
  2) torch.cuda.synchronize()로 GPU 비동기 실행을 정확히 대기
  3) 평균이 아닌 p50/p95/p99로 튐까지 관찰
  4) batch=1 고정 (실시간=프레임 단위 처리 가정)

실행:
  python3 scripts/benchmark_speed.py --model runs/detect/E3-4_combined/weights/best.pt --tag E3-4 --imgsz 640
"""

import argparse
import csv
import time
import statistics
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"


def percentile(data, p):
    return float(np.percentile(data, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=50, help="측정 전 예열 횟수")
    ap.add_argument("--runs", type=int, default=200, help="측정 반복 횟수")
    args = ap.parse_args()

    model = YOLO(args.model)

    # 측정용 샘플 이미지 하나 준비 (val에서 아무거나)
    sample = next((DATA / "images" / "val").glob("*.jpg"))
    print(f"\n{'='*50}\n[{args.tag}] 속도 벤치마크\n{'='*50}")
    print(f"모델: {args.model}")
    print(f"해상도: {args.imgsz}, warmup: {args.warmup}, 측정: {args.runs}회, batch=1")
    print(f"샘플 이미지: {sample.name}\n")

    # ── Warmup ──────────────────────────────────
    print(f"Warmup {args.warmup}회...")
    for _ in range(args.warmup):
        model(str(sample), imgsz=args.imgsz, verbose=False)
    torch.cuda.synchronize()

    # ── 측정 1: end-to-end (전처리+추론+후처리) ──
    e2e_times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        model(str(sample), imgsz=args.imgsz, verbose=False)
        torch.cuda.synchronize()   # GPU 완료까지 대기 (핵심!)
        t1 = time.perf_counter()
        e2e_times.append((t1 - t0) * 1000)  # ms

    # ── 측정 2: 순수 추론 시간 (ultralytics 내부 계측) ──
    # results[0].speed 에 preprocess/inference/postprocess(ms)가 들어있음
    infer_times = []
    for _ in range(args.runs):
        r = model(str(sample), imgsz=args.imgsz, verbose=False)
        torch.cuda.synchronize()
        infer_times.append(r[0].speed["inference"])  # 순수 추론만

    # ── 통계 ────────────────────────────────────
    def stats(times):
        return {
            "mean": statistics.mean(times),
            "p50": percentile(times, 50),
            "p95": percentile(times, 95),
            "p99": percentile(times, 99),
        }

    e2e = stats(e2e_times)
    inf = stats(infer_times)
    fps_e2e = 1000.0 / e2e["p50"]
    fps_inf = 1000.0 / inf["p50"]

    print("── 결과 (단위: ms, 낮을수록 빠름) ──")
    print(f"{'구분':16s} {'mean':>8s} {'p50':>8s} {'p95':>8s} {'p99':>8s}")
    print(f"{'순수 추론':16s} {inf['mean']:8.2f} {inf['p50']:8.2f} {inf['p95']:8.2f} {inf['p99']:8.2f}")
    print(f"{'e2e(전+추+후)':16s} {e2e['mean']:8.2f} {e2e['p50']:8.2f} {e2e['p95']:8.2f} {e2e['p99']:8.2f}")
    print(f"\n── FPS (p50 기준) ──")
    print(f"  순수 추론: {fps_inf:.1f} FPS")
    print(f"  e2e:      {fps_e2e:.1f} FPS   ← 실제 파이프라인 속도")

    # VRAM (추론 시)
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\n  추론 peak VRAM: {vram_mb:.0f} MB")
    else:
        vram_mb = 0

    # ── CSV 저장 ────────────────────────────────
    csv_path = PROJECT / "speed_benchmark.csv"
    header = ["tag", "model", "imgsz",
              "inf_p50_ms", "inf_p95_ms", "e2e_p50_ms", "e2e_p95_ms",
              "fps_inf", "fps_e2e", "vram_mb"]
    row = {
        "tag": args.tag, "model": args.model, "imgsz": args.imgsz,
        "inf_p50_ms": f"{inf['p50']:.2f}", "inf_p95_ms": f"{inf['p95']:.2f}",
        "e2e_p50_ms": f"{e2e['p50']:.2f}", "e2e_p95_ms": f"{e2e['p95']:.2f}",
        "fps_inf": f"{fps_inf:.1f}", "fps_e2e": f"{fps_e2e:.1f}",
        "vram_mb": f"{vram_mb:.0f}",
    }
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