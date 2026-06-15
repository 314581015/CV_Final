"""
FPS benchmark for all model variants.

Measures inference speed under identical conditions (same GPU, same batch size,
sequential execution). For C3TR models, builds the architecture from yolov10s.pt
then applies C3TR replacement — weight values don't affect FPS.

Usage:
    python benchmark_fps.py --device 0 --dataset datasets/furnas.yaml
"""

# ── Register custom modules ────────────────────────────────────────────────────
import ultralytics.nn.tasks as _tasks
from models.attention import CBAM, ECA, CoordAtt
from models.c3tr_lite import C3TRLite, LiteTransformerBlock, \
    replace_late_c2f_with_c3tr_lite, replace_indexed_c2f_with_c3tr_lite
_tasks.CBAM = CBAM
_tasks.ECA = ECA
_tasks.CoordAtt = CoordAtt
_tasks.C3TRLite = C3TRLite
_tasks.LiteTransformerBlock = LiteTransformerBlock

import torch
import time
import argparse
from pathlib import Path
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE  = 0
DATASET = "datasets/furnas.yaml"
IMGSZ   = 640
BATCH   = 1      # FPS 通常用 batch=1 量單張延遲
WARMUP  = 50     # warmup iterations（純 forward，不計入）
REPEAT  = 200    # 正式計時 iterations

C3TR_PAPER_LAYERS = [2, 4, 6, 8, 13, 16, 19]
C3TR_HEADS = 4

# ── Model definitions ──────────────────────────────────────────────────────────
# Each entry: (display_name, builder_fn)
# builder_fn() → YOLO model object with the right architecture

RUN_DIR = Path("runs/detect/runs")

def _load_best(run_name):
    p = RUN_DIR / run_name / "weights" / "best.pt"
    if not p.exists():
        raise FileNotFoundError(f"best.pt not found: {p}")
    return YOLO(str(p))

def _build_c3tr(num_blocks=None, paper=False):
    """Build C3TRLite model from yolov10s.pt (weights don't affect FPS)."""
    m = YOLO("yolov10s.pt")
    if paper:
        replace_indexed_c2f_with_c3tr_lite(m.model, C3TR_PAPER_LAYERS, heads=C3TR_HEADS)
    else:
        replace_late_c2f_with_c3tr_lite(m.model, num_blocks=num_blocks, heads=C3TR_HEADS)
    return m

def _build_c3tr_cbam(num_blocks=2):
    m = YOLO("models/yolov10s_cbam.yaml")
    replace_late_c2f_with_c3tr_lite(m.model, num_blocks=num_blocks, heads=C3TR_HEADS)
    return m

def _build_c3tr_eca_paper():
    # ECA yaml adds 3 attention modules at layers 17, 21, 25, shifting indices.
    # Original C3TR paper layers [2,4,6,8,13,16,19] remapped for ECA yaml:
    #   0-16 → same; 17-19 → +1; 20-22 → +2; 23 → +3
    # So 19 → 20 (only layer 19 is in the shifted range)
    eca_paper_layers = [2, 4, 6, 8, 13, 16, 20]
    m = YOLO("models/yolov10s_eca.yaml")
    replace_indexed_c2f_with_c3tr_lite(m.model, eca_paper_layers, heads=C3TR_HEADS)
    return m

MODELS = [
    ("Baseline",                    lambda: _load_best("furnas_baseline-2")),
    ("CBAM",                        lambda: _load_best("furnas_cbam-2")),
    ("CoordAtt",                    lambda: _load_best("furnas_ca-2")),
    ("ECA",                         lambda: _load_best("furnas_eca-2")),
    ("C3TR-lite (b=1)",             lambda: _build_c3tr(num_blocks=1)),
    ("C3TR-lite (b=2)",             lambda: _build_c3tr(num_blocks=2)),
    ("C3TR-lite (b=8)",             lambda: _build_c3tr(num_blocks=8)),
    ("C3TR-paper",                  lambda: _build_c3tr(paper=True)),
    ("C3TR-lite (b=2) + CBAM",      lambda: _build_c3tr_cbam(num_blocks=2)),
    ("C3TR-paper + ECA-paper",      lambda: _build_c3tr_eca_paper()),
]


def measure_fps(model: YOLO, device, imgsz=640, warmup=50, repeat=200) -> float:
    """Pure forward-pass FPS using random input (architecture speed only)."""
    dev = torch.device(f"cuda:{device}" if str(device) != "cpu" else "cpu")
    net = model.model.float().eval().to(dev)

    dummy = torch.zeros(1, 3, imgsz, imgsz, device=dev)

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            net(dummy)
    torch.cuda.synchronize(dev)

    # timed run
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            net(dummy)
    torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - t0

    fps = repeat / elapsed
    return fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",  default=None, type=int)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--warmup",  default=WARMUP, type=int)
    parser.add_argument("--repeat",  default=REPEAT, type=int)
    args = parser.parse_args()

    device  = args.device  if args.device  is not None else DEVICE
    warmup  = args.warmup
    repeat  = args.repeat

    print(f"\n{'='*60}")
    print(f"  FPS Benchmark  |  GPU:{device}  |  warmup={warmup}  repeat={repeat}")
    print(f"  batch=1, imgsz={IMGSZ}x{IMGSZ}")
    print(f"{'='*60}\n")
    print(f"  {'Model':<35} {'FPS':>8}  {'ms/img':>8}")
    print(f"  {'-'*55}")

    results = []
    for name, builder in MODELS:
        try:
            model = builder()
            fps = measure_fps(model, device, imgsz=IMGSZ, warmup=warmup, repeat=repeat)
            ms  = 1000.0 / fps
            print(f"  {name:<35} {fps:>8.1f}  {ms:>8.2f}")
            results.append((name, fps, ms))
            del model
            torch.cuda.empty_cache()
        except FileNotFoundError as e:
            print(f"  {name:<35} SKIP  ({e})")
        except Exception as e:
            print(f"  {name:<35} ERROR ({e})")

    print(f"\n  {'='*55}")
    print(f"  Results (sorted by FPS desc):")
    print(f"  {'─'*55}")
    for name, fps, ms in sorted(results, key=lambda x: -x[1]):
        print(f"  {name:<35} {fps:>8.1f} FPS  ({ms:.2f} ms/img)")
    print()


if __name__ == "__main__":
    main()
