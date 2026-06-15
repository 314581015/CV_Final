"""
Inference & evaluation script for YOLOv10s variants on VisDrone.

CONFIG (edit below):
  MODEL   → "baseline" | "cbam" | "cbam_p3" | "c3tr_lite" | "c3tr_paper"
  WEIGHTS → "best" | "last" | "/absolute/path/to/custom.pt"
  SPLITS  → which splits to evaluate
"""

# ── Register custom modules before importing YOLO ─────────────────────────────
import ultralytics.nn.tasks as _tasks
from models.attention import CBAM, ECA, CoordAtt
from models.c3tr_lite import C3TRLite, LiteTransformerBlock
_tasks.CBAM = CBAM
_tasks.ECA = ECA
_tasks.CoordAtt = CoordAtt
_tasks.C3TRLite = C3TRLite
_tasks.LiteTransformerBlock = LiteTransformerBlock

import time
import torch
from pathlib import Path
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these
# ══════════════════════════════════════════════════════════════════════════════

MODEL   = "baseline"          # "baseline" | "cbam" | "cbam_p3" | "c3tr_lite" | "c3tr_paper"
WEIGHTS = "best"                  # "best" | "last" | "/path/to/file.pt"
SPLITS  = ["train", "val", "test"]  # remove any you don't want
DATASET = "datasets/visdrone.yaml"
IMGSZ   = 640
BATCH   = 16
DEVICE  = 0                       # GPU index or "cpu"
CONF    = 0.001                   # keep low for proper mAP calculation
IOU     = 0.7
WORKERS = 4

# ══════════════════════════════════════════════════════════════════════════════

WEIGHTS_DIR = {
    "baseline": "runs/detect/runs/baseline/weights",
    "cbam":     "runs/detect/runs/cbam/weights",
    "cbam_p3":  "runs/detect/runs/cbam_p3/weights",
    "c3tr_lite": "runs/detect/runs/c3tr_lite/weights",
    "c3tr_paper": "runs/detect/runs/c3tr_paper/weights",
}

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def resolve_weights(model_name: str, weights: str) -> Path:
    p = Path(weights)
    if p.is_absolute() or p.exists():
        if not p.exists():
            raise FileNotFoundError(f"Weights not found: {p}")
        return p
    pt = Path(WEIGHTS_DIR[model_name]) / f"{weights}.pt"
    if not pt.exists():
        raise FileNotFoundError(
            f"Weights not found: {pt}\n"
            f"  Have you trained '{model_name}' yet?"
        )
    return pt


def print_model_stats(model: YOLO, weights_path: Path) -> None:
    print("\n" + "═" * 62)
    print("  MODEL STATISTICS")
    print("═" * 62)

    total_params     = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    num_layers       = len(list(model.model.modules()))

    print(f"  Architecture:           {MODEL}")
    print(f"  Weights:                {weights_path}")
    print(f"  Modules (total):        {num_layers:>12,}")
    print(f"  Parameters (total):     {total_params:>12,}  ({total_params/1e6:.2f} M)")
    print(f"  Parameters (trainable): {trainable_params:>12,}  ({trainable_params/1e6:.2f} M)")

    # FLOPs — try thop first, fall back to ultralytics info()
    try:
        from thop import profile as thop_profile
        dev = torch.device(f"cuda:{DEVICE}" if str(DEVICE) != "cpu" else "cpu")
        dummy = torch.zeros(1, 3, IMGSZ, IMGSZ, device=dev)
        model.model.to(dev)
        model.model.eval()
        with torch.no_grad():
            flops, _ = thop_profile(model.model, inputs=(dummy,), verbose=False)
        print(f"  GFLOPs (thop):          {flops / 1e9:>12.2f}")
    except Exception:
        # ultralytics model.info() prints and returns (layers, params, grads, flops)
        try:
            info = model.info(verbose=False, imgsz=IMGSZ)
            if isinstance(info, (list, tuple)) and len(info) >= 4:
                print(f"  GFLOPs (ultralytics):   {float(info[3]):>12.2f}")
            else:
                print("  GFLOPs: install `pip install thop` for FLOPs count")
        except Exception:
            print("  GFLOPs: unavailable")

    size_mb = weights_path.stat().st_size / 1e6
    print(f"  Model size (disk):      {size_mb:>12.2f} MB")
    print("═" * 62)


def eval_split(model: YOLO, split: str) -> dict:
    print(f"\n{'─' * 62}")
    print(f"  Evaluating split: {split.upper()}")
    print(f"{'─' * 62}")

    t0 = time.time()
    results = model.val(
        data=DATASET,
        split=split,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        conf=CONF,
        iou=IOU,
        workers=WORKERS,
        verbose=False,
        plots=False,
        save_json=False,
        save_hybrid=False,
    )
    elapsed = time.time() - t0

    m = {
        "mAP50":    float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "P":        float(results.box.mp),
        "R":        float(results.box.mr),
        "elapsed":  elapsed,
    }

    # per-class AP@0.5
    if hasattr(results.box, "ap_class_index") and results.box.ap50 is not None:
        m["per_class_ap50"] = results.box.ap50.tolist()
        m["class_index"]    = results.box.ap_class_index.tolist()

    # inference speed breakdown (ms per image)
    if hasattr(results, "speed") and results.speed:
        m["speed"] = dict(results.speed)

    return m


def print_summary(all_results: dict) -> None:
    print("\n" + "═" * 62)
    print("  EVALUATION SUMMARY")
    print("═" * 62)
    print(f"  {'Split':<8} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8} {'Time(s)':>9}")
    print("  " + "─" * 58)
    for split, m in all_results.items():
        print(
            f"  {split.upper():<8} "
            f"{m['mAP50']:>8.4f} "
            f"{m['mAP50-95']:>10.4f} "
            f"{m['P']:>10.4f} "
            f"{m['R']:>8.4f} "
            f"{m['elapsed']:>9.1f}"
        )
    print("  " + "─" * 58)

    # per-class breakdown (val split)
    for split in ["val", "test", "train"]:
        if split in all_results and "per_class_ap50" in all_results[split]:
            ap50s  = all_results[split]["per_class_ap50"]
            cidxs  = all_results[split].get("class_index", list(range(len(ap50s))))
            print(f"\n  Per-class AP@0.50 ({split.upper()}):")
            print(f"  {'Class':<20} {'AP50':>6}  Bar")
            print("  " + "─" * 45)
            for ci, ap in zip(cidxs, ap50s):
                name = CLASS_NAMES[ci] if ci < len(CLASS_NAMES) else f"cls{ci}"
                bar  = "█" * int(ap * 25)
                print(f"  {name:<20} {ap:6.4f}  {bar}")
            break  # only print once

    # inference speed (val → fps)
    for split in ["val", "test", "train"]:
        if split in all_results and "speed" in all_results[split]:
            spd      = all_results[split]["speed"]
            total_ms = sum(spd.values())
            print(f"\n  Inference speed ({split.upper()}, per image):")
            for k, v in spd.items():
                print(f"    {k:<16} {v:6.2f} ms")
            print(f"    {'─'*25}")
            print(f"    {'total':<16} {total_ms:6.2f} ms  →  {1000/total_ms:.1f} FPS")
            break

    print("\n" + "═" * 62)


def main() -> None:
    weights_path = resolve_weights(MODEL, WEIGHTS)

    print(f"\n{'═' * 62}")
    print(f"  MODEL:   {MODEL}")
    print(f"  WEIGHTS: {weights_path}")
    print(f"  SPLITS:  {SPLITS}")
    print(f"  IMGSZ:   {IMGSZ}  |  BATCH: {BATCH}  |  DEVICE: {DEVICE}")
    print(f"  CONF:    {CONF}   |  IOU:   {IOU}")
    print(f"{'═' * 62}\n")

    model = YOLO(str(weights_path))
    print_model_stats(model, weights_path)

    all_results = {}
    for split in SPLITS:
        all_results[split] = eval_split(model, split)

    print_summary(all_results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None, help="direct path to .pt file")
    parser.add_argument("--dataset", default=None, help="path to dataset yaml")
    parser.add_argument("--splits",  default=None, nargs="+", help="splits to eval (e.g. val test)")
    parser.add_argument("--device",  default=None)
    parser.add_argument("--batch",   type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    if args.dataset is not None: DATASET = args.dataset
    if args.splits  is not None: SPLITS  = args.splits
    if args.device  is not None: DEVICE  = int(args.device) if str(args.device).isdigit() else args.device
    if args.batch   is not None: BATCH   = args.batch
    if args.workers is not None: WORKERS = args.workers
    if args.weights is not None: WEIGHTS = args.weights

    main()
