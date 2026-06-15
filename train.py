"""
Training script for YOLOv10s variants on Furnas Dataset.

Models:
  "baseline"   → YOLOv10s original
  "cbam"       → YOLOv10s + CBAM at P3/P4 neck
  "ca"         → YOLOv10s + CoordAtt at P3/P4/P5 pre-detect
  "eca"        → YOLOv10s + ECA at P3/P4/P5 pre-detect
  "c3tr_lite"  → YOLOv10s with last N C2f blocks replaced by C3TR-lite
  "c3tr_paper" → YOLOv10s with paper-style C3TR-lite replacements

IMPORTANT: register custom modules into ultralytics BEFORE importing YOLO.
"""

# ── Step 1: Register custom modules into ultralytics ──────────────────────────
import ultralytics.nn.tasks as _tasks
from models.attention import CBAM, ECA, CoordAtt
_tasks.CBAM = CBAM
_tasks.ECA = ECA
_tasks.CoordAtt = CoordAtt

# ── Step 2: Normal imports ─────────────────────────────────────────────────────
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from pathlib import Path
from models.c3tr_lite import replace_indexed_c2f_with_c3tr_lite, replace_late_c2f_with_c3tr_lite

# Models that use programmatic C3TR replacement (not in any YAML).
# For these, ultralytics' get_model() would rebuild from YAML and lose the C3TRLite
# blocks, so we inject a custom trainer that returns our pre-built model directly.
_C3TR_MODELS = {"c3tr_lite", "c3tr_paper", "c3tr_cbam", "c3tr_eca"}


def _make_c3tr_trainer(prebuilt_module):
    """Return a DetectionTrainer subclass that skips YAML reconstruction.

    Ultralytics normally calls get_model(cfg=yaml, weights=...) which creates a
    fresh DetectionModel from the YAML (losing any programmatic modifications like
    C3TRLite). Overriding get_model to return prebuilt_module keeps C3TRLite intact.
    """
    class _C3TRTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return prebuilt_module
    return _C3TRTrainer

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these before each run
# ══════════════════════════════════════════════════════════════════════════════

MODEL = "c3tr_paper"          # "baseline" | "cbam" | "cbam_p3" | "c3tr_lite" | "c3tr_paper"
DATASET = "datasets/visdrone.yaml"
EPOCHS = 100
IMGSZ = 640
BATCH = 16               # reduce to 4 if OOM
DEVICE = 2              # GPU index (0 for single GPU)
WORKERS = 4
PROJECT = "runs"        # all results saved under runs/<MODEL>/
PRETRAINED = True       # load COCO weights before fine-tuning
C3TR_BLOCKS = 1          # start with 1; try 2 only after checking speed/memory
C3TR_HEADS = 4
C3TR_PAPER_LAYERS = [2, 4, 6, 8, 13, 16, 19]

# ══════════════════════════════════════════════════════════════════════════════

MODEL_MAP = {
    "baseline":   None,
    "cbam":       "models/yolov10s_cbam.yaml",
    "cbam_p3":    "models/yolov10s_cbam_p3only.yaml",
    "ca":         "models/yolov10s_ca.yaml",
    "eca":        "models/yolov10s_eca.yaml",
    "c3tr_lite":  None,
    "c3tr_paper": None,
    "c3tr_cbam":  "models/yolov10s_cbam.yaml",   # CBAM yaml + C3TR-lite on top
    "c3tr_eca":   "models/yolov10s_eca.yaml",    # ECA yaml + C3TR-lite on top
}

# Pretrained layer index → Custom layer index mapping.
#
# cbam:    2 CBAM at layers 14 and 18
#   orig 0-13  → 0-13  (no shift)
#   orig 14-16 → 15-17 (+1, skip 14=CBAM)
#   orig 17-23 → 19-25 (+2, skip 18=CBAM)
#
# cbam_p3: 1 CBAM at layer 17
#   orig 0-16  → 0-16  (no shift)
#   orig 17-23 → 18-24 (+1, skip 17=CBAM)
#
# ca / eca: 3 attention layers at 17, 21, 25
#   orig 0-16  → 0-16  (no shift)
#   orig 17-19 → 18-20 (+1, skip 17=attn)
#   orig 20-22 → 22-24 (+2, skip 21=attn)
#   orig 23    → 26    (+3, skip 25=attn)
_ORIG_TO_CUSTOM = {
    "cbam": {**{i: i for i in range(14)},
             **{i: i + 1 for i in range(14, 17)},
             **{i: i + 2 for i in range(17, 24)}},
    "cbam_p3": {**{i: i for i in range(17)},
                **{i: i + 1 for i in range(17, 24)}},
    "ca":  {**{i: i for i in range(17)},
            **{i: i + 1 for i in range(17, 20)},
            **{i: i + 2 for i in range(20, 23)},
            23: 26},
    "eca": {**{i: i for i in range(17)},
            **{i: i + 1 for i in range(17, 20)},
            **{i: i + 2 for i in range(20, 23)},
            23: 26},
}


def _remap_state_dict(pretrained_sd: dict, model_name: str) -> dict:
    """Rename pretrained keys so layer indices match the custom YAML layout."""
    mapping = _ORIG_TO_CUSTOM[model_name]
    new_sd = {}
    for key, val in pretrained_sd.items():
        parts = key.split(".")
        if parts[0] == "model" and len(parts) > 1 and parts[1].isdigit():
            orig_idx = int(parts[1])
            if orig_idx in mapping:
                new_key = "model." + str(mapping[orig_idx]) + "." + ".".join(parts[2:])
                new_sd[new_key] = val
            # keys with no mapping are dropped (shouldn't happen with correct table)
        else:
            new_sd[key] = val
    return new_sd


def load_model(model_name: str, pretrained: bool) -> YOLO:
    yaml_path = MODEL_MAP[model_name]

    if model_name == "baseline":
        model = YOLO("yolov10s.pt")
        print("Loaded: YOLOv10s baseline (COCO pretrained)")
        return model

    if model_name in ("c3tr_cbam", "c3tr_eca"):
        # Load attention yaml with pretrained backbone, then apply C3TR-lite on top
        yaml_path = MODEL_MAP[model_name]
        # remap key: cbam for c3tr_cbam, eca for c3tr_eca
        remap_key = "cbam" if model_name == "c3tr_cbam" else "eca"
        attn_label = "CBAM" if model_name == "c3tr_cbam" else "ECA"
        model = YOLO(yaml_path)
        if pretrained:
            pretrained_model = YOLO("yolov10s.pt")
            remapped_sd = _remap_state_dict(pretrained_model.model.state_dict(), remap_key)
            del pretrained_model
            target_sd = model.model.state_dict()
            filtered_sd = {k: v for k, v in remapped_sd.items()
                           if k in target_sd and v.shape == target_sd[k].shape}
            model.model.load_state_dict(filtered_sd, strict=False)
            print(f"Loaded pretrained weights into {yaml_path} (filtered nc mismatch)")
        replaced = replace_late_c2f_with_c3tr_lite(
            model.model, num_blocks=C3TR_BLOCKS, heads=C3TR_HEADS)
        print(f"Loaded: YOLOv10s + {attn_label} + C3TR-lite")
        print("Applied C3TR-lite replacements:")
        for line in replaced:
            print(f"  - {line}")
        return model

    if model_name == "c3tr_lite":
        model = YOLO("yolov10s.pt" if pretrained else "yolov10s.yaml")
        replaced = replace_late_c2f_with_c3tr_lite(
            model.model,
            num_blocks=C3TR_BLOCKS,
            heads=C3TR_HEADS,
        )
        print("Loaded: YOLOv10s + C3TR-lite")
        print("Applied C3TR-lite replacements:")
        for line in replaced:
            print(f"  - {line}")
        return model

    if model_name == "c3tr_paper":
        model = YOLO("yolov10s.pt" if pretrained else "yolov10s.yaml")
        replaced = replace_indexed_c2f_with_c3tr_lite(
            model.model,
            target_indices=C3TR_PAPER_LAYERS,
            heads=C3TR_HEADS,
        )
        print("Loaded: YOLOv10s + paper-style C3TR-lite")
        print("Applied C3TR-lite replacements:")
        for line in replaced:
            print(f"  - {line}")
        return model

    model = YOLO(yaml_path)

    if pretrained:
        pretrained_model = YOLO("yolov10s.pt")
        if model_name in _ORIG_TO_CUSTOM:
            remapped_sd = _remap_state_dict(pretrained_model.model.state_dict(), model_name)
        else:
            remapped_sd = pretrained_model.model.state_dict()
        del pretrained_model

        # Drop detection-head classification keys that have nc-dependent shapes.
        # yolov10s.pt uses nc=80 (COCO); our YAML uses nc=11 (Furnas).
        # strict=False skips missing/unexpected keys but still raises on size mismatch,
        # so we must remove the mismatched keys before calling load_state_dict.
        target_sd = model.model.state_dict()
        filtered_sd = {
            k: v for k, v in remapped_sd.items()
            if k in target_sd and v.shape == target_sd[k].shape
        }
        skipped = [k for k in remapped_sd if k not in filtered_sd]

        missing, unexpected = model.model.load_state_dict(filtered_sd, strict=False)
        attn_keys = [k for k in missing if any(s in k for s in (".ca.", ".sa.", ".conv.", "coord", "eca"))]
        other_missing = [k for k in missing if k not in attn_keys and k not in skipped]
        print(f"Loaded pretrained weights into {yaml_path}")
        print(f"  Attention layers (randomly init): {len(attn_keys)}")
        print(f"  Detect head skipped (nc mismatch): {len(skipped)}")
        print(f"  Other missing:                     {len(other_missing)}")
        if other_missing:
            for k in other_missing[:5]:
                print(f"    {k}")

    return model


def main():
    print(f"\n{'='*60}")
    print(f" Training: {MODEL}")
    print(f" Dataset:  {DATASET}")
    print(f" Epochs:   {EPOCHS}  |  Batch: {BATCH}  |  Imgsz: {IMGSZ}")
    print(f"{'='*60}\n")

    model = load_model(MODEL, PRETRAINED)

    import sys, datetime
    _epoch_log = open(f"{MODEL}_epoch.log", "a")

    def _log_epoch(trainer):
        ep = trainer.epoch + 1
        loss = float(trainer.loss)
        lr = trainer.optimizer.param_groups[0]["lr"]
        metrics = trainer.metrics or {}
        map50 = metrics.get("metrics/mAP50(B)", float("nan"))
        map50_95 = metrics.get("metrics/mAP50-95(B)", float("nan"))
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = (f"[{ts}] epoch {ep:>4}/{EPOCHS}  loss={loss:.4f}  lr={lr:.6f}"
                f"  mAP50={map50:.4f}  mAP50-95={map50_95:.4f}\n")
        _epoch_log.write(line)
        _epoch_log.flush()

    model.add_callback("on_train_epoch_end", _log_epoch)

    trainer_cls = _make_c3tr_trainer(model.model) if MODEL in _C3TR_MODELS else None

    model.train(
        data=DATASET,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT,
        name=MODEL,
        # ── Optimizer ──────────────────────────────────────
        optimizer="SGD",
        lr0=0.01,           # initial learning rate
        lrf=0.01,           # final lr = lr0 * lrf (cosine decay)
        momentum=0.937,
        weight_decay=0.001,
        patience=100,
        # ── Warmup ─────────────────────────────────────────
        warmup_epochs=5,
        warmup_momentum=0.8,
        # ── Augmentation ───────────────────────────────────
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.1,
        close_mosaic=50,
        # ── Logging ────────────────────────────────────────
        save=True,
        save_period=-1,     # only save best and last
        plots=True,
        verbose=True,
        **({"trainer": trainer_cls} if trainer_cls else {}),
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=None)
    parser.add_argument("--name",    default=None, help="run name / save dir (default: model)")
    parser.add_argument("--blocks",  type=int,   default=None)
    parser.add_argument("--device",  default=None)
    parser.add_argument("--epochs",  type=int,   default=None)
    parser.add_argument("--batch",   type=int,   default=None)
    parser.add_argument("--workers", type=int,   default=None)
    parser.add_argument("--dataset", default=None, help="path to dataset yaml (default: datasets/visdrone.yaml)")
    parser.add_argument("--logdir",       default=None, help="directory for epoch log (default: current dir)")
    parser.add_argument("--lr0",          type=float, default=None, help="initial learning rate (default: 0.01)")
    parser.add_argument("--base_weights", default=None, help="load this .pt as starting model, then apply C3TR-lite on top")
    parser.add_argument("--resume",  action="store_true",
                        help="resume from last.pt of --name run")
    args = parser.parse_args()

    if args.model   is not None: MODEL        = args.model
    if args.blocks  is not None: C3TR_BLOCKS  = args.blocks
    if args.device  is not None: DEVICE       = int(args.device) if args.device.isdigit() else args.device
    if args.epochs  is not None: EPOCHS       = args.epochs
    if args.batch   is not None: BATCH        = args.batch
    if args.workers is not None: WORKERS      = args.workers
    if args.dataset is not None: DATASET      = args.dataset
    LR0 = args.lr0 if args.lr0 is not None else 0.01
    RUN_NAME = args.name if args.name else MODEL
    LOG_PATH = Path(args.logdir) / f"{RUN_NAME}_epoch.log" if args.logdir else Path(f"{RUN_NAME}_epoch.log")

    def main():
        import datetime
        from pathlib import Path

        resume_weights = None
        if args.resume:
            resume_weights = Path(PROJECT) / "detect" / "runs" / RUN_NAME / "weights" / "last.pt"
            if not resume_weights.exists():
                raise FileNotFoundError(f"Cannot resume: {resume_weights} not found")
            print(f"\n{'='*60}")
            print(f" RESUMING: {RUN_NAME}  →  total epochs: {EPOCHS}")
            print(f" Weights:  {resume_weights}")
            print(f"{'='*60}\n")
            model = YOLO(str(resume_weights))
        elif args.base_weights:
            # Load from a trained .pt, then apply C3TR-lite on top
            bw = Path(args.base_weights)
            if not bw.exists():
                raise FileNotFoundError(f"base_weights not found: {bw}")
            print(f"\n{'='*60}")
            print(f" Fine-tune: {MODEL}  →  run name: {RUN_NAME}")
            print(f" Base weights: {bw}")
            print(f" Dataset:  {DATASET}")
            print(f" Epochs:   {EPOCHS}  |  LR0: {LR0}  |  Batch: {BATCH}")
            print(f" C3TR_BLOCKS: {C3TR_BLOCKS}")
            print(f"{'='*60}\n")
            model = YOLO(str(bw))
            replaced = replace_late_c2f_with_c3tr_lite(
                model.model, num_blocks=C3TR_BLOCKS, heads=C3TR_HEADS)
            print("Applied C3TR-lite on top of base weights:")
            for line in replaced:
                print(f"  - {line}")
        else:
            print(f"\n{'='*60}")
            print(f" Training: {MODEL}  →  run name: {RUN_NAME}")
            print(f" Dataset:  {DATASET}")
            print(f" Epochs:   {EPOCHS}  |  Batch: {BATCH}  |  Imgsz: {IMGSZ}")
            print(f" C3TR_BLOCKS: {C3TR_BLOCKS}")
            print(f"{'='*60}\n")
            model = load_model(MODEL, PRETRAINED)

        _epoch_log = open(LOG_PATH, "a")

        def _log_epoch(trainer):
            ep = trainer.epoch + 1
            loss = float(trainer.loss)
            lr = trainer.optimizer.param_groups[0]["lr"]
            metrics = trainer.metrics or {}
            map50    = metrics.get("metrics/mAP50(B)",    float("nan"))
            map50_95 = metrics.get("metrics/mAP50-95(B)", float("nan"))
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            line = (f"[{ts}] epoch {ep:>4}/{EPOCHS}  loss={loss:.4f}  lr={lr:.6f}"
                    f"  mAP50={map50:.4f}  mAP50-95={map50_95:.4f}\n")
            _epoch_log.write(line)
            _epoch_log.flush()

        model.add_callback("on_train_epoch_end", _log_epoch)

        # C3TR models are modified programmatically (not in any YAML).
        # Without a custom trainer, ultralytics rebuilds the model from YAML
        # inside get_model() and discards the C3TRLite blocks.
        trainer_cls = None
        if not args.resume and MODEL in _C3TR_MODELS:
            trainer_cls = _make_c3tr_trainer(model.model)
        elif args.base_weights:
            # base_weights path also applies C3TR-lite programmatically
            trainer_cls = _make_c3tr_trainer(model.model)

        model.train(
            data=DATASET,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            project=PROJECT,
            name=RUN_NAME,
            resume=args.resume,
            optimizer="SGD",
            lr0=LR0,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.001,
            patience=100,
            warmup_epochs=5,
            warmup_momentum=0.8,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.1,
            close_mosaic=50,
            save=True,
            save_period=-1,
            plots=True,
            verbose=True,
            **({"trainer": trainer_cls} if trainer_cls else {}),
        )

    main()
