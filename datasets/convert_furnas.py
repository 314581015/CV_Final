"""
Convert Furnas Dataset v0.07 (COCO JSON) to YOLO format.

Furnas structure:
  furnas_dataset_v0.07/
    imgs/train/   *.jpg
    imgs/test/    *.jpg
    data/coco/train.json
    data/coco/test.json   (may not exist)

Output structure (YOLO):
  <dst>/
    images/train/  images/val/  images/test/
    labels/train/  labels/val/  labels/test/

Usage:
  python datasets/convert_furnas.py --src /path/to/furnas_dataset_v0.07 --dst datasets/Furnas
"""

import argparse, json, shutil, random
from pathlib import Path

CLASSES = [
    "baliser_ok", "baliser_aok", "baliser_nok",
    "insulator_ok", "insulator_nok", "bird_nest",
    "stockbridge_ok", "stockbridge_nok",
    "spacer_ok", "spacer_nok", "insulator_unk",
]

VAL_RATIO   = 0.15
RANDOM_SEED = 42


def coco_to_yolo(bbox, img_w, img_h):
    """COCO [xmin, ymin, w, h] → YOLO [cx, cy, w, h] normalised."""
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(nw, 1.0)
    nh = min(nh, 1.0)
    return cx, cy, nw, nh


def convert_split(coco_json: Path, img_src_dir: Path, dst_img_dir: Path,
                  dst_lbl_dir: Path, cat_to_cls: dict):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json) as f:
        data = json.load(f)

    # image_id → {file_name, width, height}
    img_info = {img["id"]: img for img in data["images"]}

    # image_id → list of YOLO lines
    labels: dict[str, list[str]] = {img["id"]: [] for img in data["images"]}

    for ann in data["annotations"]:
        if ann.get("ignore", 0) or ann.get("iscrowd", 0):
            continue
        cat_id = ann["category_id"]
        if cat_id not in cat_to_cls:
            continue
        cls_id = cat_to_cls[cat_id]
        info = img_info[ann["image_id"]]
        result = coco_to_yolo(ann["bbox"], info["width"], info["height"])
        if result is None:
            continue
        cx, cy, nw, nh = result
        labels[ann["image_id"]].append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    copied = skipped = 0
    for img in data["images"]:
        img_id   = img["id"]
        filename = img["file_name"]
        src_path = img_src_dir / filename

        if not src_path.exists():
            skipped += 1
            continue

        shutil.copy(src_path, dst_img_dir / filename)

        stem = Path(filename).stem
        lbl_path = dst_lbl_dir / f"{stem}.txt"
        lbl_path.write_text("\n".join(labels.get(img_id, [])))
        copied += 1

    print(f"  copied {copied} images  (skipped {skipped})")


def convert(src: Path, dst: Path):
    coco_dir  = src / "data" / "coco"
    train_json = coco_dir / "train.json"
    test_json  = coco_dir / "test.json"

    # Build category → class-index mapping from train.json
    with open(train_json) as f:
        meta = json.load(f)
    cat_to_cls = {}
    for cat in meta["categories"]:
        name = cat["name"]
        if name in CLASSES:
            cat_to_cls[cat["id"]] = CLASSES.index(name)
        else:
            print(f"  [warn] unknown category '{name}', skipping")

    # ── Train → split into train / val ────────────────────────────────────────
    print("Building train / val split ...")
    with open(train_json) as f:
        data = json.load(f)

    all_imgs = data["images"]
    random.seed(RANDOM_SEED)
    random.shuffle(all_imgs)
    n_val  = int(len(all_imgs) * VAL_RATIO)
    val_ids  = {img["id"] for img in all_imgs[:n_val]}
    train_ids = {img["id"] for img in all_imgs[n_val:]}

    def subset_json(split_ids):
        imgs = [i for i in data["images"]      if i["id"] in split_ids]
        anns = [a for a in data["annotations"] if a["image_id"] in split_ids]
        return {**data, "images": imgs, "annotations": anns}

    import tempfile, os
    tmp = Path(tempfile.mkdtemp())
    for split_name, split_ids in [("train", train_ids), ("val", val_ids)]:
        tmp_json = tmp / f"{split_name}.json"
        with open(tmp_json, "w") as f:
            json.dump(subset_json(split_ids), f)
        print(f"[{split_name}] {len(split_ids)} images")
        convert_split(
            tmp_json,
            src / "imgs" / "train",
            dst / "images" / split_name,
            dst / "labels" / split_name,
            cat_to_cls,
        )

    # ── Test split ─────────────────────────────────────────────────────────────
    if test_json.exists():
        print(f"[test]  converting ...")
        convert_split(
            test_json,
            src / "imgs" / "test",
            dst / "images" / "test",
            dst / "labels" / "test",
            cat_to_cls,
        )
    else:
        print("[test]  test.json not found, skipping")

    print(f"\nDone. Output at: {dst}")
    print(f"Set `path` in datasets/furnas.yaml to: {dst.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to furnas_dataset_v0.07/")
    parser.add_argument("--dst", default="datasets/Furnas", help="Output YOLO dataset dir")
    args = parser.parse_args()
    convert(Path(args.src), Path(args.dst))
