# YOLOv10s Attention Mechanism Comparison on Furnas Dataset

This repository contains the **code**, **model definitions**, and **experiment setup** for comparing multiple attention mechanisms and C3TR-based variants on the PTL-AI Furnas power-line inspection dataset.

The GitHub version of this project is intentionally **lightweight**. Large assets such as the full dataset, trained weights, logs, and complete training outputs are provided through external cloud links instead of being stored directly in the repository.

---

## Repository Contents

```text
.
├── train.py
├── inference.py
├── benchmark_fps.py
├── EXPERIMENT_NOTES.md
├── models/
│   ├── attention.py
│   ├── c3tr_lite.py
│   ├── yolov10s_cbam.yaml
│   ├── yolov10s_ca.yaml
│   ├── yolov10s_eca.yaml
│   └── yolov10s_cbam_p3only.yaml
├── datasets/
│   ├── convert_furnas.py
│   └── furnas.yaml
```

---

## Large Files

The following items are **not included in GitHub** because of repository size limits:

- Full YOLO-format Furnas dataset
- Trained model weights (`.pt`)
- Full training logs
- Full `runs/` outputs (curves, confusion matrices, sample predictions)
- Optional pretrained local checkpoints

Large files, including the dataset, trained weights, logs, and full experiment outputs, are available at: `https://drive.google.com/file/d/1wH6HSmLHrf-Tb3aqQM56CKXiVUvp4xNl/view?usp=share_link`

---

## Environment

- Python 3.9
- PyTorch 2.8
- ultralytics 8.4.53

Example:

```bash
conda activate train
```

---

## Dataset Preparation

This repository only keeps:

- `datasets/convert_furnas.py`
- `datasets/furnas.yaml`

After downloading the dataset from the cloud link above, convert it with:

```bash
python datasets/convert_furnas.py \
  --src /path/to/furnas_dataset_v0.07 \
  --dst /path/to/Furnas
```

Then update the `path:` field in `datasets/furnas.yaml` to your local dataset directory.

---

## Training

Supported model variants:

- `cbam`
- `ca`
- `eca`
- `c3tr_lite --blocks 1/2/8`
- `c3tr_paper`
- `c3tr_cbam --blocks 2`
- `c3tr_eca --blocks 2`

Example:

```bash
python train.py \
  --model c3tr_lite --blocks 2 \
  --name furnas_b2 \
  --dataset datasets/furnas.yaml \
  --epochs 600 --device 0 --workers 2 \
  --logdir logs/furnas
```

---

## Evaluation

Example:

```bash
python inference.py \
  --weights /path/to/best.pt \
  --dataset datasets/furnas.yaml \
  --splits val test \
  --device 0 --workers 2
```

---

## Results

Detailed experimental results are available in the final report.

## Source Code Availability

The source code, training scripts, and experiment settings used in this project are publicly available in this repository. Large files are distributed separately through the cloud links listed above.
