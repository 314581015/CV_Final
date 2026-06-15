# YOLOv10s Attention Mechanism Comparison — Furnas Power Line Inspection

比較多種 attention 機制（CBAM、CoordAtt、ECA、C3TR-lite）在電力線零件偵測任務上的效果，基礎模型為 YOLOv10s，資料集為 Furnas Dataset v0.07。

---

## 專案結構

```
final_project/
├── train.py                      # 訓練腳本（支援 CLI 參數）
├── inference.py                  # 評估腳本
├── yolov10s.pt                   # YOLOv10s COCO pretrained weights
├── EXPERIMENT_NOTES.md           # 詳細實驗紀錄與分析
├── processes-13-00529-v2.pdf     # 參考論文
│
├── models/
│   ├── attention.py              # CBAM / ECA / CoordAtt 模組
│   ├── c3tr_lite.py              # C3TR-lite 模組
│   ├── yolov10s_cbam.yaml        # CBAM 架構（nc=11）
│   ├── yolov10s_ca.yaml          # CoordAtt 架構（nc=11）
│   └── yolov10s_eca.yaml         # ECA 架構（nc=11）
│
├── datasets/
│   ├── Furnas/                   # 已轉換的 YOLO 格式資料集
│   │   ├── images/{train,val,test}/
│   │   └── labels/{train,val,test}/
│   ├── furnas.yaml               # 資料集設定檔
│   └── convert_furnas.py         # 格式轉換腳本（COCO JSON → YOLO）
│
├── runs/detect/runs/             # 各模型訓練結果
│   ├── furnas_baseline-2/        ├── furnas_cbam-2/
│   ├── furnas_ca-2/              ├── furnas_eca-2/
│   ├── furnas_b1-3/              ├── furnas_b2-2/
│   ├── furnas_b8-2/              ├── furnas_paper-2/
│   ├── furnas_c3tr_cbam/         └── furnas_c3tr_cbam_ft/
│
└── logs/
    ├── furnas/                   # epoch-by-epoch 訓練 log
    └── score/                    # inference 評估結果 log
```

---

## 環境

```bash
conda activate train
# Python 3.9 | PyTorch 2.8+cu128 | ultralytics 8.4.53
```

---

## 資料集準備

資料集已轉換完成，位於 `datasets/Furnas/`。若需重新轉換：

```bash
tar -xjf furnas_dataset_v0.07.tar.bz2
python datasets/convert_furnas.py \
  --src datasets/furnas_dataset_v0.07 \
  --dst datasets/Furnas
```

---

## 訓練

```bash
conda activate train
cd /workplace/pcchu/final_project

# 可用 --model 參數
#   cbam                          CBAM（P3/P4 neck）
#   ca                            CoordAtt（P3/P4/P5）
#   eca                           ECA（P3/P4/P5）
#   c3tr_lite  --blocks 1/2/8     C3TR-lite（指定替換層數）
#   c3tr_paper                    C3TR-paper（論文指定層）
#   c3tr_cbam  --blocks 2         C3TR-lite（最後2層）+ CBAM
#   c3tr_eca   --blocks 2         C3TR-paper + ECA-paper

# 範例
nohup python train.py \
  --model c3tr_lite --blocks 2 \
  --name furnas_b2 \
  --dataset datasets/furnas.yaml \
  --epochs 600 --device 0 --workers 2 \
  --logdir logs/furnas \
  > logs/furnas_b2.log 2>&1 &

# Fine-tune 從已訓練模型載入
nohup python train.py \
  --model c3tr_cbam --blocks 2 \
  --name furnas_c3tr_cbam_ft \
  --base_weights runs/detect/runs/furnas_cbam-2/weights/best.pt \
  --dataset datasets/furnas.yaml \
  --epochs 300 --lr0 0.001 \
  --device 0 --workers 2 \
  --logdir logs/furnas \
  > logs/furnas_c3tr_cbam_ft.log 2>&1 &

# 監看進度
tail -f logs/furnas/furnas_b2_epoch.log
```

---

## 評估

```bash
python inference.py \
  --weights runs/detect/runs/furnas_b2-2/weights/best.pt \
  --dataset datasets/furnas.yaml \
  --splits val test \
  --device 0 --workers 2 \
  > logs/score/b2.log 2>&1
```

> ¹ Baseline 含 optimizer state；GFLOPs 以 thop 計算（640×640 input）；FPS 為 GPU A100 inference 實測

**最佳模型（TEST 精度）**：`runs/detect/runs/furnas_b1-3/weights/best.pt`  
**最佳模型（VAL 精度）**：`runs/detect/runs/furnas_b2-2/weights/best.pt`

詳細分析見 [EXPERIMENT_NOTES.md](EXPERIMENT_NOTES.md)。

---

## 注意事項

- 訓練用 `nc=11` 的 YAML，直接從 COCO pretrained 載入時 detection head 的 nc mismatch 會被自動過濾（隨機初始化）
- `best.pt` 由 ultralytics 自動在 val mAP50 最高時儲存
- 各模型的 600ep 訓練結果存在 `-2` / `-3` 後綴目錄
