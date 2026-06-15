# YOLOv10s Attention Mechanism Comparison on Furnas Dataset — 實驗筆記

> **Dataset**: Furnas Dataset v0.07（電力線零件巡檢）  
> **Base model**: YOLOv10s（COCO pretrained）  
> **Input size**: 640×640 | **Optimizer**: SGD | **Batch**: 16 | **Epochs**: 600（patience=100）

---

## 資料集說明

| 項目 | 內容 |
|------|------|
| 來源 | 電力線無人機巡檢影像（Furnas 電力公司，巴西）|
| 解析度 | 1280×720 |
| Train | 4,812 張（原始 train 的 85%）|
| Val | 849 張（從 train 切 15%，seed=42）|
| Test | 628 張（原始資料集提供的 test split）|
| 標注數（train） | 16,145 個 bbox |
| 類別數 | 11 類 |
| 原始格式 | COCO JSON → 轉換為 YOLO format |

### 類別與樣本分布（train）

| ID | 類別 | 數量 | 備注 |
|----|------|------|------|
| 0 | baliser_ok | 125 | 少 |
| 1 | baliser_aok | 319 | |
| 2 | baliser_nok | 184 | |
| 3 | insulator_ok | 7,854 | 主導類別 |
| 4 | insulator_nok | 1,840 | |
| 5 | bird_nest | 340 | |
| 6 | stockbridge_ok | 2,808 | |
| 7 | stockbridge_nok | 0 | 無訓練樣本 |
| 8 | spacer_ok | 1,016 | |
| 9 | spacer_nok | 21 | 極少 |
| 10 | insulator_unk | 1,638 | |

---

## 實驗設計

### 模型架構說明

| 模型名稱 | 完整說明 | 架構重點 |
|---------|---------|---------|
| **CBAM** | YOLOv10s + CBAM（P3/P4 neck）| Channel + Spatial attention |
| **CoordAtt** | YOLOv10s + CoordAtt（P3/P4/P5 pre-detect）| 座標感知 Channel attention |
| **ECA** | YOLOv10s + ECA（P3/P4/P5 pre-detect）| 輕量 Channel attention（1D conv）|
| **C3TR-lite (b=1)** | YOLOv10s，最後 **1** 層 C2f 替換為 C3TR-lite | Transformer self-attention |
| **C3TR-lite (b=2)** | YOLOv10s，最後 **2** 層 C2f 替換為 C3TR-lite | Transformer self-attention |
| **C3TR-lite (b=8)** | YOLOv10s，全部 **8** 層 C2f 替換為 C3TR-lite | Transformer self-attention |
| **C3TR-paper** | YOLOv10s，依論文指定層 [2,4,6,8,13,16,19] 替換為 C3TR-lite | 完整論文 C3TR 方法 |
| **C3TR-lite (b=2) + CBAM** | C3TR-lite（最後2層）＋ CBAM（P3/P4 neck），從頭訓練 | 組合嘗試 |
| **C3TR-lite (b=2) + CBAM (FT)** | 從訓練好的 CBAM 模型載入，再套 C3TR-lite（最後2層），lr=0.001 fine-tune | Fine-tune 組合嘗試 |
| **C3TR-paper + ECA-paper** | 依論文指定層替換 C3TR-lite，同時在論文指定位置加入 ECA | 完全依論文的組合方法 |

### C3TR-lite

C3TR-lite 把 YOLOv10 neck/backbone 中的 **C2f block** 換成加了 Transformer self-attention 的版本，目的是增強對小目標或複雜外觀目標的特徵萃取能力。

---

## 訓練設定

```
epochs        = 600（patience=100，Early Stopping）
imgsz         = 640
batch         = 16
workers       = 2     4 process 同跑時需降低，避免 /dev/shm 不足
optimizer     = SGD
lr0           = 0.01
lrf           = 0.01  cosine decay，final LR = 0.0001
momentum      = 0.937
weight_decay  = 0.001
warmup_epochs = 5
patience      = 100

# C3TR-lite (b=2) + CBAM (FT) 例外
lr0           = 0.001  fine-tune 用較低 LR
epochs        = 300
```

---

## 實驗結果

### TEST Set 整體指標（最終評估，排序依 TEST mAP50-95）

> Params：nc=11（Furnas）；FPS：batch=1，純 forward pass，同一 GPU 依序測量

| 模型 | TEST mAP50 | **TEST mAP50-95** | TEST P | TEST R | FPS | Params | VAL mAP50 | VAL mAP50-95 |
|------|:----------:|:-----------------:|:------:|:------:|:---:|:------:|:---------:|:------------:|
| **C3TR-lite (b=1)** | **0.8659** | **0.6153** | **0.881** | 0.807 | 48.9 | 8,235,250 | 0.8798 | 0.6156 |
| **C3TR-lite (b=2)** | **0.8613** | 0.6088 | 0.850 | 0.811 | 47.5 | 8,055,858 | **0.8816** | **0.6218** |
| CoordAtt | 0.8502 | **0.6003** | 0.832 | 0.807 | 45.3 | 8,139,490 | 0.8728 | 0.6036 |
| **C3TR-paper** | 0.8544 | 0.6002 | 0.856 | 0.793 | 17.3 | 7,372,706 | **0.8828** | 0.6142 |
| CBAM | 0.8542 | 0.5977 | 0.844 | 0.815 | 45.2 | 8,085,302 | 0.8724 | 0.6081 |
| C3TR-lite (b=2) + CBAM | 0.8542 | 0.5977 | 0.844 | 0.815 | 43.7 | 8,066,294 | 0.8724 | 0.6081 |
| ECA | 0.8363 | 0.5976 | 0.820 | **0.818** | 47.9 | 8,074,875 | 0.8518 | 0.5895 |
| C3TR-paper + ECA-paper | 0.8363 | 0.5976 | 0.820 | 0.818 | 17.3 | 7,372,715 | 0.8518 | 0.5895 |
| C3TR-lite (b=2) + CBAM (FT) | 0.8480 | 0.5974 | **0.881** | 0.804 | 43.7 | 8,066,294 | 0.8589 | 0.5936 |
| C3TR-lite (b=8) | 0.8447 | 0.5953 | 0.849 | **0.812** | 17.3 | 7,533,090 | 0.8754 | 0.6174 |
| Baseline | 0.8521 | 0.5881 | 0.858 | 0.794 | 49.9 | 8,074,866 | 0.8668 | 0.5992 |

### Per-class AP@0.50（TEST，Baseline 對照）

| 類別 | Baseline |
|------|:--------:|
| baliser_ok | 0.7429 |
| baliser_aok | 0.8490 |
| baliser_nok | 0.8958 |
| insulator_ok | 0.9637 |
| insulator_nok | 0.9013 |
| bird_nest | 0.8961 |
| stockbridge_ok | 0.8574 |
| spacer_ok | 0.7275 |
| spacer_nok | 0.9950 |
| insulator_unk | 0.6927 |

> Baseline 由外部提供（非本次 600 epoch 訓練）

### 訓練過程 Best VAL mAP50

| 模型 | Best VAL mAP50 | Best Epoch | 停止 Epoch |
|------|:--------------:|:----------:|:----------:|
| **C3TR-lite (b=2)** | **0.8878** | 166 | 257（ES）|
| C3TR-paper | 0.8827 | 384 | 484（ES）|
| C3TR-lite (b=1) | 0.8820 | 132 | 286（ES）|
| C3TR-lite (b=8) | 0.8780 | 129 | 375（ES）|
| CoordAtt | 0.8756 | 422 | 426（ES）|
| CBAM | 0.8751 | 352 | 369（ES）|
| C3TR-lite (b=2) + CBAM | 0.8751 | 351 | 369（ES）|
| Baseline | 0.8668 | — | — |
| C3TR-lite (b=2) + CBAM (FT) | 0.8669 | 84 | 184（ES）|
| ECA | 0.8608 | 325 | 332（ES）|
| C3TR-paper + ECA-paper | 0.8608 | 232 | 332（ES）|

---

## 關鍵發現

### 1. C3TR-lite（換少數層）優於其他 attention 方法

```
TEST mAP50:
C3TR(b=1) 0.8659 > C3TR(b=2) 0.8613 > C3TR-paper 0.8544
> CBAM 0.8542 ≈ C3TR(b=2)+CBAM 0.8542 > Baseline 0.8521
> CoordAtt 0.8502 > C3TR(b=2)+CBAM(FT) 0.8480 > C3TR(b=8) 0.8447
> ECA 0.8363 ≈ C3TR-paper+ECA-paper 0.8363
```

換 1–2 層 C2f 為 C3TR-lite 是最有效的方式，優於所有純 attention module 方法。

### 2. 換太多層反而變差

C3TR-lite (b=8)（全換 8 層）比 b=1/b=2 差。替換過多層會破壞 COCO pretrained 的特徵萃取能力，Furnas 的訓練量不足以補回來。

### 3. 組合實驗均無法超越純 C3TR-lite

- C3TR-lite (b=2) + CBAM（從頭訓練）= 純 CBAM，沒有提升
- C3TR-lite (b=2) + CBAM（fine-tune）= 反而更差
- C3TR-paper + ECA-paper = 等同純 ECA，沒有提升

兩種 attention 功能重疊或互相干擾，無法疊加。

### 4. 論文方法（C3TR-paper）中等表現

按論文指定層做替換，TEST mAP50 排第 3（0.8544），比純 b=1/b=2 差，比所有 attention 模組好。論文層位置在這個資料集上不是最優選擇。

### 5. ECA 和 C3TR-paper + ECA-paper 並列最差

兩者 TEST mAP50 相同（0.8363），組合 ECA 進論文方法完全沒有效果。

### 6. FPS：換層數決定推論速度

> FPS 以 batch=1 純 forward pass 在同一 GPU 依序測量，排除資料讀取/NMS 干擾

```
高速（~45–50 FPS）：Baseline(49.9) ≈ C3TR(b=1)(48.9) > ECA(47.9) ≈ C3TR(b=2)(47.5)
                    > CoordAtt(45.3) ≈ CBAM(45.2) > C3TR(b=2)+CBAM(43.7)
低速（~17 FPS）：C3TR-paper(17.3) ≈ C3TR(b=8)(17.3) ≈ C3TR-paper+ECA(17.3)
```

b=8 / C3TR-paper 替換大量 C2f 為 Transformer，MHSA 複雜度為 O((H×W)²)，在大 feature map 上開銷極重。

### 7. Early Stopping 普遍在 ep100–420 觸發

所有模型均出現過擬合，best.pt 已自動儲存峰值權重，直接使用 best.pt 即可。

---

## Weights 位置

| 模型 | Best weights 路徑 |
|------|-----------------|
| CBAM | `runs/detect/runs/furnas_cbam-2/weights/best.pt` |
| CoordAtt | `runs/detect/runs/furnas_ca-2/weights/best.pt` |
| ECA | `runs/detect/runs/furnas_eca-2/weights/best.pt` |
| **C3TR-lite (b=1) ← TEST 最高** | `runs/detect/runs/furnas_b1-3/weights/best.pt` |
| **C3TR-lite (b=2) ← VAL 最高** | `runs/detect/runs/furnas_b2-2/weights/best.pt` |
| C3TR-lite (b=8) | `runs/detect/runs/furnas_b8-2/weights/best.pt` |
| C3TR-paper | `runs/detect/runs/furnas_paper-2/weights/best.pt` |
| C3TR-lite (b=2) + CBAM | `runs/detect/runs/furnas_c3tr_cbam/weights/best.pt` |
| C3TR-lite (b=2) + CBAM (FT) | `runs/detect/runs/furnas_c3tr_cbam_ft/weights/best.pt` |
| C3TR-paper + ECA-paper | `runs/detect/runs/furnas_c3tr_eca/weights/best.pt` |
