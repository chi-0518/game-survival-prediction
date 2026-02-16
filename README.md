# Steam 遊戲生存年數預測系統 (Steam Survival Prediction)

這是一項基於深度學習的學術研究專案，旨在透過分析 Steam 平台的**遊戲評論（文本）**與**營運數據（數值）**，預測遊戲在平台上的生存年數。

## 研究摘要
本專案結合了自然語言處理 (NLP) 與多模態融合技術：
1. **領域適應 (Domain Adaptation)**：針對 Steam 評論進行 MLM (Masked Language Modeling) 預訓練。
2. **序數回歸 (Ordinal Regression)**：解決生存年數具有「順序性」的類別預測問題。
3. **Adapter / Prefix融合**：將遊戲的銷售預估、新聞量等數值特徵以及是否推薦(二分類特徵)整合進 BERT 模型中。

## 技術棧
- **Language**: Python 3.8+
- **Framework**: PyTorch, HuggingFace Transformers
- **Database**: MongoDB (原始資料存儲)
- **Model**: BERT-base-uncased

## 資料夾說明
- `src/model.py`: 定義 `OrdinalBERTClassifier` 與序數回歸層。
- `src/dataset.py`: 處理資料清洗、Log 轉換及 DataLoader 封裝。
- `src/train_mlm.py`: 用於執行第一階段的 MLM 預訓練。
- `src/train_main.py`: 主訓練程式，包含 Adapter 融合與模型評估。

## 快速開始

### 1. 安裝環境
請確保您的 Python 版本為 3.8+，並執行以下指令安裝依賴套件：
```bash
pip install -r requirements.txt
```

### 2. 資料準備
本專案預設連接本地 MongoDB 。

### 3. 執行訓練流程
本專案分為兩個階段進行：

第一階段：MLM (Masked Language Modeling) 領域適應訓練
這將針對 Steam 評論文字進行預訓練，優化 BERT 的語言理解能力

```bash
python src/train_mlm.py
```

第二階段：主模型訓練（Ordinal Regression + Adapter）
結合文本特徵與營運數據，進行生存年數的序數回歸預測。

```bash
python src/train_main.py
```

### 實驗結果評估
模型評估將產出以下指標，用於數據分析：

Accuracy: 年數預測精準度。

MAE (Mean Absolute Error): 平均絕對誤差（預測年數與實際年數的落差）。
