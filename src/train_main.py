import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer, 
    BertConfig, 
    AdamW, 
    get_cosine_schedule_with_warmup
)
from torch.cuda.amp import GradScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 這裡修正了匯入路徑，確保直接執行時能抓到同資料夾的其他檔案
from dataset import load_data, prepare_data
from model import OrdinalBERTClassifier, ordinal_loss
from train_mlm import train_mlm

def evaluate(model, dataloader, device):
    """
    評估模型表現，包含 Accuracy, MAE 以及 ±N 年準確率
    """
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels, adapter_feats = [x.to(device) for x in batch]
            logits = model(input_ids=input_ids, attention_mask=attention_mask, adapter_features=adapter_feats)
            # 序數回歸預測：計算大於 0.5 的數量即為預測年數
            pred = torch.sum(logits > 0.5, dim=1).cpu().numpy()
            predictions.extend(pred)
            true_labels.extend(labels.cpu().numpy())

    y_pred = np.array(predictions)
    y_true = np.array(true_labels)

    from sklearn.metrics import accuracy_score, mean_absolute_error
    acc = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    within_1 = np.mean(np.abs(y_true - y_pred) <= 1)
    within_2 = np.mean(np.abs(y_true - y_pred) <= 2)

    print(f"\n--- 評估結果 ---")
    print(f"Accuracy (準確率): {acc:.4f}")
    print(f"MAE (平均年數誤差): {mae:.4f}")
    print(f"±1年內準確率: {within_1:.4f}")
    print(f"±2年內準確率: {within_2:.4f}")

    # Bootstrap 信心區間
    boot_samples = 1000
    boot_mae = []
    for _ in range(boot_samples):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        boot_mae.append(np.mean(np.abs(y_true[idx] - y_pred[idx])))
    upper_bound = np.percentile(np.array(boot_mae), 97.5)
    print(f"95% 信心水準誤差範圍: ±{upper_bound:.2f} 年")

def train(model, dataloader, device, test_loader, num_epochs=1, scheduler=None):
    """
    主訓練迴圈
    """
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids, attention_mask, labels, adapter_feats = [x.to(device) for x in batch]
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, adapter_features=adapter_feats)
            loss = ordinal_loss(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            if scheduler:
                scheduler.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")
        evaluate(model, test_loader, device)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 1. 載入資料 (透過 dataset.py)
    merged_data, num_classes = load_data()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 2. 第一階段：MLM 預訓練 (透過 train_mlm.py)
    print("開始第一階段：MLM 預訓練...")
    mlm_model = train_mlm(tokenizer, merged_data['評論內容'].tolist(), device)

    # 3. 初始化主模型並載入 MLM 權重
    print("開始第二階段：主模型訓練準備...")
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = OrdinalBERTClassifier(config, num_classes + 1).to(device)
    model.bert.load_state_dict(mlm_model.bert.state_dict(), strict=False)

    # 4. 凍結 BERT 前三層
    for name, param in model.bert.named_parameters():
        if any(layer in name for layer in [f"encoder.layer.{i}" for i in range(3)]):
            param.requires_grad = False

    # 5. 準備主訓練資料
    texts_with_prefix = [
        ("positive: " if r == "Recommended" else "negative: ") + t
        for r, t in zip(merged_data['推薦狀態'], merged_data['評論內容'])
    ]
    adapter_feats = merged_data[['owners_estimate', 'news_count', 'average_forever', 'median_forever']].values
    
    dataset_obj = prepare_data(tokenizer, texts_with_prefix, merged_data['year'].tolist(), adapter_feats)
    train_set, test_set = train_test_split(dataset_obj, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)

    # 6. 設定 Scheduler 與訓練
    total_steps = len(train_loader) * 1  # 範例跑 1 epoch
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.05 * total_steps), 
        num_training_steps=total_steps
    )

    train(model, train_loader, device, test_loader, num_epochs=1, scheduler=scheduler)

    # 7. 儲存模型
    model.save_pretrained("./ordinal_bert_model")
    tokenizer.save_pretrained("./ordinal_bert_model")
    print("所有流程執行完畢，模型已儲存。")

if __name__ == "__main__":
    main()
