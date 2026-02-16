import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, AdamW, get_cosine_schedule_with_warmup
from torch.cuda.amp import GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from tqdm import tqdm

# 從你建立的其他模組匯入
from .dataset import load_data, prepare_data
from .model import OrdinalBERTClassifier, ordinal_loss
from .train_mlm import train_mlm

def evaluate(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels, adapter_feats = [x.to(device) for x in batch]
            logits = model(input_ids=input_ids, attention_mask=attention_mask, adapter_features=adapter_feats)
            pred = torch.sum(logits > 0.5, dim=1).cpu().numpy()
            predictions.extend(pred)
            true_labels.extend(labels.cpu().numpy())

    y_pred = np.array(predictions)
    y_true = np.array(true_labels)

    acc = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    within_1 = np.mean(np.abs(y_true - y_pred) <= 1)
    within_2 = np.mean(np.abs(y_true - y_pred) <= 2)

    print(f"Accuracy (年數預測準確率): {acc:.4f}")
    print(f"MAE (平均年數誤差): {mae:.4f}")
    print(f"±1年準確率: {within_1:.4f}")
    print(f"±2年準確率: {within_2:.4f}")

    # Bootstrap 評估
    boot_samples = 1000
    sample_size = len(y_true)
    boot_mae = []
    for _ in range(boot_samples):
        idx = np.random.choice(sample_size, sample_size, replace=True)
        sample_mae = np.mean(np.abs(y_true[idx] - y_pred[idx]))
        boot_mae.append(sample_mae)
    upper_bound = np.percentile(np.array(boot_mae), 97.5)
    print(f"95% 信心水準下的預測年數誤差範圍為：±{upper_bound:.2f} 年")

def train(model, dataloader, device, test_loader, num_epochs=1, scheduler=None):
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader):
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
        print(f"Epoch {epoch + 1}, Ordinal Loss: {total_loss / len(dataloader):.4f}")
        evaluate(model, test_loader, device)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    merged_data, num_classes = load_data()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 第一階段：MLM 預訓練
    mlm_model = train_mlm(tokenizer, merged_data['評論內容'].tolist(), device)

    # 第二階段：主模型初始化與權重載入
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = OrdinalBERTClassifier(config, num_classes + 1).to(device)
    model.bert.load_state_dict(mlm_model.bert.state_dict(), strict=False)

    # 凍結前三層 (如原始碼設定)
    for name, param in model.bert.named_parameters():
        if any(layer in name for layer in [f"encoder.layer.{i}" for i in range(3)]):
            param.requires_grad = False

    # 資料預處理
    texts_with_prefix = [
        ("positive: " if recommend == "Recommended" else "negative: ") + text
        for recommend, text in zip(merged_data['推薦狀態'], merged_data['評論內容'])
    ]
    adapter_feats = merged_data[['owners_estimate', 'news_count', 'average_forever', 'median_forever']].values
    dataset = prepare_data(tokenizer, texts_with_prefix, merged_data['year'].tolist(), adapter_feats)

    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)

    # 設定 Scheduler
    total_steps = len(train_loader) * 1  # 假設跑 1 Epoch
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)

    # 開始訓練
    train(model, train_loader, device, test_loader, num_epochs=1, scheduler=scheduler)
    
    # 儲存結果
    model.save_pretrained("./ordinal_bert_model")
    tokenizer.save_pretrained("./ordinal_bert_model")
    print("模型與 tokenizer 已儲存完畢，訓練完成！")

if __name__ == "__main__":
    main()
