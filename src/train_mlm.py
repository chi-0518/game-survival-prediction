import torch
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, AdamW, DataCollatorForLanguageModeling
from tqdm import tqdm
from .dataset import MLMDataset

def train_mlm(tokenizer, texts, device):
    mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
    mlm_optimizer = AdamW(mlm_model.parameters(), lr=5e-5)
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
    dataset = MLMDataset(encodings)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collator)

    mlm_model.train()
    for epoch in range(1):
        total_loss = 0
        for batch in tqdm(dataloader):
            batch["labels"] = batch["input_ids"].clone()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = mlm_model(**batch)
            loss = outputs.loss
            mlm_optimizer.zero_grad()
            loss.backward()
            mlm_optimizer.step()
            total_loss += loss.item()
        print(f"MLM Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
    return mlm_model
