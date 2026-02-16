from transformers import BertForMaskedLM, AdamW, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import MLMDataset

def run_mlm_training(tokenizer, texts, device):
    model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    dataset = MLMDataset(tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt'))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=True))
    
    model.train()
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["labels"] = batch["input_ids"].clone()
        loss = model(**batch).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model
