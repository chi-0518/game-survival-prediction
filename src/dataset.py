import torch
from torch.utils.data import Dataset, TensorDataset
from pymongo import MongoClient
import pandas as pd
import numpy as np

class MLMDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx]
        }

def load_data():
    client = MongoClient("localhost:27017")
    db = client.bigdata
    game_data = pd.DataFrame(list(db.after_2023_game_id.find({}, {"遊戲ID": 1, "year": 1, "_id": 0})))
    sentiment_data = pd.DataFrame(
        list(db.disable_game_allcomment_notrepeat_04302025.find({}, {"遊戲ID": 1, "評論內容": 1, "推薦狀態": 1, "_id": 0})))
    adapter_data = pd.DataFrame(list(db.disable_game_year_news_5000_0525.find({}, {
        "遊戲ID": 1, "owners_estimate": 1, "news_count": 1, 
        "average_forever": 1, "median_forever": 1, "_id": 0
    })))

    merged_data = sentiment_data.merge(game_data, on="遊戲ID", how="inner")
    merged_data = merged_data.merge(adapter_data, on="遊戲ID", how="left")
    merged_data = merged_data[merged_data['評論內容'].notna()]
    merged_data['評論內容'] = merged_data['評論內容'].astype(str)
    merged_data['year'] = merged_data['year'].clip(0, 18)

    for col in ['owners_estimate', 'news_count', 'average_forever', 'median_forever']:
        merged_data[col] = merged_data[col].fillna(0)
        merged_data[col] = np.log1p(merged_data[col])

    return merged_data, 18

def prepare_data(tokenizer, texts, labels, adapter_feats):
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
    return TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(labels),
        torch.tensor(adapter_feats, dtype=torch.float32)
    )
