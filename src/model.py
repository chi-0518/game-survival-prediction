import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel

class OrdinalRegressionHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes - 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class OrdinalBERTClassifier(BertPreTrainedModel):
    def __init__(self, config, num_classes):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.adapter_proj = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, config.hidden_size), nn.ReLU())
        self.ordinal_head = OrdinalRegressionHead(config.hidden_size, num_classes)
        self.init_weights()

    def forward(self, input_ids, attention_mask, adapter_features=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        if adapter_features is not None:
            pooled_output = pooled_output + self.adapter_proj(adapter_features)
        return self.ordinal_head(self.dropout(pooled_output))

def ordinal_loss(preds, labels):
    labels = labels.view(-1, 1)
    range_matrix = torch.arange(preds.shape[1], device=preds.device).unsqueeze(0)
    label_matrix = (range_matrix < labels).float()
    return nn.BCELoss()(preds, label_matrix)
