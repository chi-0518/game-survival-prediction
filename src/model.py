import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel

class OrdinalRegressionHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(input_dim, num_classes - 1)

    def forward(self, x):
        logits = self.linear(x)
        prob = torch.sigmoid(logits)
        return prob

class OrdinalBERTClassifier(BertPreTrainedModel):
    def __init__(self, config, num_classes):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.adapter_proj = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, config.hidden_size),
            nn.ReLU()
        )
        self.ordinal_head = OrdinalRegressionHead(config.hidden_size, num_classes)
        self.init_weights()

    def forward(self, input_ids, attention_mask, adapter_features=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        if adapter_features is not None:
            adapter_proj = self.adapter_proj(adapter_features)
            pooled_output = pooled_output + adapter_proj
        pooled_output = self.dropout(pooled_output)
        return self.ordinal_head(pooled_output)

def ordinal_loss(preds, labels):
    device = preds.device
    labels = labels.view(-1, 1)
    range_matrix = torch.arange(preds.shape[1], device=device).unsqueeze(0)
    label_matrix = (range_matrix < labels).float()
    bce = nn.BCELoss()
    return bce(preds, label_matrix)
