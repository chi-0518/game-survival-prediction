import torch
from transformers import BertTokenizer, BertConfig, get_cosine_schedule_with_warmup
from .dataset import load_data, prepare_data
from .model import OrdinalBERTClassifier, ordinal_loss
from .train_mlm import run_mlm_training

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, num_classes = load_data()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Step 1: MLM
    mlm_model = run_mlm_training(tokenizer, data['評論內容'].tolist(), device)
    
    # Step 2: Main Model
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = OrdinalBERTClassifier(config, num_classes + 1).to(device)
    model.bert.load_state_dict(mlm_model.bert.state_dict(), strict=False)
    
    # ... (後續訓練與 evaluate 邏輯請參照原始碼貼入)
    print("訓練完成！")

if __name__ == "__main__":
    main()
