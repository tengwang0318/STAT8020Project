import torch.nn as nn
from config import *
from transformers import AutoModel, AutoConfig, AutoTokenizer


class FeedbackModel(nn.Module):
    def __init__(self, config):
        super(FeedbackModel, self).__init__()
        model_config = AutoConfig.from_pretrained(config.model_name)
        self.backbone = AutoModel.from_pretrained(config.model_name, config=model_config)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.head = nn.Linear(model_config.hidden_size, config.num_labels)

    def forward(self, input_ids, mask):
        x = self.backbone(input_ids, mask)
        logits1 = self.head(self.dropout1(x[0]))
        logits2 = self.head(self.dropout2(x[0]))
        logits3 = self.head(self.dropout3(x[0]))
        logits4 = self.head(self.dropout4(x[0]))
        logits5 = self.head(self.dropout5(x[0]))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        return logits


def build_model_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, add_prefix_space=True)
    model = FeedbackModel(config)
    return model, tokenizer
