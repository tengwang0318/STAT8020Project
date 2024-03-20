import os
import torch


class Config:
    def __init__(self, args):
        self.is_training = "train" if args.is_training else "test"
        model_name = args.model_name
        if model_name == "longformer":
            self.model_name = "allenai/longformer-base-4096"
        elif model_name == "roberta-base":
            self.model_name = "FacebookAI/roberta-base"
        elif model_name == "roberta-large":
            self.model_name = "FacebookAI/roberta-large"
        self.base_dir = os.getcwd()
        self.data_dir = os.path.join(self.base_dir, "feedback-prize-2021")
        self.output_dir = os.path.join(self.base_dir, "outputs")
        self.model_dir = os.path.join(self.base_dir, "models")
        self.epochs = args.epochs
        self.n_fold = args.n_folds

        if model_name == "longformer":
            self.max_length = 1024
            self.inference_max_length = 4096
            self.lr = 4e-5
        elif model_name == "roberta-large":
            self.max_length = 512
            self.inference_max_length = 512
            self.lr = 1e-5
        elif model_name == "roberta-base":
            self.max_length = 512
            self.inference_max_length = 512
            self.lr = 8e-5
        self.train_batch_size = args.train_batch_size
        self.valid_batch_size = args.valid_batch_size

        self.num_labels = 15
        self.label_subtokens = True
        self.hidden_dropout_prob = 0.1

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


IGNORE_INDEX = -100
NON_LABEL = -1
OUTPUT_LABELS = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim',
                 'I-Counterclaim',
                 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement',
                 'I-Concluding Statement']
LABELS_TO_IDS = {v: k for k, v in enumerate(OUTPUT_LABELS)}
IDS_TO_LABELS = {k: v for k, v in enumerate(OUTPUT_LABELS)}

MIN_THRESH = {
    "I-Lead": 9,
    "I-Position": 5,
    "I-Evidence": 14,
    "I-Claim": 3,
    "I-Concluding Statement": 11,
    "I-Counterclaim": 6,
    "I-Rebuttal": 4,
}

PROB_THRESH = {
    "I-Lead": 0.7,
    "I-Position": 0.55,
    "I-Evidence": 0.65,
    "I-Claim": 0.55,
    "I-Concluding Statement": 0.7,
    "I-Counterclaim": 0.5,
    "I-Rebuttal": 0.55,
}
