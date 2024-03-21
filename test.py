import argparse
import numpy as np
import random
from model import build_model_tokenizer
from dataset import FeedbackPrizeDataset
from utility import get_preds_folds
import torch.nn as nn
from torch.utils.data import DataLoader

from data_proprocess import preprocess
from config import *

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default="")
parser.add_argument("--is_training", type=bool, default=False)
parser.add_argument("--model_name", type=str, default="longformer")
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--n_folds", type=int, default=5)
parser.add_argument("--train_batch_size", type=int, default=2)
parser.add_argument("--valid_batch_size", type=int, default=2)
args = parser.parse_args()
config = Config(args)


def seed_everything(seed=64):
    # os.environ['PYTHONSEED'] = str(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()
IGNORE_INDEX = -100
test_texts = preprocess(config)

model, tokenizer = build_model_tokenizer(config)
criterion = nn.CrossEntropyLoss()
ds_test = FeedbackPrizeDataset(test_texts, tokenizer, config.max_length, False, config)
dl_test = DataLoader(ds_test, batch_size=config.train_batch_size, shuffle=False, num_workers=0, pin_memory=True)
sub = get_preds_folds(model, test_texts, dl_test, criterion, False, config)
sub.columns = ['id', 'class', 'predictionstring']

sub_filename = 'submission.csv'
sub.to_csv(sub_filename, index=False)
