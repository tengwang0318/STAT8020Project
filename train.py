import argparse
import numpy as np
import random
from model import build_model_tokenizer
from dataset import FeedbackPrizeDataset
import pandas as pd
from utility import train_fn, valid_fn, oof_score
import torch.nn as nn
from torch.utils.data import DataLoader

from data_proprocess import preprocess
from config import *

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default="")
parser.add_argument("--is_training", type=bool, default=True)
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
all_train_df = pd.read_csv(config.data_dir + "/corrected_train.csv")
all_train_texts = preprocess(config)

oof = pd.DataFrame()
for i_fold in range(config.n_fold):
    print(f'=== fold{i_fold} training ===')
    model, tokenizer = build_model_tokenizer(config)
    model = model.to(config.device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)

    df_train = all_train_texts[all_train_texts['fold'] != i_fold].reset_index(drop=True)
    ds_train = FeedbackPrizeDataset(df_train, tokenizer, config.max_length, True, config)
    df_val = all_train_texts[all_train_texts['fold'] == i_fold].reset_index(drop=True)
    val_idlist = df_val['id'].unique().tolist()
    df_val_eval = all_train_df.query('id==@val_idlist').reset_index(drop=True)
    ds_val = FeedbackPrizeDataset(df_val, tokenizer, config.max_length, True, config)
    dl_train = DataLoader(ds_train, batch_size=config.train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=config.valid_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    best_val_loss = np.inf
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, config.epochs + 1):
        train_fn(model, dl_train, optimizer, epoch, criterion,config)
        valid_loss, _oof = valid_fn(model, df_val, df_val_eval, dl_val, epoch, criterion, config)
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            _oof_fold_best = _oof
            _oof_fold_best['fold'] = i_fold 
            model_filename = f'{config.model_dir}/{config.model_save_path}_{i_fold}.bin'
            torch.save(model.state_dict(), model_filename)
            print(f'{model_filename} saved')

    oof = pd.concat([oof, _oof_fold_best])

# oof = pd.DataFrame()
# for i_fold in range(config.n_fold):
#     print(f'=== fold{i_fold} training ===')
#     model, tokenizer = build_model_tokenizer(config)
#     model = model.to(config.device)
#     optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
#
#     df_train = all_train_texts[all_train_texts['fold'] != i_fold].reset_index(drop=True)
#     ds_train = FeedbackPrizeDataset(df_train, tokenizer, config.max_length, True, config)
#     df_val = all_train_texts[all_train_texts['fold'] == i_fold].reset_index(drop=True)
#     val_idlist = df_val['id'].unique().tolist()
#     df_val_eval = all_train_df.query('id==@val_idlist').reset_index(drop=True)
#     ds_val = FeedbackPrizeDataset(df_val, tokenizer, config.max_length, True, config)
#     dl_train = DataLoader(ds_train, batch_size=config.train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
#     dl_val = DataLoader(ds_val, batch_size=config.valid_batch_size, shuffle=False, num_workers=2, pin_memory=True)
#
#     best_val_loss = np.inf
#     criterion = nn.CrossEntropyLoss()
#
#     for epoch in range(1, config.epochs + 1):
#         train_fn(model, dl_train, optimizer, epoch, criterion, config)
#         valid_loss, _oof = valid_fn(model, df_val, df_val_eval, dl_val, epoch, criterion, config)
#         if valid_loss < best_val_loss:
#             best_val_loss = valid_loss
#             _oof_fold_best = _oof
#             _oof_fold_best['fold'] = i_fold
#             model_filename = f'{config.model_dir}/{config.model_name}_{i_fold}.bin'
#             torch.save(model.state_dict(), model_filename)
#             print(f'{model_filename} saved')
#
#     oof = pd.concat([oof, _oof_fold_best])
oof.to_csv(f"{config.output_dir}/oof_{config.model_save_path}", index=False)
print(f"overall cv score: {oof_score(all_train_df, oof)}")
