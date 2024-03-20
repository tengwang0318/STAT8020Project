from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
from config import *
import pandas as pd
import torch.nn as nn
import gc

def active_logits(raw_logits: torch.Tensor, word_ids: torch.Tensor, config: Config):
    word_ids = word_ids.view(-1)
    active_mask = word_ids.unsqueeze(1).expand(word_ids.shape[0], config.num_labels)
    active_mask = active_mask != NON_LABEL
    active_logits = raw_logits.view(-1, config.num_labels)
    active_logits = torch.masked_select(active_logits, active_mask)
    active_logits = active_logits.view(-1, config.num_labels)
    return active_logits


def active_labels(labels: torch.Tensor):
    active_mask = labels.view(-1) != IGNORE_INDEX
    active_labels = torch.masked_select(labels.view(-1), active_mask)
    return active_labels


def active_preds_prob(active_logits):
    active_preds = torch.argmax(active_logits, dim=1)
    active_preds_prob, _ = torch.max(active_logits, dim=1)
    return active_preds, active_preds_prob


def calc_overlap(row):
    """
    calculate the overlap between prediction and ground truth
    """
    set_pred = set(row.new_predictionstring_pred.split(' '))
    set_gt = set(row.new_predictionstring_gt.split(' '))
    # length of each and intersection
    len_pred = len(set_pred)
    len_gt = len(set_gt)
    intersection = len(set_gt.intersection(set_pred))
    overlap_1 = intersection / len_gt
    overlap_2 = intersection / len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp(pred_df, gt_df):
    gt_df = gt_df[['id', 'discourse_type', 'new_predictionstring']].reset_index(drop=True).copy()
    pred_df = pred_df[['id', 'class', 'new_predictionstring']].reset_index(drop=True).copy()
    gt_df['gt_id'] = gt_df.index
    pred_df['pred_id'] = pred_df.index
    joined = pred_df.merge(
        gt_df,
        left_on=['id', 'class'],
        right_on=['id', 'discourse_type'],
        how='outer',
        suffixes=['_pred', '_gt']
    )
    joined['new_predictionstring_gt'] = joined['new_predictionstring_gt'].fillna(' ')
    joined['new_predictionstring_pred'] = joined['new_predictionstring_pred'].fillna(' ')
    joined['overlaps'] = joined.apply(calc_overlap, axis=1)
    # overlap over 0.5: true positive
    # If nultiple overlaps exists, the higher is taken.
    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])

    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >= 0.5)
    joined['max_overlap'] = joined[['overlap1', 'overlap2']].max(axis=1)
    tp_pred_ids = joined.query('potential_TP').sort_values('max_overlap', ascending=False) \
        .groupby(['id', 'new_predictionstring_gt']).first()['pred_id'].values

    fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in tp_pred_ids]
    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
    unmatched_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    macro_f1_score = TP / (TP + 1 / 2 * (FP + FN))
    return macro_f1_score


def oof_score(df_val, oof):
    f1score = []
    classes = ['Lead', 'Position', 'Claim', 'Counterclaim', 'Rebuttal', 'Evidence', 'Concluding Statement']
    for c in classes:
        pred_df = oof.loc[oof['class'] == c].copy()
        gt_df = df_val.loc[df_val['discourse_type'] == c].copy()
        f1 = score_feedback_comp(pred_df, gt_df)
        print(f'{c:<10}: {f1:4f}')
        f1score.append(f1)
    f1avg = np.mean(f1score)
    return f1avg


def inference(model, dl, criterion, valid_flg, config: Config):
    stream = tqdm(dl)
    model.eval()

    valid_loss = 0
    valid_accuracy = 0
    all_logits = None
    device = config.device
    for batch_idx, batch in enumerate(stream, start=1):
        ids = batch['input_ids'].to(device, dtype=torch.long)
        mask = batch['attention_mask'].to(device, dtype=torch.long)
        with torch.no_grad():
            raw_logits = model(input_ids=ids, mask=mask)
        del ids, mask

        word_ids = batch['word_ids'].to(device, dtype=torch.long)
        logits = active_logits(raw_logits, word_ids, config)
        sf_logits = torch.softmax(logits, dim=-1)
        sf_raw_logits = torch.softmax(raw_logits, dim=-1)
        if valid_flg:
            raw_labels = batch['labels'].to(device, dtype=torch.long)
            labels = active_labels(raw_labels)
            preds, preds_prob = active_preds_prob(sf_logits)
            valid_accuracy += accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            loss = criterion(logits, labels)
            valid_loss += loss.item()

        if batch_idx == 1:
            all_logits = sf_raw_logits.cpu().numpy()
        else:
            all_logits = np.append(all_logits, sf_raw_logits.cpu().numpy(), axis=0)

    if valid_flg:
        epoch_loss = valid_loss / batch_idx
        epoch_accuracy = valid_accuracy / batch_idx
    else:
        epoch_loss, epoch_accuracy = 0, 0
    return all_logits, epoch_loss, epoch_accuracy


def preds_class_prob(all_logits, dl, config):
    print("predict target class and its probabilty")
    final_predictions = []
    final_predictions_score = []
    stream = tqdm(dl)
    len_sample = all_logits.shape[0]

    for batch_idx, batch in enumerate(stream, start=0):
        for minibatch_idx in range(config.valid_batch_size):
            sample_idx = int(batch_idx * config.valid_batch_size + minibatch_idx)
            if sample_idx > len_sample - 1: break
            word_ids = batch['word_ids'][minibatch_idx].numpy()
            predictions = []
            predictions_prob = []
            pred_class_id = np.argmax(all_logits[sample_idx], axis=1)
            pred_score = np.max(all_logits[sample_idx], axis=1)
            pred_class_labels = [IDS_TO_LABELS[i] for i in pred_class_id]
            prev_word_idx = -1
            for idx, word_idx in enumerate(word_ids):
                if word_idx == -1:
                    pass
                elif word_idx != prev_word_idx:
                    predictions.append(pred_class_labels[idx])
                    predictions_prob.append(pred_score[idx])
                    prev_word_idx = word_idx
            final_predictions.append(predictions)
            final_predictions_score.append(predictions_prob)
    return final_predictions, final_predictions_score


def get_preds_onefold(model, df, dl, criterion, valid_flg, config):
    logits, valid_loss, valid_acc = inference(model, dl, criterion, valid_flg, config)
    all_preds, all_preds_prob = preds_class_prob(logits, dl, config)
    df_pred = post_process_pred(df, all_preds, all_preds_prob)
    return df_pred, valid_loss, valid_acc


def get_preds_folds(model, df, dl, criterion, valid_flg=False, config: Config = None):
    for i_fold in range(config.n_fold):
        model_filename = os.path.join(config.model_dir, f"{config.model_name}_{i_fold}.bin")
        print(f"{model_filename} inference")
        model = model.to(config.device)
        model.load_state_dict(torch.load(model_filename))
        logits, valid_loss, valid_acc = inference(model, dl, criterion, valid_flg, config)
        if i_fold == 0:
            avg_pred_logits = logits
        else:
            avg_pred_logits += logits
    avg_pred_logits /= config.n_fold
    all_preds, all_preds_prob = preds_class_prob(avg_pred_logits, dl, config)
    df_pred = post_process_pred(df, all_preds, all_preds_prob)
    return df_pred


def post_process_pred(df, all_preds, all_preds_prob):
    final_preds = []
    for i in range(len(df)):
        idx = df.id.values[i]
        pred = all_preds[i]
        pred_prob = all_preds_prob[i]
        j = 0
        while j < len(pred):
            cls = pred[j]
            if cls == 'O':
                j += 1
            else:
                cls = cls.replace('B', 'I')
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1
            if cls != 'O' and cls != '':
                avg_score = np.mean(pred_prob[j:end])
                if end - j > MIN_THRESH[cls] and avg_score > PROB_THRESH[cls]:
                    final_preds.append((idx, cls.replace('I-', ''), ' '.join(map(str, list(range(j, end))))))
            j = end
    df_pred = pd.DataFrame(final_preds)
    df_pred.columns = ['id', 'class', 'new_predictionstring']
    return df_pred


def train_fn(model, dl_train, optimizer, epoch, criterion, config: Config):
    model.train()
    train_loss = 0
    train_accuracy = 0
    stream = tqdm(dl_train)
    scaler = GradScaler()

    for batch_idx, batch in enumerate(stream, start=1):
        ids = batch['input_ids'].to(config.device, dtype=torch.long)
        mask = batch['attention_mask'].to(config.device, dtype=torch.long)
        raw_labels = batch['labels'].to(config.device, dtype=torch.long)
        word_ids = batch['word_ids'].to(config.device, dtype=torch.long)
        optimizer.zero_grad()
        with autocast():
            raw_logits = model(input_ids=ids, mask=mask)

        logits = active_logits(raw_logits, word_ids, config)
        labels = active_labels(raw_labels)
        sf_logits = torch.softmax(logits, dim=-1)
        preds, preds_prob = active_preds_prob(sf_logits)
        train_accuracy += accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

        if batch_idx % Config.verbose_steps == 0:
            loss_step = train_loss / batch_idx
            print(f'Training loss after {batch_idx:04d} training steps: {loss_step}')

    epoch_loss = train_loss / batch_idx
    epoch_accuracy = train_accuracy / batch_idx
    del dl_train, raw_logits, logits, raw_labels, preds, labels
    torch.cuda.empty_cache()
    gc.collect()
    print(f'epoch {epoch} - training loss: {epoch_loss:.4f}')
    print(f'epoch {epoch} - training accuracy: {epoch_accuracy:.4f}')


def valid_fn(model, df_val, df_val_eval, dl_val, epoch, criterion):
    oof, valid_loss, valid_acc = get_preds_onefold(model, df_val, dl_val, criterion, valid_flg=True)
    f1score = []
    # classes = oof['class'].unique()
    classes = ['Lead', 'Position', 'Claim', 'Counterclaim', 'Rebuttal', 'Evidence', 'Concluding Statement']
    print(f"Validation F1 scores")

    for c in classes:
        pred_df = oof.loc[oof['class'] == c].copy()
        gt_df = df_val_eval.loc[df_val_eval['discourse_type'] == c].copy()
        f1 = score_feedback_comp(pred_df, gt_df)
        print(f' * {c:<10}: {f1:4f}')
        f1score.append(f1)
    f1avg = np.mean(f1score)
    print(f'Overall Validation avg F1: {f1avg:.4f} val_loss:{valid_loss:.4f} val_accuracy:{valid_acc:.4f}')
    return valid_loss, oof
