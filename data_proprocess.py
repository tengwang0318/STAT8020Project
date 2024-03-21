import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
import os
from config import Config


def get_df_train(config: Config):
    df_train = pd.read_csv(config.data_dir + "/corrected_train.csv")
    return df_train


def agg_essays(config: Config):
    folder = config.is_training
    names, texts = [], []
    for f in tqdm(list(os.listdir(f'{config.data_dir}/{folder}'))):
        names.append(f.replace('.txt', ''))
        texts.append(open(f'{config.data_dir}/{folder}/' + f, 'r').read())
    df_texts = pd.DataFrame({'id': names, 'text': texts})
    df_texts['text_split'] = df_texts.text.str.split()
    print('Completed tokenizing texts.')
    return df_texts


def ner(df_texts, df_train):
    all_entities = []
    for _, row in tqdm(df_texts.iterrows(), total=len(df_texts)):
        total = len(row['text_split'])
        entities = ['O'] * total

        for _, row2 in df_train[df_train['id'] == row['id']].iterrows():
            discourse = row2['discourse_type']
            list_ix = [int(x) for x in row2['predictionstring'].split(' ')]
            entities[list_ix[0]] = f'B-{discourse}'
            for k in list_ix[1:]: entities[k] = f'I-{discourse}'
        all_entities.append(entities)

    df_texts['entities'] = all_entities
    print('Completed mapping discourse to each token.')
    return df_texts


def split_fold(df_train, config):
    ids = df_train['id'].unique()
    kf = KFold(n_splits=config.n_fold, shuffle=True, random_state=64)
    for i_fold, (_, valid_index) in enumerate(kf.split(ids)):
        df_train.loc[valid_index, 'fold'] = i_fold
    return df_train


def preprocess(config):
    # df_train = get_df_train(config)
    df_texts = agg_essays(config)

    if config.is_training=="train":
        df_train = get_df_train(config)
        df_texts = ner(df_texts, df_train)
        df_texts = split_fold(df_texts, config)
    return df_texts
