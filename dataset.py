from torch.utils.data import Dataset
from config import *


class FeedbackPrizeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, has_labels, config):
        super(FeedbackPrizeDataset, self).__init__()
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.has_labels = has_labels
        self.config = config

    def __getitem__(self, index):
        text = self.data.text[index]
        encoded_text = self.tokenizer(text.split(),
                                      is_split_into_words=True,
                                      truncation=True,
                                      max_length=self.max_len,
                                      padding="max_length",
                                      return_tensors="pt",
                                      )
        word_ids = encoded_text.word_ids()

        encoded_text = {k: v.squeeze(0) for k, v in encoded_text.items()}

        # encoded_text = self.tokenizer(text,
        #                               truncation=True,
        #                               max_length=self.max_len,
        #                               padding=True,
        #                               return_tensors="pt",
        #                               )
        # print(encoded_text['input_ids'].shape)

        if self.has_labels:
            word_labels = self.data.entities[index]
            prev_word_idx = None
            labels_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    labels_ids.append(IGNORE_INDEX)
                elif word_idx != prev_word_idx:
                    labels_ids.append(LABELS_TO_IDS[word_labels[word_idx]])
                else:
                    if self.config.label_subtokens:
                        labels_ids.append(LABELS_TO_IDS[word_labels[word_idx]])
                    else:
                        labels_ids.append(IGNORE_INDEX)

                prev_word_idx = word_idx
            encoded_text['labels'] = labels_ids

        item = {k: torch.as_tensor(v) for k, v in encoded_text.items()}
        # if index == 6: print(item)
        word_ids2 = [w if w is not None else NON_LABEL for w in word_ids]
        item['word_ids'] = torch.as_tensor(word_ids2)
        return item

    def __len__(self):
        return self.len