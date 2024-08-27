# This source code is licensed under CC BY 4.0 license.
# Copyright 2024 Samsung Electronics Co., Ltd.
import pickle

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel, logging

logging.set_verbosity_error()


class Bert_Features(nn.Module):
    def __init__(self, meta_path):
        super(Bert_Features, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-multilingual-uncased")
        with open(meta_path, 'rb') as file_to_read:
            self.meta_file = pickle.load(file_to_read)

    def get_seq_features(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=50, padding='max_length',
                                truncation=True)  # .to(self.device)

        with torch.no_grad():
            last_hidden_states = self.bert(**inputs)["last_hidden_state"]

        return last_hidden_states

    def get_features_filename(self, filename):
        text = self.meta_file[filename]['text']

        return self.get_seq_features(text)
