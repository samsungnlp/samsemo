# This source code is licensed under CC BY 4.0 license.
# Copyright 2024 Samsung Electronics Co., Ltd.
import os
import pickle

import torch
from torch.utils.data.dataset import Dataset


############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################


class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data, split_type='train'):
        super(Multimodal_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, data)
        dataset = pickle.load(open(dataset_path, 'rb'))

        self.audio = dataset[split_type]['audio']
        self.vision = dataset[split_type]['vision']
        self.text = dataset[split_type]['text']

        self.spec_lens = dataset[split_type]['spec_lens']
        self.imgs_lens = dataset[split_type]['imgs_lens']

        self.labels = torch.tensor(dataset[split_type]['labels'])
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None
        self.n_modalities = 3  # vision/ text/ audio

    def get_n_modalities(self):
        return self.n_modalities

    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return index, self.text[index], self.audio[index], self.spec_lens[index], self.vision[index], self.imgs_lens[
            index], self.labels[index]
