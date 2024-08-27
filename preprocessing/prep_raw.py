# This source code is licensed under CC BY 4.0 license.
# Copyright 2024 Samsung Electronics Co., Ltd.
import os
import argparse
import pickle

import torch
from preprocessing.prep_aud import Specgram
from preprocessing.prep_text import Bert_Features
from preprocessing.prep_video import VideoProcessing
from tqdm import tqdm


class Preparation:
    def __init__(self, files_path, utt_names, meta_path, split_type='train'):
        super(Preparation, self).__init__()
        self.files_path = files_path
        self.utt_names = utt_names
        self.split_type = split_type
        self.meta_path = meta_path

        self.bert = Bert_Features(meta_path=self.meta_path)
        self.prep_video = VideoProcessing(frames_path=files_path)

    def prep_raw_text(self):
        text_features = []
        for utt_name in tqdm(self.utt_names):
            filename = utt_name[:-1]
            text_feature = self.bert.get_seq_features(filename)
            text_features.append(text_feature)
        text_features = torch.cat(text_features)
        return text_features

    def prep_raw_audio(self):
        audio_features = []
        spec_lens = []
        for utt_name in tqdm(self.utt_names):
            utt_name = utt_name[:-1]
            audio_feature, spec_len = Specgram(files_path=self.files_path, utt_name=utt_name).specgram()
            audio_features.append(audio_feature)
            spec_lens.append(spec_len)

        return torch.cat(audio_features, dim=0).cpu(), spec_lens

    def prep_video_frames(self):
        vid_features = []
        imgs_lens = []
        for utt_name in tqdm(self.utt_names):
            try:
                vid_feature, imgs_len = self.prep_video.frames_processing_filename(video_filename=utt_name[:-1])
                vid_features.append(vid_feature)
                imgs_lens.append(imgs_len)
            except Exception as e:
                print(e)
                vid_features.append(torch.zeros(1, 3, 48, 48))
                imgs_lens.append(1)

        return torch.cat(vid_features).cpu(), imgs_lens


def prepare_dict(files_path, utt_names_path: str, utt_file: str, meta_path: str, split_type: str) -> dict:
    with open(f'{utt_names_path}{split_type}{utt_file}', 'r') as f:
        utt_names = f.readlines()

    preparation = Preparation(files_path=files_path, utt_names=utt_names, meta_path=meta_path, split_type=split_type)
    vision_features, imgs_lens = preparation.prep_video_frames()
    text_features = preparation.prep_raw_text()
    audio_features, spec_lens = preparation.prep_raw_audio()

    with open(meta_path, 'rb') as meta:
        meta_file = pickle.load(meta)

    labels = [meta_file[idx[:-1]]['label'] for idx in utt_names]

    dict_split = {'text': text_features, 'audio': audio_features, 'spec_lens': spec_lens, 'vision': vision_features,
                  'imgs_lens': imgs_lens, 'labels': labels}
    return dict_split


def prepare_pickle_data(files_path, utt_names_path, utt_file, meta_path, data_path, dataset_name):
    dict_train = prepare_dict(files_path, utt_names_path, utt_file, meta_path, split_type='train')
    dict_valid = prepare_dict(files_path, utt_names_path, utt_file, meta_path, split_type='valid')
    dict_test = prepare_dict(files_path, utt_names_path, utt_file, meta_path, split_type='test')
    dict_all = {'train': dict_train, 'valid': dict_valid, 'test': dict_test}
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(f'{data_path}{dataset_name}', 'wb') as handle:
        pickle.dump(dict_all, handle)
