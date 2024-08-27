# This source code is licensed under CC BY 4.0 license.
# Copyright 2024 Samsung Electronics Co., Ltd.
import importlib
import sys

import torch
from src.transformer_encoder import WrappedTransformerEncoder
from src.vgg_block import VggBasicBlock
from torch import nn
import lightning as L


class MME2E(nn.Module):
    def __init__(self, hyp_params):
        super(MME2E, self).__init__()
        self.num_classes = len(hyp_params.emotions)
        self.hyp_params = hyp_params
        self.mod = hyp_params.modalities.lower()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        nlayers = hyp_params.trans_nlayers
        nheads = hyp_params.trans_nheads
        trans_dim = hyp_params.trans_dim

        text_cls_dim = 768
        self.text_cls_dim = text_cls_dim

        self.V = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), VggBasicBlock(in_planes=64, out_planes=64),
            nn.Dropout(hyp_params.dropout), VggBasicBlock(in_planes=64, out_planes=64), nn.Dropout(hyp_params.dropout),
            VggBasicBlock(in_planes=64, out_planes=128), nn.Dropout(hyp_params.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2), VggBasicBlock(in_planes=128, out_planes=256),
            nn.Dropout(hyp_params.dropout), nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=256, out_planes=512), nn.Dropout(hyp_params.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2), )

        self.A = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2),
            nn.Dropout(hyp_params.dropout), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=64, out_planes=64), nn.Dropout(hyp_params.dropout),
            VggBasicBlock(in_planes=64, out_planes=64), nn.Dropout(hyp_params.dropout),
            VggBasicBlock(in_planes=64, out_planes=128), nn.Dropout(hyp_params.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2), VggBasicBlock(in_planes=128, out_planes=256),
            nn.Dropout(hyp_params.dropout), nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=256, out_planes=512), nn.Dropout(hyp_params.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2), )

        self.v_flatten = nn.Sequential(nn.Linear(512 * 3 * 3, 1024), nn.ReLU(), nn.Linear(1024, trans_dim))

        self.a_flatten = nn.Sequential(nn.Linear(512 * 8 * 2, 1024), nn.ReLU(), nn.Linear(1024, trans_dim))

        self.v_transformer = WrappedTransformerEncoder(dim=trans_dim, num_layers=nlayers, num_heads=nheads)
        self.a_transformer = WrappedTransformerEncoder(dim=trans_dim, num_layers=nlayers, num_heads=nheads)

        self.v_out = nn.Linear(trans_dim, self.num_classes)
        self.t_out = nn.Linear(text_cls_dim, self.num_classes)
        self.a_out = nn.Linear(trans_dim, self.num_classes)

        self.weighted_fusion = nn.Linear(len(self.mod), 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text, specs, spec_lens, faces, imgs_lens):
        all_logits = []

        if 't' in self.mod:
            text = text[:, 0].to(self.device)
            text_out = self.t_out(text)
            all_logits.append(text_out)

        if 'v' in self.mod:
            # the following lines are for removing zeros from the tensor with faces, which were
            # added to match in the batch
            # 16 is the maximal number of images

            imgs_lens = torch.where(imgs_lens < 16, imgs_lens, 16)
            mask = (torch.arange(int(faces.shape[1])).repeat(int(faces.shape[0]), 1).to(
                self.device) < imgs_lens.unsqueeze(1)).to(self.device)

            faces = faces.to(self.device)[mask].to(self.device)

            # before convs we should reduce faces variable to true faces using imgs_lens

            faces = self.V(faces)
            faces = self.v_flatten(faces.flatten(start_dim=1))

            faces = self.v_transformer(faces, lens=imgs_lens.tolist(), get_cls=True)
            v_out = self.v_out(faces)
            all_logits.append(v_out)

        if 'a' in self.mod:
            # 32 is number of specgrams in one file

            # the following lines are for remove zeros from the specs tensor which were
            # added to match in the batch
            spec_lens = torch.where(spec_lens < 32, spec_lens, 32)
            mask = torch.arange(int(specs.shape[1])).repeat(int(specs.shape[0]), 1).to(
                self.device) < spec_lens.unsqueeze(1).to(self.device)

            specs = specs.to(self.device)[mask].to(self.device)

            for a_module in self.A:
                specs = a_module(specs)

            specs = self.a_flatten(specs.flatten(start_dim=1))

            specs = self.a_transformer(specs, lens=spec_lens.tolist(), get_cls=True)
            audio_out = self.a_out(specs)

            all_logits.append(audio_out)

        if len(self.mod) == 1:
            return all_logits[0], 0

        last_hs = torch.stack(all_logits, dim=-1)

        return self.weighted_fusion(last_hs).squeeze(-1), last_hs
