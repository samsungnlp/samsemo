# This source code is licensed under CC BY 4.0 license.
# Copyright 2024 Samsung Electronics Co., Ltd.
import os

import torch
import torch.nn.functional as F
import torchaudio


class Specgram():
    def __init__(self, files_path: str, utt_name: str, audio_dir: str = ''):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.files_path = files_path
        self.utt_name = utt_name
        self.predict = False
        self.audio_dir = audio_dir

    @staticmethod
    def cutSpecToPieces(spec, stride=32):
        # Split the audio waveform by second
        total = -(-spec.size(-1) // stride)
        specs = []
        for i in range(total):
            specs.append(spec[:, :, :, i * stride:(i + 1) * stride])

        # Pad the last piece
        lastPieceLength = specs[-1].size(-1)
        if lastPieceLength < stride:
            padRight = stride - lastPieceLength
            specs[-1] = F.pad(specs[-1], (0, padRight))

        return specs

    def specgram(self):
        if self.predict:
            waveform, sr = torchaudio.load(os.path.join(self.audio_dir, 'audio.wav'))
        else:
            waveform, sr = torchaudio.load(os.path.join(self.files_path, self.utt_name, 'audio.wav'))

        specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sr, win_length=int(float(sr) / 16000 * 400),
            n_fft=int(float(sr) / 16000 * 400)).cpu()(waveform.cpu()).unsqueeze(0)
        specgrams = self.cutSpecToPieces(specgram)
        specgrams = torch.cat(specgrams)

        # version with uniform dims
        length = len(specgrams)
        if length > 32:
            specgrams = specgrams[:32]
        else:
            shapes = specgrams.shape
            oriLen, dim = shapes[0], (shapes[1], shapes[2], shapes[3])
            specgrams = torch.cat((specgrams, torch.zeros(32 - oriLen, *dim).cpu()), dim=0)

        return specgrams.unsqueeze(0), length
