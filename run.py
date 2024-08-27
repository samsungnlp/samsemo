# This source code is licensed under CC BY 4.0 license.
# Copyright 2024 Samsung Electronics Co., Ltd.

import argparse
from tabulate import tabulate

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src.dataset import Multimodal_Datasets
from src.model_E2E import MME2E
from src.evaluate import eval_emo
from src.evaluate import make_stat
from preprocessing.prep_raw import prepare_pickle_data

torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--train_from_raw_data', help='train from raw data or preprocess', type=bool, required=True,
                    default=True)

# arguments for preprocessing data
parser.add_argument('--base_dir', help='Base directory with raw files', type=str, required=True,
                    default='/usr/share/emotions/data/samsemo/')
parser.add_argument('--utt_names_path', help='Dir with utterances (text)', type=str, required=True,
                    default='final_selection/meta_and_splits/')
parser.add_argument('--utt_file', help='Name of utt file', type=str, required=False, default='_split_EN.txt')
parser.add_argument('--meta_path', help='Path of meta file', type=str, required=False,
                    default='final_selection/meta_and_splits/meta.pkl')
parser.add_argument('-fp', '--files_path', help='Path of audio and frames', type=str, required=False,
                    default='final_selection/segmented/combined/')
parser.add_argument('--data_path', type=str, default='preprocessing/data/', help='path for storing the dataset')

# training params
parser.add_argument('--dataset_name', type=str, default="samsemo_en_article.pkl", required=True,
                    help='dataset to use')
parser.add_argument("--learning_rate", default=1e-4, type=float, required=False, help="learning rate")
parser.add_argument("--batch_size", default=16, type=int, required=False, help="batch size")
parser.add_argument("--max_epochs", default=50, type=int, required=False, help="number of epochs")
parser.add_argument("--checkpoints_dir", default="checkpoints_emo", type=str, required=False,
                    help="directory to save checkpoints")
parser.add_argument('--emotions', type=str, default=['anger', 'happiness', 'sadness', 'surprise', 'neutral'],
                    help='emotions (default: emotions for mosei)')
parser.add_argument('--modalities', type=str, default='tav')
parser.add_argument('--clip', type=float, default=0.8, help='gradient clip value (default: 0.8)')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='l2-regularization (default: 1e-4)')
parser.add_argument('--trans-nlayers', help='Number of layers of the transformer after CNN', type=int,
                    required=False, default=4)
parser.add_argument('--trans-nheads', help='Number of heads of the transformer after CNN', type=int, required=False,
                    default=4)
parser.add_argument('--trans-dim', help='Dimension of the transformer after CNN', type=int, required=False,
                    default=64)
parser.add_argument('--num_heads', type=int, default=4,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--when', type=int, default=20, help='when to decay learning rate (default: 20)')

args = parser.parse_args()


class EmoModel(L.LightningModule):
    def __init__(self, hyp_params):
        super(EmoModel, self).__init__()
        self.model = MME2E(hyp_params).cuda()
        self.hyp_params = hyp_params
        self.scheduler = ReduceLROnPlateau(self.configure_optimizers(), mode='min', patience=hyp_params.when,
                                           factor=0.1, verbose=True)
        self.results_train = []
        self.gt_emotions_train = []

        self.results_valid = []
        self.gt_emotions_valid = []

        self.results_test = []
        self.gt_emotions_test = []
        self.train_thresholds = [0.5] * len(hyp_params.emotions)

        n = len(hyp_params.emotions) + 1
        self.prev_train_stats = [[-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n,
                                 [-float('inf')] * n]
        self.prev_valid_stats = [[-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n,
                                 [-float('inf')] * n]
        self.prev_test_stats = [[-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n,
                                [-float('inf')] * n]

    def forward(self, text, audio, spec_lens, vision, imgs_lens):
        return self.model(text, audio, spec_lens, vision, imgs_lens)

    def _common_step(self, batch):
        utteranceId, text, audio, spec_lens, vision, imgs_lens, labels = batch
        preds, hiddens = self.forward(text, audio, spec_lens, vision, imgs_lens)
        loss_func = torch.nn.BCEWithLogitsLoss()
        loss = loss_func(preds, labels.type_as(preds))
        return preds, loss

    def training_step(self, batch):
        preds, loss = self._common_step(batch)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyp_params.clip)
        self.results_train.extend(preds)
        self.gt_emotions_train.extend(batch[-1])
        self.log("loss", loss, prog_bar=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch):
        preds, loss = self._common_step(batch)
        self.scheduler.step(loss)
        self.results_valid.extend(preds)
        self.gt_emotions_valid.extend(batch[-1])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch):
        preds, loss = self._common_step(batch)
        self.results_test.extend(preds)
        self.gt_emotions_test.extend(batch[-1])
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def _common_step_epoch_end(self, phase):
        if phase == "train":
            results = torch.stack(self.results_train)
            gt_emotions = torch.stack(self.gt_emotions_train)
            prev_stats = self.prev_train_stats
            stats, self.train_thresholds = eval_emo(results, gt_emotions)
        elif phase == "valid":
            print(f'self.train_thresholds are {self.train_thresholds}')
            results = torch.stack(self.results_valid)
            gt_emotions = torch.stack(self.gt_emotions_valid)
            prev_stats = self.prev_valid_stats
            stats, _ = eval_emo(results, gt_emotions)
        else:
            results = torch.stack(self.results_test)
            gt_emotions = torch.stack(self.gt_emotions_test)
            prev_stats = self.prev_test_stats
            stats, _ = eval_emo(results, gt_emotions, self.train_thresholds)

        headers = [['phase (acc)', *self.hyp_params.emotions, 'average'],
                   ['phase (recall)', *self.hyp_params.emotions, 'average'],
                   ['phase (precision)', *self.hyp_params.emotions, 'average'],
                   ['phase (f1)', *self.hyp_params.emotions, 'average'],
                   ['phase (auc)', *self.hyp_params.emotions, 'average']]
        if phase == "train":
            f1_average = stats[-2][-1]
            self.log("f1_average_train", f1_average, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.prev_train_stats = stats
        elif phase == "valid":
            f1_average = stats[-2][-1]
            self.log("f1_average_val", f1_average, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.prev_valid_stats = stats
        else:
            f1_average = stats[-2][-1]
            self.log("f1_average_test", f1_average, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.prev_test_stats = stats

        for i in range(len(headers)):
            stats_str = make_stat(prev_stats[i], stats[i])
            print(tabulate([[phase, *stats_str], ], headers=headers[i]))

    def on_train_epoch_end(self):
        self._common_step_epoch_end(phase="train")

    def on_validation_epoch_end(self):
        self._common_step_epoch_end(phase="valid")

    def on_test_epoch_end(self):
        self._common_step_epoch_end(phase="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hyp_params.learning_rate,
                                     weight_decay=self.hyp_params.weight_decay)
        return optimizer


def main():
    if args.train_from_raw_data:
        base_dir = args.base_dir
        files_path = f"{base_dir}{args.files_path}"
        utt_names_path = f"{base_dir}{args.utt_names_path}"
        meta_path = f"{base_dir}{args.meta_path}"

        prepare_pickle_data(files_path, utt_names_path, args.utt_file, meta_path, args.data_path, args.dataset_name)

    train_set = Multimodal_Datasets(args.data_path, args.dataset_name, 'train')
    dev_set = Multimodal_Datasets(args.data_path, args.dataset_name, 'valid')
    test_set = Multimodal_Datasets(args.data_path, args.dataset_name, 'test')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=63)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=63)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=63)

    model_ = EmoModel(args)

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='f1_average_val', dirpath=args.checkpoints_dir,
                                                              mode="max",
                                                              filename='{epoch:02d}-{val_loss:.2f}--{f1_average_val:.2f}')
    trainer = Trainer(max_epochs=args.max_epochs, default_root_dir=args.checkpoints_dir, check_val_every_n_epoch=1,
                      num_sanity_val_steps=0, callbacks=[RichProgressBar(), checkpoint_callback,
                                                         EarlyStopping(monitor="f1_average_val", mode="max")])

    trainer.fit(model_, train_loader, dev_loader)
    trainer.test(dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
