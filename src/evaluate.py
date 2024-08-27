# This source code is licensed under CC BY 4.0 license.
# Copyright 2024 Samsung Electronics Co., Ltd.
from copy import deepcopy

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score


def make_stat(prev, curr):
    new_stats = []
    for i in range(len(prev)):
        if curr[i] > prev[i]:
            new_stats.append(f'{curr[i]:.4f} \u2191')
        elif curr[i] < prev[i]:
            new_stats.append(f'{curr[i]:.4f} \u2193')
        else:
            new_stats.append(f'{curr[i]:.4f} -')
    return new_stats


def eval_emo(preds, truths, best_thresholds=None):
    '''
    preds: (bs, num_emotions)
    truths: (bs, num_emotions)
    '''

    num_emo = preds.size(1)

    preds = preds.cpu().detach()
    truths = truths.cpu().detach()

    preds = torch.sigmoid(preds)

    aucs = roc_auc_score(truths, preds, labels=list(range(num_emo)), average=None).tolist()
    aucs.append(np.average(aucs))

    if best_thresholds is None:
        # select the best threshold for each emotion category, based on F1 score
        thresholds = np.arange(0.005, 1, 0.005)
        _f1s = []
        for t in thresholds:
            _preds = deepcopy(preds)
            _preds[_preds > t] = 1
            _preds[_preds <= t] = 0

            this_f1s = []

            for i in range(num_emo):
                pred_i = _preds[:, i]
                truth_i = truths[:, i]
                this_f1s.append(f1_score(truth_i, pred_i))

            _f1s.append(this_f1s)
        _f1s = np.array(_f1s)
        best_thresholds = (np.argmax(_f1s, axis=0) + 1) * 0.005

    for i in range(num_emo):
        pred = preds[:, i]
        pred[pred > best_thresholds[i]] = 1
        pred[pred <= best_thresholds[i]] = 0
        preds[:, i] = pred

    accs = []
    recalls = []
    precisions = []
    f1s = []
    for i in range(num_emo):
        pred_i = preds[:, i]
        truth_i = truths[:, i]

        acc = accuracy_score(truth_i, pred_i)
        recall = recall_score(truth_i, pred_i)
        precision = precision_score(truth_i, pred_i)
        f1 = f1_score(truth_i, pred_i)

        accs.append(acc)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    # It is not weighted!
    accs.append(np.average(accs))
    recalls.append(np.average(recalls))
    precisions.append(np.average(precisions))

    # it is weighting of f1s:

    ef1 = np.dot(f1s, truths.sum_to_size(truths.shape[1]) / truths.sum())
    f1s.append(ef1)

    return (accs, recalls, precisions, f1s, aucs), best_thresholds
