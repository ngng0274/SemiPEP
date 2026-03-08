import os
import torch
import numpy as np
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryMatthewsCorrCoef, BinaryAveragePrecision, BinaryAUROC, BinaryF1Score

def evalution_prot(preds, targets, device):
    # AUROC
    auroc = BinaryAUROC().to(device)
    auroc.update(preds, targets)
    auroc_i = (auroc.compute()).item()
    # AUPRC
    auprc = BinaryAveragePrecision().to(device)
    auprc.update(preds, targets)
    auprc_i = (auprc.compute()).item()
    # Precision
    precision = BinaryPrecision().to(device)
    precision_i = precision(preds, targets).item()
    # Recall
    recall = BinaryRecall().to(device)
    recall_i = recall(preds, targets).item()
    # MCC
    mcc = BinaryMatthewsCorrCoef().to(device)
    mcc_i = mcc(preds, targets).item()
    # F1
    f1 = BinaryF1Score().to(device)
    f1_i = f1(preds, targets).item()

    return auprc_i, auroc_i, precision_i, recall_i, mcc_i, f1_i