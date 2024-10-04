# metrics.py

import torch

def confusion_matrix(preds, labels):
    preds = (preds > 0).float()
    labels = (labels > 0).float()
    preds = preds.view(-1)
    labels = labels.view(-1)
    TP = ((preds == 1) & (labels == 1)).sum().float()
    TN = ((preds == 0) & (labels == 0)).sum().float()
    FP = ((preds == 1) & (labels == 0)).sum().float()
    FN = ((preds == 0) & (labels == 1)).sum().float()
    return TP, TN, FP, FN

def compute_metrics(preds, labels):
    TP, TN, FP, FN = confusion_matrix(preds, labels)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-7)
    sensitivity = TP / (TP + FN + 1e-7)  # Recall
    specificity = TN / (TN + FP + 1e-7)
    precision = TP / (TP + FP + 1e-7)
    f1_score = 2 * precision * sensitivity / (precision + sensitivity + 1e-7)
    iou = TP / (TP + FP + FN + 1e-7)
    ''''
    smooth = 1
    size = preds.size(0)
    pred_ = preds.reshape(size, -1)
    target_ = labels.reshape(size, -1)
    intersection = pred_ * target_
    dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
    '''''
    return {
        'accuracy': accuracy.item(),
        'sensitivity': sensitivity.item(),
        'specificity': specificity.item(),
        'f1_score': f1_score.item(),
        #'dice_score': dice_score,
        'iou': iou.item(),
    }
