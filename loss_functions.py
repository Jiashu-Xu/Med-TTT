# loss_functions.py

import torch
import torch.nn as nn

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.reshape(size, -1)
        target_ = target.reshape(size, -1)
        return self.bceloss(pred_, target_)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)
        pred_ = pred.reshape(size, -1)
        target_ = target.reshape(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size
        return dice_loss

class BceDiceLoss(nn.Module):
    def __init__(self, wb=0.8, wd=1.2):
        super(BceDiceLoss, self).__init__()
        #self.bce = nn.BCEWithLogitsLoss()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        #print("pred",pred.shape)
        #print("target",target.shape)
        pred = pred.squeeze(1)
        bceloss = self.bce(torch.sigmoid(pred), target)
        diceloss = self.dice(torch.sigmoid(pred), target)
        loss = self.wd * diceloss + self.wb * bceloss
        return loss
