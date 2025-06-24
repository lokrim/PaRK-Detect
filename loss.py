import torch
import torch.nn as nn
from torch.autograd import Variable as V
import cv2
import numpy as np

# Combined Dice + Binary Cross Entropy (BCE) loss for segmentation
class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    # Computes soft Dice coefficient
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # smoothing factor to avoid division by zero

        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            # Compute Dice per sample, channel-wise
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)

        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    # Converts Dice coefficient into loss
    def soft_dice_loss(self, y_true, y_pred):
        return 1 - self.soft_dice_coeff(y_true, y_pred)

    # Returns sum of BCE loss and Dice loss
    def __call__(self, y_true, y_pred):
        bce = self.bce_loss(y_pred, y_true)
        dice = self.soft_dice_loss(y_true, y_pred)
        return bce + dice

# L1 loss that only considers valid (non-masked) positions
class partial_l1_loss(nn.Module):
    def __init__(self):
        super(partial_l1_loss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def __call__(self, trunk_posi_true, trunk_posi_pred):
        # Use predicted value only at positions where ground truth is valid
        trunk_posi_pseudo_pred = torch.where(trunk_posi_true == -1, trunk_posi_true, trunk_posi_pred)
        return self.l1_loss(trunk_posi_true, trunk_posi_pseudo_pred)

# BCE loss that ignores masked-out values (-1) in ground truth
class partial_bce_loss(nn.Module):
    def __init__(self):
        super(partial_bce_loss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def __call__(self, trunk_link_true, trunk_link_pred):
        # Replace ignored positions with zeros and compute BCE loss only for valid entries
        trunk_link_pseudo_pred = torch.where(trunk_link_true != -1, trunk_link_pred, torch.zeros_like(trunk_link_pred))
        trunk_link_pseudo_true = torch.where(trunk_link_true != -1, trunk_link_true, torch.zeros_like(trunk_link_true))
        return self.bce_loss(trunk_link_pseudo_pred, trunk_link_pseudo_true)
