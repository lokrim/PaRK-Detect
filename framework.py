import torch
import torch.nn as nn
from torch.autograd import Variable as V

import cv2
import numpy as np

class MyFrame():
    def __init__(self, net, dice_bce_loss, partial_l1_loss, partial_bce_loss, lr=2e-4, evalmode=False):
        # Initialize model, wrap with DataParallel for multi-GPU training
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        
        # Optimizer setup
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)

        # Loss functions for segmentation and auxiliary outputs
        self.dice_bce_loss = dice_bce_loss()
        self.partial_l1_loss = partial_l1_loss()
        self.partial_bce_loss = partial_bce_loss()

        self.old_lr = lr

        # If in evaluation mode, set all BatchNorm layers to eval
        if evalmode:
            for module in self.net.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

    def set_input(self, img_batch, mask_batch=None, if_key_points=None, all_key_points_position=None, anchor_link=None, img_id=None):
        # Load input data and annotations into the object
        self.img = img_batch
        self.mask = mask_batch
        self.if_key_points = if_key_points
        self.all_key_points_position = all_key_points_position
        self.anchor_link = anchor_link
        self.img_id = img_id

    def forward(self, volatile=False):
        # Move inputs to GPU and wrap with Variable (for older PyTorch versions)
        self.img = V(self.img.cuda(), volatile=volatile)

        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)
        if self.if_key_points is not None:
            self.if_key_points = V(self.if_key_points.cuda(), volatile=volatile)
        if self.all_key_points_position is not None:
            self.all_key_points_position = V(self.all_key_points_position.cuda(), volatile=volatile)
        if self.anchor_link is not None:
            self.anchor_link = V(self.anchor_link.cuda(), volatile=volatile)

    def optimize(self):
        # Perform forward pass and compute losses
        self.forward()
        self.optimizer.zero_grad()

        # Forward through network to get predictions
        pred, trunk_prob, trunk_posi, trunk_link = self.net.forward(self.img)

        # Compute individual loss components
        loss_pred = self.dice_bce_loss(self.mask, pred)                          # Main segmentation loss
        loss_prob = self.dice_bce_loss(self.if_key_points, trunk_prob)          # Keypoint presence classification
        loss_posi = 14.5 * self.partial_l1_loss(self.all_key_points_position, trunk_posi)  # Keypoint position regression
        loss_link = 14.5 * self.partial_bce_loss(self.anchor_link, trunk_link)  # Link map prediction

        # Weighted sum of losses (experimentally adjustable)
        loss = loss_pred + 0.5 * (loss_prob + loss_posi + loss_link)

        loss.backward()
        self.optimizer.step()

        return loss.item(), loss_pred.item(), loss_prob.item(), loss_posi.item(), loss_link.item()

    def save(self, path):
        # Save model weights to a file
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        # Load model weights from a file
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, mylog, factor=False):
        # Update learning rate (absolute or as a divisor)
        if factor:
            new_lr = self.old_lr / new_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        # Log and print learning rate change
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr), file=mylog)
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))

        self.old_lr = new_lr
