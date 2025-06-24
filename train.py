import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np

from time import time

# Import model architecture
# from networks.unet import Unet
# from networks.dunet import Dunet
# from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from networks.dinknet import DinkNet34_WithBranch  # Using DinkNet34 with keypoint branches
from framework import MyFrame                    # Training wrapper for model/loss/optimizer
from loss import dice_bce_loss, partial_l1_loss, partial_bce_loss  # Custom loss functions
from data import ImageFolder                    # Custom dataset class

def main():
    SHAPE = (1024,1024)                          # Input image shape (not directly used in code)
    ROOT = 'dataset/train/'                      # Root directory for training data
    imagelist = filter(lambda x: x.find('sat')!=-1, os.listdir(ROOT))  # Select satellite image filenames
    trainlist = list(map(lambda x: x[:-8], imagelist))  # Remove '_sat.jpg' to get base ID
    NAME = 'log04_dink34'                        # Model/log file name prefix
    BATCHSIZE_PER_CARD = 4                       # Number of samples per GPU

    print(f"Train list length: {len(trainlist)}")

    # Initialize training framework with model and loss functions
    solver = MyFrame(DinkNet34_WithBranch, dice_bce_loss, partial_l1_loss, partial_bce_loss, 2e-4)

    # Total batch size = num GPUs × batch size per card
    batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

    # Initialize dataset and DataLoader
    dataset = ImageFolder(trainlist, ROOT)
    print("Dataset created")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0)
    print("DataLoader created")

    # Open log file
    mylog = open('logs/'+NAME+'.log','w')
    tic = time()                                 # Track start time
    no_optim = 0                                 # Epochs with no improvement
    total_epoch = 5                              # Total number of epochs to train
    train_epoch_best_loss = 100.                 # Initialize best loss as a high value

    try:
        for epoch in range(1, total_epoch + 1):
            print(f"Epoch {epoch} started")
            data_loader_iter = iter(data_loader)

            # Accumulate loss values for logging
            train_epoch_loss = 0
            train_epoch_loss_pred = 0
            train_epoch_loss_prob = 0
            train_epoch_loss_posi = 0
            train_epoch_loss_link = 0

            for batch_idx, (img, mask, if_key_points, all_key_points_position, anchor_link) in enumerate(data_loader_iter):
                # Log input shapes for sanity check
                print(f"Batch {batch_idx}: img {img.shape}, mask {mask.shape}, if_key_points {if_key_points.shape}, all_key_points_position {all_key_points_position.shape}, anchor_link {anchor_link.shape}")
                
                # Pass batch to model
                solver.set_input(img, mask, if_key_points, all_key_points_position, anchor_link)

                # Perform forward + backward pass and optimizer step
                train_loss, train_loss_pred, train_loss_prob, train_loss_posi, train_loss_link = solver.optimize()

                # Print batch loss
                print(f"Loss: {train_loss:.4f}, pred: {train_loss_pred:.4f}, prob: {train_loss_prob:.4f}, posi: {train_loss_posi:.4f}, link: {train_loss_link:.4f}")

                # Accumulate batch losses for epoch
                train_epoch_loss += train_loss
                train_epoch_loss_pred += train_loss_pred
                train_epoch_loss_prob += train_loss_prob
                train_epoch_loss_posi += train_loss_posi
                train_epoch_loss_link += train_loss_link

            print(f"Epoch {epoch} finished\n")

            # Average losses over batches in epoch
            train_epoch_loss /= len(data_loader_iter)
            train_epoch_loss_pred /= len(data_loader_iter)
            train_epoch_loss_prob /= len(data_loader_iter)
            train_epoch_loss_posi /= len(data_loader_iter)
            train_epoch_loss_link /= len(data_loader_iter)

            # Write epoch log
            print('********', file=mylog)
            print('epoch:', epoch, '    time:', int(time() - tic), file=mylog)
            print('train_loss:', train_epoch_loss, file=mylog)
            print('pred_loss:', train_epoch_loss_pred, file=mylog)
            print('prob_loss:', train_epoch_loss_prob, file=mylog)
            print('posi_loss:', train_epoch_loss_posi, file=mylog)
            print('link_loss:', train_epoch_loss_link, file=mylog)
            print('SHAPE:', SHAPE, file=mylog)

            # Also print log to stdout
            print('********')
            print('epoch:', epoch, '    time:', int(time() - tic))
            print('train_loss:', train_epoch_loss)
            print('pred_loss:', train_epoch_loss_pred)
            print('prob_loss:', train_epoch_loss_prob)
            print('posi_loss:', train_epoch_loss_posi)
            print('link_loss:', train_epoch_loss_link)
            print('SHAPE:', SHAPE)

            # Check for improvement
            if train_epoch_loss >= train_epoch_best_loss:
                no_optim += 1                    # No improvement
            else:
                no_optim = 0                     # Improvement
                train_epoch_best_loss = train_epoch_loss
                solver.save('weights/'+NAME+'.th')  # Save model checkpoint

            # Early stopping condition
            if no_optim > 6:
                print('early stop at %d epoch' % epoch, file=mylog)
                print('early stop at %d epoch' % epoch)
                break

            # If plateaued for a few epochs, reduce LR and reload best model
            if no_optim > 3:
                if solver.old_lr < 5e-7:
                    break
                solver.load('weights/'+NAME+'.th')
                solver.update_lr(5.0, factor = True, mylog = mylog)

            mylog.flush()  # Save log to disk
    except KeyboardInterrupt:
        # Gracefully handle Ctrl+C
        print("Training interrupted! Saving model checkpoint...")
        solver.save('weights/'+NAME+'_interrupt.th')
        print(f"Model saved to weights/{NAME}_interrupt.th")

    print('Finish!', file=mylog)
    print('Finish!')
    mylog.close()

# Entry point
if __name__ == "__main__":
    main()
