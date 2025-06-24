import torch
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import numpy as np
import os
import scipy.io

# Randomly adjust hue, saturation, and value of the image
def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image

# Randomly apply horizontal flip to image, mask, and annotations
def randomHorizontalFlip(image, mask, if_key_points, all_key_points_position, anchor_link, u=0.5):
    new_anchor_link = np.zeros((8,64,64))
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        if_key_points = np.flip(if_key_points, 2)
        all_key_points_position = np.flip(all_key_points_position, 2)
        all_key_points_position[1,:,:] = 1023 - all_key_points_position[1,:,:]
        all_key_points_position[all_key_points_position == 1024] = -1
        anchor_link = np.flip(anchor_link, 2)
        new_anchor_link[0,:,:] = anchor_link[0,:,:]
        new_anchor_link[1,:,:] = anchor_link[7,:,:]
        new_anchor_link[2,:,:] = anchor_link[6,:,:]
        new_anchor_link[3,:,:] = anchor_link[5,:,:]
        new_anchor_link[4,:,:] = anchor_link[4,:,:]
        new_anchor_link[5,:,:] = anchor_link[3,:,:]
        new_anchor_link[6,:,:] = anchor_link[2,:,:]
        new_anchor_link[7,:,:] = anchor_link[1,:,:]
    else:
        new_anchor_link = anchor_link
    return image, mask, if_key_points, all_key_points_position, new_anchor_link

# Randomly apply vertical flip to image, mask, and annotations
def randomVerticleFlip(image, mask, if_key_points, all_key_points_position, anchor_link, u=0.5):
    new_anchor_link = np.zeros((8,64,64))
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
        if_key_points = np.flip(if_key_points, 1)
        all_key_points_position = np.flip(all_key_points_position, 1)
        all_key_points_position[0,:,:] = 1023 - all_key_points_position[0,:,:]
        all_key_points_position[all_key_points_position == 1024] = -1
        anchor_link = np.flip(anchor_link, 1)
        new_anchor_link[0,:,:] = anchor_link[4,:,:]
        new_anchor_link[1,:,:] = anchor_link[3,:,:]
        new_anchor_link[2,:,:] = anchor_link[2,:,:]
        new_anchor_link[3,:,:] = anchor_link[1,:,:]
        new_anchor_link[4,:,:] = anchor_link[0,:,:]
        new_anchor_link[5,:,:] = anchor_link[7,:,:]
        new_anchor_link[6,:,:] = anchor_link[6,:,:]
        new_anchor_link[7,:,:] = anchor_link[5,:,:]
    else:
        new_anchor_link = anchor_link
    return image, mask, if_key_points, all_key_points_position, new_anchor_link

# Randomly rotate image and annotation by 90 degrees
def randomRotate90(image, mask, if_key_points, all_key_points_position, anchor_link, u=0.5):
    new_all_key_points_position = np.zeros((2,64,64))
    new_anchor_link = np.zeros((8,64,64))
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

        if_key_points = if_key_points.transpose(1,2,0)
        if_key_points = np.rot90(if_key_points)
        if_key_points = if_key_points.transpose(2,0,1)

        all_key_points_position = all_key_points_position.transpose(1,2,0)
        all_key_points_position = np.rot90(all_key_points_position)
        all_key_points_position = all_key_points_position.transpose(2,0,1)
        new_all_key_points_position[0,:,:] = 1023 - all_key_points_position[1,:,:]
        new_all_key_points_position[1,:,:] = all_key_points_position[0,:,:]
        new_all_key_points_position[new_all_key_points_position == 1024] = -1

        anchor_link = anchor_link.transpose(1,2,0)
        anchor_link = np.rot90(anchor_link)
        anchor_link = anchor_link.transpose(2,0,1)
        new_anchor_link[0,:,:] = anchor_link[2,:,:]
        new_anchor_link[1,:,:] = anchor_link[3,:,:]
        new_anchor_link[2,:,:] = anchor_link[4,:,:]
        new_anchor_link[3,:,:] = anchor_link[5,:,:]
        new_anchor_link[4,:,:] = anchor_link[6,:,:]
        new_anchor_link[5,:,:] = anchor_link[7,:,:]
        new_anchor_link[6,:,:] = anchor_link[0,:,:]
        new_anchor_link[7,:,:] = anchor_link[1,:,:]
    else:
        new_all_key_points_position = all_key_points_position
        new_anchor_link = anchor_link
    return image, mask, if_key_points, new_all_key_points_position, new_anchor_link

# Load image, mask, and annotation data from disk and apply augmentation
def default_loader(id, root):
    img = cv2.imread(os.path.join(root, f'{id}_sat.jpg'))
    mask = cv2.imread(os.path.join(root, f'{id}_mask.png'), cv2.IMREAD_GRAYSCALE)
    key_points = scipy.io.loadmat(os.path.join(root, f'{id}_mask.mat'))
    
    if_key_points = key_points["if_key_points"]
    all_key_points_position = key_points["all_key_points_position"]
    anchor_link = key_points["anchor_link"]

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask, if_key_points, all_key_points_position, anchor_link = randomHorizontalFlip(img, mask, if_key_points, all_key_points_position, anchor_link)
    img, mask, if_key_points, all_key_points_position, anchor_link = randomVerticleFlip(img, mask, if_key_points, all_key_points_position, anchor_link)
    img, mask, if_key_points, all_key_points_position, anchor_link = randomRotate90(img, mask, if_key_points, all_key_points_position, anchor_link)
    
    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
    mask[mask>=0.5] = 1
    mask[mask<=0.5] = 0

    # Normalize keypoint positions to range [0, 1]
    new_all_key_points_position = all_key_points_position % 16
    new_all_key_points_position[all_key_points_position == -1] = -16
    new_all_key_points_position = new_all_key_points_position.astype(np.float64) / 16

    return img, mask, if_key_points, new_all_key_points_position, anchor_link

# Custom PyTorch Dataset class
class ImageFolder(data.Dataset):

    def __init__(self, trainlist, root):
        self.ids = trainlist
        self.loader = default_loader
        self.root = root

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask, if_key_points, all_key_points_position, anchor_link = self.loader(id, self.root)

        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        if_key_points = torch.Tensor(np.ascontiguousarray(if_key_points))
        all_key_points_position = torch.Tensor(np.ascontiguousarray(all_key_points_position))
        anchor_link = torch.Tensor(np.ascontiguousarray(anchor_link))

        return img, mask, if_key_points, all_key_points_position, anchor_link

    def __len__(self):
        return len(self.ids)
