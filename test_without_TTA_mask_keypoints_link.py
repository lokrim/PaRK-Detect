import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import cv2
import os
import numpy as np
import scipy.io
from time import time
from networks.dinknet import DinkNet34_WithBranch

# Wrapper class for applying test-time inference with optional model ensembling
class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    # Forward pass for a single image (no TTA)
    def test_one_img_without_TTAFrame(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        img = cv2.imread(path)
        img = np.expand_dims(np.array(img), axis=0)
        img = img.transpose(0, 3, 1, 2)
        img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())

        mask, prob, posi, link = self.net.forward(img)

        return (
            mask.squeeze().cpu().data.numpy(),
            prob.squeeze(0).cpu().data.numpy(),
            posi.squeeze().cpu().data.numpy(),
            link.squeeze().cpu().data.numpy()
        )

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

# Set up paths and model
source = 'dataset/valid/sat/'
val = os.listdir(source)
solver = TTAFrame(DinkNet34_WithBranch)
solver.load('weights/log04_dink34.th')

# Output folders
target = 'submits/log04_dink34_without_TTA/'
os.makedirs(target + 'mask/')
os.makedirs(target + 'mat/')
os.makedirs(target + 'prob_posi_link/')
os.makedirs(target + 'merge/')

tic = time()
for img_id, name in enumerate(val):
    if img_id % 10 == 0:
        print(img_id / 10, '    ', '%.2f' % (time() - tic))

    mask, prob, posi, link = solver.test_one_img_without_TTAFrame(source + name)

    # Binarize outputs
    mask[mask > 0.1] = 255
    mask[mask <= 0.5] = 0
    prob[prob > 0.1] = 1
    prob[prob <= 0.5] = 0
    link[link > 0.1] = 1
    link[link <= 0.5] = 0

    # Decode keypoint coordinates from normalized 16x16 cell-based predictions
    posi_final = np.zeros((2, 64, 64), np.int64)
    for i in range(64):
        for j in range(64):
            if prob[0, i, j] == 1:
                posi_final[0, i, j] = int(posi[0, i, j] * 15 + 0.5) + i * 16
                posi_final[1, i, j] = int(posi[1, i, j] * 15 + 0.5) + j * 16

                # Remove invalid links to non-existent keypoints
                for d, (di, dj) in enumerate([(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]):
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < 64 and 0 <= nj < 64) or prob[0, ni, nj] != 1:
                        link[d, i, j] = 0

                # Add diagonal links if neighbor exists and straight neighbors are empty
                if i - 1 >= 0 and j + 1 < 64 and link[1, i, j] == 0:
                    if prob[0, i - 1, j + 1] == 1 and prob[0, i - 1, j] == 0 and prob[0, i, j + 1] == 0:
                        link[1, i, j] = 1
                if i + 1 < 64 and j + 1 < 64 and link[3, i, j] == 0:
                    if prob[0, i + 1, j + 1] == 1 and prob[0, i + 1, j] == 0 and prob[0, i, j + 1] == 0:
                        link[3, i, j] = 1
                if i + 1 < 64 and j - 1 >= 0 and link[5, i, j] == 0:
                    if prob[0, i + 1, j - 1] == 1 and prob[0, i + 1, j] == 0 and prob[0, i, j - 1] == 0:
                        link[5, i, j] = 1
                if i - 1 >= 0 and j - 1 >= 0 and link[7, i, j] == 0:
                    if prob[0, i - 1, j - 1] == 1 and prob[0, i - 1, j] == 0 and prob[0, i, j - 1] == 0:
                        link[7, i, j] = 1
            else:
                posi_final[:, i, j] = -1
                link[:, i, j] = -1

    # Remove cycles formed by 2x2 keypoints (RemoveCircle_Update equivalent)
    posi_cal = posi_final.astype(np.int64)
    for i in range(63):
        for j in range(63):
            if all(prob[0, ni, nj] == 1 for ni, nj in [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]):
                a = np.sum((posi_cal[:, i, j] - posi_cal[:, i + 1, j]) ** 2)
                b = np.sum((posi_cal[:, i + 1, j] - posi_cal[:, i + 1, j + 1]) ** 2)
                c = np.sum((posi_cal[:, i, j] - posi_cal[:, i, j + 1]) ** 2)
                d = np.sum((posi_cal[:, i, j + 1] - posi_cal[:, i + 1, j + 1]) ** 2)
                max_dist = max(a, b, c, d)
                if a == max_dist:
                    link[4, i, j] = link[0, i + 1, j] = 0
                elif b == max_dist:
                    link[2, i + 1, j] = link[6, i + 1, j + 1] = 0
                elif c == max_dist:
                    link[2, i, j] = link[6, i, j + 1] = 0
                elif d == max_dist:
                    link[4, i, j + 1] = link[0, i + 1, j + 1] = 0

    # Save mask as RGB image
    mask_rgb = np.concatenate([mask[:, :, None]] * 3, axis=2)
    cv2.imwrite(target + 'mask/' + name[:-7] + 'mask.png', mask_rgb.astype(np.uint8))

    # Save position and link matrices as .mat file
    scipy.io.savemat(
        target + 'mat/' + name[:-7] + 'mask.mat',
        mdict={'if_key_points': prob, 'all_key_points_position': posi_final, 'anchor_link': link}
    )

    # Generate overlay visualization of anchors and links
    new_img = np.zeros((1024, 1024, 3), np.uint8)
    for i in range(64):
        for j in range(64):
            if prob[0, i, j] == 0:
                new_img[i*16:(i+1)*16, j*16:(j+1)*16] = [0, 255, 255]
            else:
                new_img[i*16:(i+1)*16, j*16:(j+1)*16] = [255, 255, 255]
                # draw links
                for d, (di, dj) in enumerate([(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 64 and 0 <= nj < 64 and link[d, i, j] == 1 and prob[0, ni, nj] == 1:
                        pt1 = (posi_final[1, i, j], posi_final[0, i, j])
                        pt2 = (posi_final[1, ni, nj], posi_final[0, ni, nj])
                        cv2.line(new_img, pt1, pt2, (0, 255, 0), 1)

    # draw keypoints
    for i in range(64):
        for j in range(64):
            if prob[0, i, j] == 1:
                m, n = posi_final[0, i, j], posi_final[1, i, j]
                new_img[m, n] = [0, 0, 255]

    cv2.imwrite(target + 'prob_posi_link/' + name[:-7] + 'prob_posi_link.png', new_img)

    # Merge satellite image with link visualization
    sat = cv2.imread(source + name)
    sat_merge = cv2.addWeighted(sat, 0.8, new_img, 0.2, 0)
    cv2.imwrite(target + 'merge/' + name[:-7] + 'merge.png', sat_merge)
