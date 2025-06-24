import cv2
import os
import numpy as np
import random
from skimage.morphology import skeletonize
from skimage.feature import corner_harris, corner_peaks
from skimage.measure import label

# Function to convert full segmentation labels into sparse scribble annotations
def full2scribble(full_label):
    height, width, _ = full_label.shape
    full_label_BW = np.zeros((height, width), np.uint8)

    # Convert full label image to binary (black = background, anything else = road)
    for i in range(height):
        for j in range(width):
            if (full_label[i][j] == [0, 0, 0]).all():
                full_label_BW[i][j] = 0
            else:
                full_label_BW[i][j] = 255

    # Helper to remove corner pixels from a mask given detected coordinates
    def remove_corner(mask, coords):
        for x, y in coords:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1]:
                        mask[nx][ny] = 0
        return mask

    # Main scribblization process with thinning, corner removal, and partial label drop
    def scribblize(mask, ratio):
        sk = skeletonize(mask)  # Skeleton of the foreground mask
        i_mask = np.abs(mask - 1) // 255  # Inverse mask (background)
        i_sk = skeletonize(i_mask)  # Skeleton of background

        # Detect and remove corners to prevent noise in annotations
        try:
            corners = corner_peaks(corner_harris(i_sk), min_distance=10, threshold_rel=0.5)
            i_sk = remove_corner(i_sk, corners)
        except Exception as e:
            print(f"[WARN] Corner detection skipped due to: {e}")

        # Randomly remove some components in the foreground skeleton
        label_sk = label(sk)
        n_sk = np.max(label_sk)
        n_remove = int(n_sk * (1 - ratio))
        removes = random.sample(range(1, n_sk + 1), min(n_remove, n_sk))
        for i in removes:
            label_sk[label_sk == i] = 0
        sk = (label_sk > 0).astype('uint8')

        # Randomly remove extra components in the background to balance
        label_i_sk = label(i_sk)
        n_i_sk = np.max(label_i_sk)
        n_i_remove = max(0, n_i_sk - (n_sk - n_remove))
        removes = random.sample(range(1, n_i_sk + 1), min(n_i_remove, n_i_sk))
        for i in removes:
            label_i_sk[label_i_sk == i] = 0
        i_sk = (label_i_sk > 0).astype('uint8')

        return sk, i_sk

    # Convert RGB mask to binary labels: roads = 1, background = 0
    labels = np.zeros((height, width), np.uint16)
    labels[full_label_BW > 0] = 1
    mask = (labels > 0).astype('uint8')

    # Generate foreground and background scribbles
    sk, i_sk = scribblize(mask, ratio=1.0)

    # Initialize output scribble map: white = unlabeled, black = road, gray = background
    scr = np.ones_like(mask, dtype=np.uint8) * 255
    scr[i_sk == 1] = 255  # Background scribble
    scr[sk == 1] = 1      # Foreground (road) scribble

    return scr

# Main execution block to process all masks and save their corresponding scribble maps
if __name__ == '__main__':
    imageID = 0
    mask_dir = "./mask"           # Directory containing original mask images
    scribble_dir = "./scribble"   # Output directory for generated scribbles
    os.makedirs(scribble_dir, exist_ok=True)

    # Iterate through each mask image
    for file in os.listdir(mask_dir):
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        imageID += 1
        full_label_path = os.path.join(mask_dir, file)
        scribble_label_path = os.path.join(scribble_dir, file)

        # Read the full label image
        full_label = cv2.imread(full_label_path)
        if full_label is None:
            print(f"[ERROR] Failed to read image {file}")
            continue

        # Convert full label to scribble label
        scribble_label = full2scribble(full_label)

        # Save the scribble label image
        cv2.imwrite(scribble_label_path, scribble_label)
        print(f"[OK] Image {imageID}: {file} processed successfully.")
