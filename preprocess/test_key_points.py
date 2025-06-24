import numpy as np
import cv2
import scipy.io

# Load the .mat file that contains the keypoint information
key_points_info = scipy.io.loadmat("key_points_final/113_mask.mat")
if_key_points = key_points_info["if_key_points"]  # Binary map indicating if a keypoint exists at each 64×64 patch
all_key_points_position = key_points_info["all_key_points_position"]  # Coordinates of the keypoint for each patch

# Load the corresponding scribble image (grayscale)
file = "113_mask.png"
image = cv2.imread("scribble/" + file)
img = image[:, :, 0]  # Extract only one channel (assumes single-channel data)

# Initialize a new RGB image for visualization
new_img = np.zeros((1024, 1024, 3))

# Iterate through each 64×64 patch (16×16 pixels per patch)
for i in range(0, 64):
    for j in range(0, 64):
        for m in range(16 * i, 16 * i + 16):
            for n in range(16 * j, 16 * j + 16):
                new_img[m][n] = [255, 255, 255]  # Default background (white)
                
                if if_key_points[0, i, j] == 0:
                    new_img[m][n] = [0, 255, 255]  # Yellow if no keypoint in this patch
                
                if img[m][n] == 1:
                    new_img[m][n] = [0, 0, 0]  # Black for road pixels
                
                # Draw red dot at keypoint position in patch
                if (all_key_points_position[0, i, j] == m) and (all_key_points_position[1, i, j] == n):
                    new_img[m][n] = [0, 0, 255]  # Red if pixel is keypoint

# Save the visualized image
cv2.imwrite("113_mask-test.png", new_img)
print("New image generating finished!")

# This script checks if the keypoint info stored in the .mat file aligns with the road mask pixels visually.
