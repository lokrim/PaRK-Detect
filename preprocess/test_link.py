import numpy as np
import cv2
import scipy.io

file = "113_mask"  # Base filename (without extension)

# Load keypoint and link data from .mat file
key_points_info = scipy.io.loadmat("link_key_points_final/" + file + ".mat")
if_key_points = key_points_info["if_key_points"]  # 1x64x64 binary map indicating presence of keypoints
all_key_points_position = key_points_info["all_key_points_position"]  # 2x64x64 coordinates of keypoints
anchor_link = key_points_info["anchor_link"]  # 8x64x64 binary tensor indicating link directions

# Load scribble mask image (single channel)
label = cv2.imread("scribble/" + file + ".png")
img = label[:, :, 0]  # Get grayscale image
new_img = np.zeros((1024, 1024, 3))  # Create blank RGB canvas

# Color patches based on keypoint existence and scribble mask
for i in range(0, 64):
    for j in range(0, 64):
        for m in range(16 * i, 16 * i + 16):
            for n in range(16 * j, 16 * j + 16):
                new_img[m][n] = [255, 255, 255]  # Default white
                if if_key_points[0, i, j] == 0:
                    new_img[m][n] = [0, 255, 255]  # Yellow if no keypoint
                if img[m][n] == 1:
                    new_img[m][n] = [0, 0, 0]  # Black for road pixels
                if (all_key_points_position[0, i, j] == m) and (all_key_points_position[1, i, j] == n):
                    new_img[m][n] = [0, 0, 255]  # Red dot for keypoint

# Save keypoints visualization
cv2.imwrite(file + "-keypoints.png", new_img)
print("Mat Keypoints generating finished!")

# Create another blank image to visualize links between keypoints
test_img = np.zeros((1024, 1024, 3), np.uint8)
for i in range(0, 64):
    for j in range(0, 64):
        for m in range(16 * i, 16 * i + 16):
            for n in range(16 * j, 16 * j + 16):
                test_img[m][n] = [255, 255, 255]  # Default white
                if if_key_points[0, i, j] == 0:
                    test_img[m][n] = [0, 255, 255]  # Yellow if no keypoint

# Draw lines based on anchor_link tensor for 8-connected neighborhood
for i in range(0, 64):
    for j in range(0, 64):
        if anchor_link[0, i, j] == 1 and i - 1 >= 0:  # Top
            cv2.line(test_img,
                     (all_key_points_position[1, i, j], all_key_points_position[0, i, j]),
                     (all_key_points_position[1, i - 1, j], all_key_points_position[0, i - 1, j]),
                     (0, 255, 0), 1)

        if anchor_link[1, i, j] == 1 and i - 1 >= 0 and j + 1 < 64:  # Top-Right
            cv2.line(test_img,
                     (all_key_points_position[1, i, j], all_key_points_position[0, i, j]),
                     (all_key_points_position[1, i - 1, j + 1], all_key_points_position[0, i - 1, j + 1]),
                     (0, 255, 0), 1)

        if anchor_link[2, i, j] == 1 and j + 1 < 64:  # Right
            cv2.line(test_img,
                     (all_key_points_position[1, i, j], all_key_points_position[0, i, j]),
                     (all_key_points_position[1, i, j + 1], all_key_points_position[0, i, j + 1]),
                     (0, 255, 0), 1)

        if anchor_link[3, i, j] == 1 and i + 1 < 64 and j + 1 < 64:  # Bottom-Right
            cv2.line(test_img,
                     (all_key_points_position[1, i, j], all_key_points_position[0, i, j]),
                     (all_key_points_position[1, i + 1, j + 1], all_key_points_position[0, i + 1, j + 1]),
                     (0, 255, 0), 1)

        if anchor_link[4, i, j] == 1 and i + 1 < 64:  # Bottom
            cv2.line(test_img,
                     (all_key_points_position[1, i, j], all_key_points_position[0, i, j]),
                     (all_key_points_position[1, i + 1, j], all_key_points_position[0, i + 1, j]),
                     (0, 255, 0), 1)

        if anchor_link[5, i, j] == 1 and i + 1 < 64 and j - 1 >= 0:  # Bottom-Left
            cv2.line(test_img,
                     (all_key_points_position[1, i, j], all_key_points_position[0, i, j]),
                     (all_key_points_position[1, i + 1, j - 1], all_key_points_position[0, i + 1, j - 1]),
                     (0, 255, 0), 1)

        if anchor_link[6, i, j] == 1 and j - 1 >= 0:  # Left
            cv2.line(test_img,
                     (all_key_points_position[1, i, j], all_key_points_position[0, i, j]),
                     (all_key_points_position[1, i, j - 1], all_key_points_position[0, i, j - 1]),
                     (0, 255, 0), 1)

        if anchor_link[7, i, j] == 1 and i - 1 >= 0 and j - 1 >= 0:  # Top-Left
            cv2.line(test_img,
                     (all_key_points_position[1, i, j], all_key_points_position[0, i, j]),
                     (all_key_points_position[1, i - 1, j - 1], all_key_points_position[0, i - 1, j - 1]),
                     (0, 255, 0), 1)

# Re-draw keypoints as red dots after the lines
for i in range(0, 64):
    for j in range(0, 64):
        for m in range(16 * i, 16 * i + 16):
            for n in range(16 * j, 16 * j + 16):
                if (all_key_points_position[0, i, j] == m) and (all_key_points_position[1, i, j] == n):
                    test_img[m][n] = [0, 0, 255]  # Red dot for keypoint

# Save the link visualization
cv2.imwrite(file + "-link.png", test_img)
print("Mat link generating finished!")
