import numpy as np
import os
import scipy.io

imageID = 0
# Traverse all .mat files in the 'key_points' directory
for root, dirs, files in os.walk("key_points"):
    for file in files:
        imageID = imageID + 1
        
        # Load the .mat file containing keypoint flags and positions
        key_points_info = scipy.io.loadmat("key_points/" + file)
        if_key_points = key_points_info["if_key_points"]  # Shape: (4096,)
        all_key_points_position = key_points_info["all_key_points_position"]  # Shape: (4096, 2)

        # print(if_key_points.shape)
        # print(all_key_points_position.shape)

        # Reshape keypoint existence array to (1, 64, 64)
        if_key_points = if_key_points.reshape((1, 64, 64))

        # Transpose to switch (4096, 2) => (2, 4096), then reshape to (2, 64, 64)
        all_key_points_position = all_key_points_position.transpose(1, 0)
        all_key_points_position = all_key_points_position.reshape((2, 64, 64))

        # print(if_key_points.shape)
        # print(all_key_points_position.shape)

        # Save reformatted arrays to new .mat file
        final_mat_savepath = "key_points_final/" + file
        scipy.io.savemat(final_mat_savepath, mdict={'if_key_points': if_key_points, 'all_key_points_position': all_key_points_position})
        print("Image " + str(imageID) + ": Finished!")

