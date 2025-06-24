import numpy as np
import os
import cv2
import scipy.io

imageID = 0
# Traverse all images in the 'scribble' directory
for root, dirs, files in os.walk("scribble"):
    for file in files:
        imageID = imageID + 1

        '''
        # Uncomment this block to process only a specific image for debugging
        if file != 'img-14_1_0.png':
            continue
        '''

        # Read image and extract the first channel as a binary road mask
        image = cv2.imread("scribble/" + file)
        img = image[:,:,0]

        # Global counters (optional, for analysis/stats)
        count_up3 = 0  # Road pixels with degree >= 3 (potential intersections)
        count_2 = 0    # Degree == 2 (normal road points)
        count_1 = 0    # Degree == 1 (endpoints)

        if_key_points = []            # Binary flag for whether keypoint exists in a patch
        all_key_points_position = []  # Stores keypoint coordinates for each patch

        # Loop over 64x64 patches (total 4096)
        for i in range(0, 64):
            for j in range(0, 64):
                if_exist_road_node = 0     # Flag: whether there is any road pixel in the patch
                count_up3_anchorij = 0     # Count of degree >=3 pixels in the current patch
                count_2_anchorij = 0       # Count of degree 2
                count_1_anchorij = 0       # Count of degree 1

                up3_node_list = []         # Store positions of degree >= 3 nodes
                d2_node_list = []          # Store positions of degree 2 nodes
                d1_node_list = []          # Store positions of degree 1 nodes

                # Traverse each 16x16 sub-region of the patch
                for m in range(16*i,16*i+16):
                    for n in range(16*j,16*j+16):
                        if img[m][n] == 1:  # If road pixel
                            if_exist_road_node = 1
                            d = 0  # Degree counter

                            # Check all 8 neighbors and increment degree if neighbor is a road pixel
                            if (m-1>=0) and (m-1<1024) and (n-1>=0) and (n-1<1024):
                                if img[m-1][n-1] == 1:
                                    d += 1
                            if (m-1>=0) and (m-1<1024) and (n>=0) and (n<1024):
                                if img[m-1][n] == 1:
                                    d += 1
                            if (m-1>=0) and (m-1<1024) and (n+1>=0) and (n+1<1024):
                                if img[m-1][n+1] == 1:
                                    d += 1
                            if (m>=0) and (m<1024) and (n-1>=0) and (n-1<1024):
                                if img[m][n-1] == 1:
                                    d += 1
                            if (m>=0) and (m<1024) and (n+1>=0) and (n+1<1024):
                                if img[m][n+1] == 1:
                                    d += 1
                            if (m+1>=0) and (m+1<1024) and (n-1>=0) and (n-1<1024):
                                if img[m+1][n-1] == 1:
                                    d += 1
                            if (m+1>=0) and (m+1<1024) and (n>=0) and (n<1024):
                                if img[m+1][n] == 1:
                                    d += 1
                            if (m+1>=0) and (m+1<1024) and (n+1>=0) and (n+1<1024):
                                if img[m+1][n+1] == 1:
                                    d += 1

                            # Based on degree, categorize and store coordinates
                            if d == 1:
                                count_1 += 1
                                count_1_anchorij += 1
                                d1_node_list.append([m,n])
                            elif d == 2:
                                count_2 += 1
                                count_2_anchorij += 1
                                d2_node_list.append([m,n])
                            elif d >= 3:
                                count_up3 += 1
                                count_up3_anchorij += 1
                                up3_node_list.append([m,n])

                # If no valid road point found
                if (count_1_anchorij == 0) and (count_2_anchorij == 0) and (count_up3_anchorij == 0):
                    if_exist_road_node = 0

                key_point = []

                # If patch has no road points
                if if_exist_road_node == 0:
                    key_point = [-1, -1]

                # If it has road points, classify the keypoint
                elif if_exist_road_node == 1:
                    if_exist_intersection = 0
                    if_exist_endpoint = 0

                    # Look for intersection patterns using handcrafted rules
                    if count_up3_anchorij >= 1:
                        for position in up3_node_list:
                            if ( 
                                ((img[position[0]][position[1]-1]==1) and (img[position[0]][position[1]+1]==1) and (img[position[0]-1][position[1]]==1)) or
                                ((img[position[0]][position[1]-1]==1) and (img[position[0]][position[1]+1]==1) and (img[position[0]+1][position[1]]==1)) or
                                ((img[position[0]][position[1]-1]==1) and (img[position[0]-1][position[1]]==1) and (img[position[0]+1][position[1]]==1)) or
                                ((img[position[0]][position[1]+1]==1) and (img[position[0]-1][position[1]]==1) and (img[position[0]+1][position[1]]==1)) or
                                ((img[position[0]][position[1]-1]==1) and (img[position[0]-1][position[1]]==1) and (img[position[0]+1][position[1]+1]==1)) or
                                ((img[position[0]][position[1]+1]==1) and (img[position[0]+1][position[1]]==1) and (img[position[0]-1][position[1]-1]==1)) or
                                ((img[position[0]][position[1]-1]==1) and (img[position[0]+1][position[1]]==1) and (img[position[0]-1][position[1]+1]==1)) or
                                ((img[position[0]][position[1]+1]==1) and (img[position[0]-1][position[1]]==1) and (img[position[0]+1][position[1]-1]==1)) or
                                ((img[position[0]-1][position[1]-1]==1) and (img[position[0]+1][position[1]-1]==1) and (img[position[0]][position[1]+1]==1) and (img[position[0]][position[1]-1]!=1)) or
                                ((img[position[0]-1][position[1]-1]==1) and (img[position[0]-1][position[1]+1]==1) and (img[position[0]+1][position[1]]==1) and (img[position[0]-1][position[1]]!=1)) or
                                ((img[position[0]-1][position[1]+1]==1) and (img[position[0]+1][position[1]+1]==1) and (img[position[0]][position[1]-1]==1) and (img[position[0]][position[1]+1]!=1)) or
                                ((img[position[0]+1][position[1]-1]==1) and (img[position[0]+1][position[1]+1]==1) and (img[position[0]-1][position[1]]==1) and (img[position[0]+1][position[1]]!=1)) or
                                ((img[position[0]-1][position[1]-1]==1) and (img[position[0]+1][position[1]-1]==1) and (img[position[0]-1][position[1]+1]==1) and (img[position[0]][position[1]-1]!=1) and (img[position[0]-1][position[1]]!=1)) or
                                ((img[position[0]-1][position[1]-1]==1) and (img[position[0]+1][position[1]-1]==1) and (img[position[0]+1][position[1]+1]==1) and (img[position[0]][position[1]-1]!=1) and (img[position[0]+1][position[1]]!=1)) or
                                ((img[position[0]-1][position[1]-1]==1) and (img[position[0]-1][position[1]+1]==1) and (img[position[0]+1][position[1]+1]==1) and (img[position[0]][position[1]+1]!=1) and (img[position[0]-1][position[1]]!=1)) or
                                ((img[position[0]-1][position[1]+1]==1) and (img[position[0]+1][position[1]-1]==1) and (img[position[0]+1][position[1]+1]==1) and (img[position[0]][position[1]+1]!=1) and (img[position[0]+1][position[1]]!=1))
                            ):
                                if_exist_intersection = 1
                                key_point = [position[0], position[1]]
                                break

                    # If no intersection, but endpoints exist
                    if (count_1_anchorij >= 1) and (if_exist_intersection == 0):
                        if_exist_endpoint = 1
                        key_point = [d1_node_list[0][0], d1_node_list[0][1]]

                    # If no intersections or endpoints, fallback to averaging degree-2 points
                    if (count_2_anchorij >= 1) and (if_exist_intersection == 0) and (if_exist_endpoint == 0):
                        sum_x_position = 0
                        sum_y_position = 0 
                        for position in d2_node_list:
                            sum_x_position += position[0]
                            sum_y_position += position[1]
                        mean_x_position = sum_x_position // len(d2_node_list)
                        mean_y_position = sum_y_position // len(d2_node_list)

                        # Use nearest road point if center point is not a road
                        if img[mean_x_position][mean_y_position] == 1:
                            key_point = [mean_x_position, mean_y_position]
                        else:
                            closest_distance_square = 1000
                            closest_point = []
                            for position in d2_node_list:
                                distance_square = (position[0]-mean_x_position)**2 + (position[1]-mean_y_position)**2
                                if distance_square < closest_distance_square:
                                    closest_distance_square = distance_square
                                    closest_point = [position[0], position[1]]
                            key_point = [closest_point[0], closest_point[1]]

                # Final fallback
                if key_point == []:
                    key_point = [-1, -1]
                    if_exist_road_node = 0

                if_key_points.append(if_exist_road_node)
                all_key_points_position.append(key_point)

        # Convert lists to numpy arrays
        if_key_points_array = np.array(if_key_points)
        all_key_points_position_array = np.array(all_key_points_position)

        # Check shape validity before saving
        if (if_key_points_array.shape[0] != 4096) or (all_key_points_position_array.shape[0] != 4096) or (all_key_points_position_array.shape[1] != 2):
            print("Image " + file + " has something wrong!")
            break

        # Save output to .mat file
        mat_savepath = "key_points/" + file[:-4] + ".mat"
        scipy.io.savemat(mat_savepath, mdict={'if_key_points': if_key_points, 'all_key_points_position':all_key_points_position})
        print("Image " + str(imageID) + ": Finished!")
