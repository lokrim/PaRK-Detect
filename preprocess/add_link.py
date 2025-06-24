import numpy as np
import cv2
import scipy.io
import os

# Recursive function to search for neighboring keypoints and update the anchor_link array
def search_link(img, search_point_list, key_points_gt_list, point_searched_list, key_point_searched_list, i, j, anchor_link, iter_count):
    iter_count = iter_count + 1
    if search_point_list == [] or iter_count >= 100:
        return "Search Finished"
    else:
        next_search_list = []
        for search_point in search_point_list:
            if search_point not in point_searched_list:
                point_searched_list.append(search_point)
                normal_point_list = []   # Collects intermediate (non-keypoint) pixels
                key_points_count = 0     # Counts connected keypoints

                # Check 8-connected neighbors
                # ↑
                if (search_point[0]-1>=0) and (search_point[0]-1<1024) and (search_point[1]>=0) and (search_point[1]<1024):
                    if img[search_point[0]-1][search_point[1]]==1:
                        if [search_point[0]-1,search_point[1]] in key_points_gt_list:
                            key_points_count += 1
                            if [search_point[0]-1,search_point[1]] not in key_point_searched_list:
                                key_point_searched_list.append([search_point[0]-1,search_point[1]])
                                anchor_link[key_points_gt_list.index([search_point[0]-1,search_point[1]]),i,j] = 1
                        else:
                            normal_point_list.append([search_point[0]-1,search_point[1]])

                # ↗
                if (search_point[0]-1>=0) and (search_point[0]-1<1024) and (search_point[1]+1>=0) and (search_point[1]+1<1024):
                    if img[search_point[0]-1][search_point[1]+1]==1:
                        if [search_point[0]-1,search_point[1]+1] in key_points_gt_list:
                            key_points_count += 1
                            if [search_point[0]-1,search_point[1]+1] not in key_point_searched_list:
                                key_point_searched_list.append([search_point[0]-1,search_point[1]+1])
                                anchor_link[key_points_gt_list.index([search_point[0]-1,search_point[1]+1]),i,j] = 1
                        else:
                            normal_point_list.append([search_point[0]-1,search_point[1]+1])

                # →
                if (search_point[0]>=0) and (search_point[0]<1024) and (search_point[1]+1>=0) and (search_point[1]+1<1024):
                    if img[search_point[0]][search_point[1]+1]==1:
                        if [search_point[0],search_point[1]+1] in key_points_gt_list:
                            key_points_count += 1
                            if [search_point[0],search_point[1]+1] not in key_point_searched_list:
                                key_point_searched_list.append([search_point[0],search_point[1]+1])
                                anchor_link[key_points_gt_list.index([search_point[0],search_point[1]+1]),i,j] = 1
                        else:
                            normal_point_list.append([search_point[0],search_point[1]+1])

                # ↘
                if (search_point[0]+1>=0) and (search_point[0]+1<1024) and (search_point[1]+1>=0) and (search_point[1]+1<1024):
                    if img[search_point[0]+1][search_point[1]+1]==1:
                        if [search_point[0]+1,search_point[1]+1] in key_points_gt_list:
                            key_points_count += 1
                            if [search_point[0]+1,search_point[1]+1] not in key_point_searched_list:
                                key_point_searched_list.append([search_point[0]+1,search_point[1]+1])
                                anchor_link[key_points_gt_list.index([search_point[0]+1,search_point[1]+1]),i,j] = 1
                        else:
                            normal_point_list.append([search_point[0]+1,search_point[1]+1])

                # ↓
                if (search_point[0]+1>=0) and (search_point[0]+1<1024) and (search_point[1]>=0) and (search_point[1]<1024):
                    if img[search_point[0]+1][search_point[1]]==1:
                        if [search_point[0]+1,search_point[1]] in key_points_gt_list:
                            key_points_count += 1
                            if [search_point[0]+1,search_point[1]] not in key_point_searched_list:
                                key_point_searched_list.append([search_point[0]+1,search_point[1]])
                                anchor_link[key_points_gt_list.index([search_point[0]+1,search_point[1]]),i,j] = 1
                        else:
                            normal_point_list.append([search_point[0]+1,search_point[1]])

                # ↙
                if (search_point[0]+1>=0) and (search_point[0]+1<1024) and (search_point[1]-1>=0) and (search_point[1]-1<1024):
                    if img[search_point[0]+1][search_point[1]-1]==1:
                        if [search_point[0]+1,search_point[1]-1] in key_points_gt_list:
                            key_points_count += 1
                            if [search_point[0]+1,search_point[1]-1] not in key_point_searched_list:
                                key_point_searched_list.append([search_point[0]+1,search_point[1]-1])
                                anchor_link[key_points_gt_list.index([search_point[0]+1,search_point[1]-1]),i,j] = 1
                        else:
                            normal_point_list.append([search_point[0]+1,search_point[1]-1])

                # ←
                if (search_point[0]>=0) and (search_point[0]<1024) and (search_point[1]-1>=0) and (search_point[1]-1<1024):
                    if img[search_point[0]][search_point[1]-1]==1:
                        if [search_point[0],search_point[1]-1] in key_points_gt_list:
                            key_points_count += 1
                            if [search_point[0],search_point[1]-1] not in key_point_searched_list:
                                key_point_searched_list.append([search_point[0],search_point[1]-1])
                                anchor_link[key_points_gt_list.index([search_point[0],search_point[1]-1]),i,j] = 1
                        else:
                            normal_point_list.append([search_point[0],search_point[1]-1])

                # ↖
                if (search_point[0]-1>=0) and (search_point[0]-1<1024) and (search_point[1]-1>=0) and (search_point[1]-1<1024):
                    if img[search_point[0]-1][search_point[1]-1]==1:
                        if [search_point[0]-1,search_point[1]-1] in key_points_gt_list:
                            key_points_count += 1
                            if [search_point[0]-1,search_point[1]-1] not in key_point_searched_list:
                                key_point_searched_list.append([search_point[0]-1,search_point[1]-1])
                                anchor_link[key_points_gt_list.index([search_point[0]-1,search_point[1]-1]),i,j] = 1
                        else:
                            normal_point_list.append([search_point[0]-1,search_point[1]-1])

                # If no keypoints were found in neighbors, continue DFS via normal points
                if key_points_count == 0:
                    for normal_point in normal_point_list:
                        if normal_point not in next_search_list:
                            next_search_list.append(normal_point)
        # Recursive call for next layer of normal points
        return search_link(img, next_search_list, key_points_gt_list, point_searched_list, key_point_searched_list, i, j, anchor_link, iter_count)

# Process all files in the dataset
imageID = 0
for root, dirs, files in os.walk("key_points_final"):
    for file in files:
        imageID += 1
        key_points_info = scipy.io.loadmat("key_points_final/" + file)
        if_key_points = key_points_info["if_key_points"]                      # [1, 64, 64]
        all_key_points_position = key_points_info["all_key_points_position"]  # [2, 64, 64]

        mask = cv2.imread("scribble/" + file[:-4] + ".png")  # Read the road mask
        img = mask[:,:,0]                                   # Use the red channel

        anchor_link = np.zeros((8,64,64))                   # To store 8-neighbor anchor links

        for i in range(0, 64):
            for j in range(0, 64):
                if if_key_points[0,i,j] == 0:
                    for k in range(0,8):                    # Mark all directions as -1 for non-keypoints
                        anchor_link[k,i,j] = -1
                elif if_key_points[0,i,j] == 1:
                    # Extract current keypoint
                    keypoints = [[all_key_points_position[0,i,j], all_key_points_position[1,i,j]]]

                    # Collect 8 neighbors (some might be [-1, -1] if out-of-bounds)
                    keypoints_surrounds = []
                    for dx, dy in [(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]:
                        ni, nj = i+dx, j+dy
                        if 0 <= ni < 64 and 0 <= nj < 64:
                            keypoints_surrounds.append([all_key_points_position[0,ni,nj], all_key_points_position[1,ni,nj]])
                        else:
                            keypoints_surrounds.append([-1,-1])

                    point_searched_list = []         # Keep track of visited points
                    key_point_searched_list = []     # Already found keypoints
                    iter_count = 0                   # Limit recursion depth

                    # Recursive search to link this keypoint with others
                    search_link(img, keypoints, keypoints_surrounds, point_searched_list, key_point_searched_list, i, j, anchor_link, iter_count)

        # Save result as .mat file
        final_mat_savepath = "link_key_points_final/" + file
        scipy.io.savemat(final_mat_savepath, mdict={
            'if_key_points': if_key_points,
            'all_key_points_position': all_key_points_position,
            'anchor_link': anchor_link
        })
        print("Image " + str(imageID) + ": Finished!")
