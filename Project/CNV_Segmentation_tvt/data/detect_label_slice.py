import os
import numpy as np
import cv2

root = r"F:\Dataset\CNV_Seg\png\all_masks"
pat_array = []

for one_pat_time in os.listdir(root):
    one_pat_time_path = root + os.sep + one_pat_time
    for one_pic in os.listdir(one_pat_time_path):
        # print(one_pic)
        one_pic_path = one_pat_time_path+os.sep+one_pic
        # print(one_pic_path)
        one_pic_array = cv2.imread(one_pic_path,flags=cv2.IMREAD_GRAYSCALE)
        if np.sum(one_pic_array==255)!=0:
            pat_array.append(one_pat_time)
            break

print(len(pat_array))