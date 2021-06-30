import os
import cv2
import matplotlib.pyplot as plt

root = r"F:\Dataset\AMD\AMD_Origin_2d_v1_cropped_split"

for iter,f in enumerate(os.listdir(root)):
    one_f_path = root+os.sep+f
    # print(iter,len(os.listdir(one_pat_path)))
    for one_pat in os.listdir(one_f_path):
        one_pic_path = one_f_path+os.sep+one_pat
        for one_pic in os.listdir(one_pic_path):
            img = plt.imread(os.path.join(one_pic_path,one_pic))
            if(img.shape[0]!=430 or img.shape[1]!=488):
                print(one_pat)