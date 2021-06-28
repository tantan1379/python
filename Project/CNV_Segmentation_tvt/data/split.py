"""
Created on Tuesday June 1 10:59 2021

@author:twh

Notes: This script can only be used for the first time. If you want to resplit the dataset, it is a must to delete all the previous files first.
"""

import shutil
import os
import random

des_path = "F:\\Dataset\\CNV_Seg\\png_split_4_fold\\"
des_img_path = des_path + "img"
des_mask_path = des_path + "mask"
origin_img_path = "F:\\Dataset\\CNV_Seg\\png\\all_imgs\\"
origin_label_path = "F:\\Dataset\\CNV_Seg\\png\\cnv_masks\\"

for i in range(1,5):
    if not os.path.exists(des_img_path+os.sep+"f"+str(i)):
        os.mkdir(des_img_path+os.sep+"f"+str(i))
    if not os.path.exists(des_mask_path+os.sep+"f"+str(i)):
        os.mkdir(des_mask_path+os.sep+"f"+str(i))

# image split
one_time_img_path_list = list()
one_time_label_path_list = list()
img_2_label = list()
for one_time_img in os.listdir(origin_img_path):
    one_time_img_path = os.path.join(origin_img_path,one_time_img)
    one_time_img_path_list.append(one_time_img_path)

for one_time_label in os.listdir(origin_label_path):
    one_time_label_path = os.path.join(origin_label_path,one_time_label)
    one_time_label_path_list.append(one_time_label_path)

for i in range(len(one_time_label_path_list)):
    img_2_label.append((one_time_img_path_list[i],one_time_label_path_list[i]))
# print(img_2_label)
random.shuffle(img_2_label)


for iter,(img,label) in enumerate(img_2_label):
    span = len(img_2_label)/4
    if iter<span:
        for one_img in os.listdir(img):
            shutil.copyfile(img+os.sep+one_img,os.path.join(des_img_path,"f1",img.split(os.sep)[-1]+"_"+one_img))
        for one_label in os.listdir(label):
            shutil.copyfile(label+os.sep+one_label,os.path.join(des_mask_path,"f1",img.split(os.sep)[-1]+"_"+one_label))
    elif span<=iter<2*span:
        for one_img in os.listdir(img):
            shutil.copyfile(img+os.sep+one_img,os.path.join(des_img_path,"f2",img.split(os.sep)[-1]+"_"+one_img))
        for one_label in os.listdir(label):
            shutil.copyfile(label+os.sep+one_label,os.path.join(des_mask_path,"f2",img.split(os.sep)[-1]+"_"+one_label))
    elif 2*span<=iter<3*span:
        for one_img in os.listdir(img):
            shutil.copyfile(img+os.sep+one_img,os.path.join(des_img_path,"f3",img.split(os.sep)[-1]+"_"+one_img))
        for one_label in os.listdir(label):
            shutil.copyfile(label+os.sep+one_label,os.path.join(des_mask_path,"f3",img.split(os.sep)[-1]+"_"+one_label))
    elif 3*span<=iter<4*span:
        for one_img in os.listdir(img):
            shutil.copyfile(img+os.sep+one_img,os.path.join(des_img_path,"f4",img.split(os.sep)[-1]+"_"+one_img))
        for one_label in os.listdir(label):
            shutil.copyfile(label+os.sep+one_label,os.path.join(des_mask_path,"f4",img.split(os.sep)[-1]+"_"+one_label))



