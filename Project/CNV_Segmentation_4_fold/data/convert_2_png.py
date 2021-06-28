"""
Created on Sun May 31 15:42 2021

@author:twh
"""

import os
import shutil
import SimpleITK as sitk
import numpy as np
import cv2


def double_linear(input_signal, zoom_multiples):
    input_signal_cp = np.copy(input_signal)
    input_row, input_col = input_signal_cp.shape
    output_row = int(input_row*zoom_multiples)
    output_col = int(input_col*zoom_multiples)

    output_signal = np.zeros((output_row, output_col))
    for i in range(output_row):
        for j in range(output_col):
            temp_x = i/output_row*input_row
            temp_y = j/output_col*input_col
            x1 = int(temp_x)
            y1 = int(temp_y)
            x2 = x1
            y2 = y1+1
            x3 = x1+1
            y3 = y1
            x4 = x1+1
            y4 = y1+1
            u = temp_x-x1
            v = temp_y-y1
            if x4 >= input_row:
                x4 = input_row - 1
                x2 = x4
                x1 = x4 - 1
                x3 = x4 - 1
            if y4 >= input_col:
               y4 = input_col - 1
               y3 = y4
               y1 = y4 - 1
               y2 = y4 - 1
            # 插值
            output_signal[i, j] = (1-u)*(1-v)*int(input_signal_cp[x1, y1]) + (1-u)*v*int(input_signal_cp[x2, y2]) + u*(1-v)*int(input_signal_cp[x3, y3]) + u*v*int(input_signal_cp[x4, y4])
    return output_signal

label_path = r"F:\Dataset\CNV_Segmentation\origin\mask"
des_label_path = r"F:\Dataset\CNV_Segmentation\png\cnv_masks"
name_list = list()
img_list = list()

for one_img in os.listdir(label_path):
    name_list.append(one_img[:-7])
    one_img_path = label_path+os.sep+one_img
    img_list.append(one_img_path)

for i in range(len(name_list)):
    if not os.path.exists(os.path.join(des_label_path,name_list[i])):
        os.mkdir(os.path.join(des_label_path,name_list[i]))

for i in range(len(img_list)):
    one_img_array = sitk.GetArrayFromImage(sitk.ReadImage(img_list[i]))
    for iter,one_slice in enumerate(one_img_array):
        one_slice_path = des_label_path + os.sep + name_list[i]
        # print(one_slice_path)
        # print(np.sum(one_slice==1))
        one_slice[one_slice!=2] = 0
        one_slice[one_slice==2] = 255
        one_slice = double_linear(one_slice,0.5)
        des_img = one_slice_path+os.sep+str(iter+1)+".png"
        cv2.imwrite(des_img,one_slice)
    print(name_list[i],"finished!")