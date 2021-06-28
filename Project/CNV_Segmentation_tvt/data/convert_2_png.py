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

img_path = r'F:\Dataset\CNV_Seg\origin\img'
des_img_path = r'F:\Dataset\CNV_Seg\png\all_imgs'

label_path = r'F:\Dataset\CNV_Seg\origin\mask'
des_label_path = r'F:\Dataset\CNV_Seg\png\all_masks'
des_cnv_label_path = r'F:\Dataset\CNV_Seg\png\cnv_masks'
des_srf_label_path = r'F:\Dataset\CNV_Seg\png\srf_masks'
name_list = list()
img_list = list()
label_list = list()

for one_img in os.listdir(img_path):
    name_list.append(one_img[:-7])
    one_img_path = img_path+os.sep+one_img
    img_list.append(one_img_path)
label_list =[x.replace('img', 'mask') for x in img_list]

# print(img_list)
# print(len(img_list))
# print(label_list)
# print(len(label_list))

if not os.path.exists(des_img_path):
    os.mkdir(des_img_path)

if not os.path.exists(des_label_path):
    os.mkdir(des_label_path)

if not os.path.exists(des_cnv_label_path):
    os.mkdir(des_cnv_label_path)

if not os.path.exists(des_srf_label_path):
    os.mkdir(des_srf_label_path)

for i in range(len(name_list)):
    if not os.path.exists(os.path.join(des_img_path,name_list[i])):
        os.mkdir(os.path.join(des_img_path,name_list[i]))

for i in range(len(name_list)):
    if not os.path.exists(os.path.join(des_label_path,name_list[i])):
        os.mkdir(os.path.join(des_label_path,name_list[i]))

for i in range(len(name_list)):
    if not os.path.exists(os.path.join(des_cnv_label_path,name_list[i])):
        os.mkdir(os.path.join(des_cnv_label_path,name_list[i]))

for i in range(len(name_list)):
    if not os.path.exists(os.path.join(des_srf_label_path,name_list[i])):
        os.mkdir(os.path.join(des_srf_label_path,name_list[i]))

# img->png
for i in range(len(img_list)):
    one_img_array = sitk.GetArrayFromImage(sitk.ReadImage(img_list[i]))
    for iter,one_slice in enumerate(one_img_array):
        one_slice_path = des_img_path + os.sep + name_list[i]
        one_slice = double_linear(one_slice,0.5)
        des_img = one_slice_path+os.sep+name_list[i]+'_'+str(iter+1)+'.png'
        cv2.imwrite(des_img,one_slice)
    print("img for",name_list[i],'finished!')

# all_mask->png
for i in range(len(label_list)):
    one_img_array = sitk.GetArrayFromImage(sitk.ReadImage(label_list[i]))
    for iter,one_slice in enumerate(one_img_array):
        one_slice_path = des_label_path + os.sep + name_list[i]
        one_slice = double_linear(one_slice,0.5)
        one_slice[one_slice==1]=128
        one_slice[one_slice==2]=255
        des_img = one_slice_path+os.sep+name_list[i]+'_'+str(iter+1)+'.png'
        cv2.imwrite(des_img,one_slice)
    print("all label for",name_list[i],'finished!')

# cnv_mask->png
for i in range(len(label_list)):
    one_img_array = sitk.GetArrayFromImage(sitk.ReadImage(label_list[i]))
    for iter,one_slice in enumerate(one_img_array):
        one_slice_path = des_cnv_label_path + os.sep + name_list[i]
        one_slice = double_linear(one_slice,0.5)
        one_slice[one_slice==2]=255
        des_img = one_slice_path+os.sep+name_list[i]+'_'+str(iter+1)+'.png'
        cv2.imwrite(des_img,one_slice)
    print("cnv label for",name_list[i],'finished!')

# srf_mask->png
for i in range(len(label_list)):
    one_img_array = sitk.GetArrayFromImage(sitk.ReadImage(label_list[i]))
    for iter,one_slice in enumerate(one_img_array):
        one_slice_path = des_srf_label_path + os.sep + name_list[i]
        one_slice = double_linear(one_slice,0.5)
        one_slice[one_slice==1]=255
        des_img = one_slice_path+os.sep+name_list[i]+'_'+str(iter+1)+'.png'
        cv2.imwrite(des_img,one_slice)
    print("srf label for",name_list[i],'finished!')