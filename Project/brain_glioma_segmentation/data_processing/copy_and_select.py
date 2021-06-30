import os
import shutil
import SimpleITK as sitk
import numpy as np

root_path = r"D:\BaiduCloud"
data_path = root_path+os.sep+"3DT1"
label_path = root_path+os.sep+"roi_204"
des_label_path = r"F:\Dataset\glioma\2d_bingbian"

# 获取可使用的标签地址
label_list= list()
for one_pat_label in os.listdir(label_path):
    one_pat_label_path = label_path+os.sep+one_pat_label
    for label in os.listdir(one_pat_label_path):
        if label[-12:-4]=="bingbian" and label[0:2]!="rr":
            label_list.append(one_pat_label_path+os.sep+label)

# 创建序号表
index_list = list()
for label in label_list:
    label_index = label.split(os.sep)[-1][4:7]
    index_list.append(label_index)

# 创建标签文件夹
for i in range(len(index_list)):
    if not os.path.exists(os.path.join(des_label_path,"label",index_list[i])):
        os.mkdir(os.path.join(des_label_path,"label",index_list[i]))

for label in label_list:
    img_array = sitk.GetArrayFromImage(sitk.ReadImage(label))
    for index,one_slice_array in enumerate(img_array):
        print("slice:"+str(index))
        print(np.sum(img_array))
        