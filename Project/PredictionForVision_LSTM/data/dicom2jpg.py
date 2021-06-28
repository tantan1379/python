import SimpleITK as sitk
import xlrd
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

des_path = "F:\\Dataset\\AMD_TimeSeries_CL\\1_to_16_single\\"
book = xlrd.open_workbook("./dicom_path.xlsx")
sheet = book.sheet_by_index(0)

# # one image test
# txt = sheet.cell_value(1,4)
# # print(txt)
# img_array = sitk.GetArrayFromImage(sitk.ReadImage(str(txt)))
# img = Image.fromarray(img_array[0])
# img.save("1.jpg")

index_num = []
for index in sheet.col_values(0,start_rowx=1):
    if not os.path.exists(os.path.join(des_path, index)):
        os.mkdir(os.path.join(des_path, index))
    index_num.append(index)
for file in os.listdir(des_path):
    for i in range(1,4):
        if not os.path.exists(os.path.join(des_path, file,"V"+str(i))):
            os.mkdir(os.path.join(des_path, file,"V"+str(i)+"-OCT"))

names,v1_path,v2_path,v3_path = [],[],[],[]
for name in sheet.col_values(1,start_rowx=1):
    names.append(name)
for v1 in sheet.col_values(4,start_rowx=1):
    v1_path.append(v1)
for v2 in sheet.col_values(6,start_rowx=1):
    v2_path.append(v2)
for v3 in sheet.col_values(8,start_rowx=1):
    v3_path.append(v3)

# print(len(index_num),len(all_path))
all_row = []
for i in range(len(index_num)):
    all_row.append([index_num[i],names[i],v1_path[i],v2_path[i],v3_path[i]])

for iter,one_pat in enumerate(all_row):
    # if iter==0:
    print("["+str(iter+1)+"/"+str(len(all_row))+"]")
    for i in range(2,5):
        arr_array = sitk.GetArrayFromImage(sitk.ReadImage(one_pat[i]))
        if len(arr_array)!=19:
            print("wrong slices in pat:{} img:{},get {} slices".format(one_pat[0],one_pat[i],len(arr_array)))
            with open("wrong.txt","a") as f:
                f.write("wrong slices in pat:{} img:{},get {} slices.\r\n".format(one_pat[0],one_pat[i],len(arr_array)))
            continue
        for j in range(1,17):
            img = Image.fromarray(arr_array[j])
            numstr = ('0'+str(j)) if j<10 else str(j) 
            img.save(os.path.join(des_path,one_pat[0],"V"+str(i-1)+"-OCT",one_pat[1]+numstr+".jpg"))

