import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import xlrd
import os

des_path = "F:\\Dataset\\AMD_TimeSeries_CL\\1_to_16_joint_resize_label_vision\\"
root = "F:\\Dataset\\AMD_TimeSeries_CL\\1_to_16_single\\"
book = xlrd.open_workbook("./dicom_path.xlsx")
sheet = book.sheet_by_index(0)
for index in sheet.col_values(0,start_rowx=1):
    if not os.path.exists(os.path.join(des_path, index)):
        os.mkdir(os.path.join(des_path,index))



# one_v test
# one_v = "F:\\Dataset\\AMD_TimeSeries_CL\\0_to_18_single\\047\\V1-OCT\\"
# for iter,one_img in enumerate(os.listdir(one_v)):
#     one_img_array = plt.imread(os.path.join(one_v,one_img))
#     # print(one_img_array.shape)
#     roi = one_img_array[100:350,100:400]
#     if iter==0:
#         res1 = roi
#     elif(iter>0 and iter<4):
#         res1 = np.hstack((res1,roi))
#     elif iter==4:
#         res2 = roi
#     elif(iter>4 and iter<8):
#         res2 = np.hstack((res2,roi))
#     elif iter==8:
#         res3 = roi
#     elif(iter>8 and iter<12):
#         res3 = np.hstack((res3,roi))
#     elif iter==12:
#         res4 = roi
#     elif iter>12:
#         res4 = np.hstack((res4,roi))
# img_out = np.vstack((res1,res2,res3,res4))
# outimg = Image.fromarray(img_out)

    
for i,one_pat in enumerate(os.listdir(root)):
    print("["+str(i+1)+"/"+str(len(os.listdir(root)))+"]")
    for one_v in os.listdir(os.path.join(root,one_pat)):         
        for iter,one_img in enumerate(os.listdir(os.path.join(root,one_pat,one_v))):
            # print(one_img)
            one_img_array = plt.imread(os.path.join(root,one_pat,one_v,one_img))
            # print(one_img_array.shape)
            roi = one_img_array[:,:]
            if iter==0:
                res1 = roi
            elif(iter>0 and iter<4):
                res1 = np.hstack((res1,roi))
            elif iter==4:
                res2 = roi
            elif(iter>4 and iter<8):
                res2 = np.hstack((res2,roi))
            elif iter==8:
                res3 = roi
            elif(iter>8 and iter<12):
                res3 = np.hstack((res3,roi))
            elif iter==12:
                res4 = roi
            elif iter>12:
                res4 = np.hstack((res4,roi))
        img_out = np.vstack((res1,res2,res3,res4))
        outimg = Image.fromarray(img_out)
        outimg_resize = outimg.resize((448, 448),Image.ANTIALIAS)
        outimg_resize.save(os.path.join(des_path,one_pat,one_pat+"-"+one_v+".jpg"))