import os
import xlrd
from xlutils import copy


rbook = xlrd.open_workbook("./dicom_path.xlsx")
rsheet = rbook.sheet_by_index(0)
index_arr,label_arr = list(),list()
new_index_label=dict()
for index in rsheet.col_values(0,start_rowx=1,end_rowx=105):
    index_arr.append(index)

for label in rsheet.col_values(10,start_rowx=1,end_rowx=105):
    label_arr.append(str(int(label)))

for i in range(len(label_arr)):
    new_index_label[index_arr[i]] = label_arr[i]
# print(new_index_label)

des_path = "F:\\Dataset\\AMD_TimeSeries_CL\\1_to_16_joint_resize_label_vision\\"
for pat in os.listdir(des_path):
    pat_path = des_path+pat
    with open(os.path.join(pat_path,"label.txt"),"w") as f:
        f.write(new_index_label[pat])
    


    