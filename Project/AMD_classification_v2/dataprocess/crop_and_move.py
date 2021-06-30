from PIL import Image
import matplotlib.pyplot as plt
import xlrd
import os
import shutil

des_path = r"F:\Dataset\AMD\AMD_CL\not_split"
root_path = r"F:\Dataset\AMD\AMD_Origin_2d"

# 单张图片裁剪
# one_img_path = r"F:\Dataset\AMD\AMD_CL\not_split\01-A-0024-V1-OCT\沈宝根000.jpg"

# img_array = plt.imread(one_img_path)
# cropped_img_array = img_array[0:400,500:-20]
# img = plt.imshow(cropped_img_array)
# plt.show()

# 读取表格
sheet_path = "./0305AMD应答标注.xlsx"
rbook = xlrd.open_workbook(sheet_path)
rsheet = rbook.sheet_by_index(0)
index_list = list()
label_list = list()
for index in rsheet.col_values(0,start_rowx=2,end_rowx=144):
    index_list.append(index)

for label in rsheet.col_values(25,start_rowx=2,end_rowx=144):
    label_list.append(int(label))

# 建立病人和标签的字典
index_to_label = dict()
for i in range(len(index_list)):
    index_to_label[index_list[i]] = label_list[i]

# 选取有数据的病人和标签
data_index_list = list()
for file in os.listdir(root_path):
    index = file[6:12]
    # print(index)
    if index[0:3] not in data_index_list and index[-1]=="1":
        data_index_list.append(index[0:3])

# print(data_index_list)
# print(len(data_index_list))

# # 选取既有标签又有数据的病人
real_index_list = list()
for index in index_list:
    if index in data_index_list:
        real_index_list.append(index)

# print(real_index_list)
# print(len(real_index_list))


for i in range(len(real_index_list)):
    if not os.path.exists(os.path.join(des_path,real_index_list[i]+"-V1")):
        os.mkdir(os.path.join(des_path,real_index_list[i]+"-V1"))

for i in range(len(real_index_list)):
    if(i>=34):
        for one_img in os.listdir(os.path.join(root_path,"01-A-0"+str(real_index_list[i])+"-V1-OCT")):
            if(one_img[-1]!="g"):
                continue
            one_img_path = os.path.join(root_path,"01-A-0"+str(real_index_list[i])+"-V1-OCT",one_img)
            img_array = plt.imread(one_img_path)
            cropped_img_array = img_array[0:400,500:-20]
            img = plt.imshow(cropped_img_array)
            plt.axis("off")
            plt.savefig(os.path.join(des_path,real_index_list[i]+"-V1",one_img))