import glob
import os
import pandas as pd
from dataset.dataloader import *
from config import config
from torch.utils.data import DataLoader
from cv2 import cv2
from PIL import Image

# ----------------------------------------------------------------
#TODO Tensor
# a = torch.Tensor([3,3])
# a = torch.tensor([x for x in range(-2,4)]).reshape(2,3)
# print(a)
# print(torch.sigmoid(a))
# print(a.max())
# print(torch.max(a,dim=0))
# print(np.argmax(a))
# print(a.argmax())

# ----------------------------------------------------------------
#TODO 数据处理
# all_images = []
# image_folders = list(map(lambda x: root + x, os.listdir(root)))


# for f in image_folders:
#     # print(f)
#     all_images += glob.glob(os.path.join(f,"*.jpg"))

# print(image_folders)
# print("")
# print(all_images)

# ----------------------------------------------------------------
#TODO 尝试dataframe
# root = "C:\\Users\\TRT\\Desktop\\testset\\"
# images,labels,all_files = [],[],[]
# title = ['images','labels']
# for file in os.listdir(root):
#     images+=glob.glob(os.path.join(root,file,"*.jpg"))
# for image in images:
#     labels.append(int(image.split(os.sep)[-2]))

# img_label = np.concatenate((np.array(images).reshape(-1,1),np.array(labels).reshape(-1,1)),axis=1)
# # print(img_label)
# index = [i for i in range(1,len(labels)+1)]
# all_files= pd.DataFrame({"images":images,"labels":labels},index=index)
# # print(all_files)
# # print(all_files[all_files['labels']>=2]) # 布尔索引：选取labels>1的所有信息
# # print(all_files[lambda all_files:all_files.columns[0]])
# # all_files2= pd.DataFrame(img_label,columns=title,index=index)
# images=[]
# for _,row in all_files.iterrows():
#     images.append((row["images"],row["labels"]))
# print(images[:5])

# ----------------------------------------------------------------
#TODO 查看dataloader原理
# train_data_list = get_files(config.train_data)
# train_dataloader = DataLoader(ChaojieDataset(train_data_list, transform = 'val'),batch_size=config.batch_size, shuffle=True,
#                                pin_memory=True, num_workers=0)
# for iter,(image,label) in enumerate(train_dataloader):
#     pass


# ----------------------------------------------------------------
#TODO 解决cv2必须以英文路径作为输入问题：用PIL.Image读入，再转换
# img = "F:\\Lab\\AMD_CL\\split\\val\\2\\高章红_000007.jpg"
# img_1 = Image.open(img).convert("RGB")
# img_r1 = img_1.resize((int(config.img_height * 1.5), int(config.img_weight * 1.5)),Image.ANTIALIAS)
# print(img_1.size)
# # print(img_r1.size)
# img_2 = cv2.imread(img)
# print(img_2.shape)

# ----------------------------------------------------------------
#TODO 删除列表中的空值
# mylist = ['1','2','3','','4']
# # while None in mylist:
# #     mylist.remove(None)
# while "" in mylist:
#     mylist.remove("")

# print(mylist)

# ----------------------------------------------------------------
#TODO size和shape
# a = torch.tensor([[[1,2,3],[4,5,6]]])
# print(a.shape)
# print(a.size())
# print(a.dim())

# ----------------------------------------------------------------
#TODO squeeze
# a = torch.tensor([_ for _ in range(1,7)]).reshape(1,2,3)
# print(a)
# print(a.numel())
# print(torch.squeeze(a))

# ----------------------------------------------------------------
#TODO zip
# matrix = np.zeros((3, 3))
# preds = [0,0,1,0,1,2,2,1]
# labels = [0,1,1,1,0,2,1,2]

# for p,t in zip(preds, labels):
#     matrix[p,t]+=1

# print(matrix)

# ----------------------------------------------------------------
#TODO topk
# a = torch.tensor([1,2,3,4,5,6])
# b = torch.tensor([2,2,3,4,5,6])
# # b = b.view(1,-1)
# print(torch.topk(a,3,dim=0))

# ----------------------------------------------------------------
#TODO densenet
# import torch.nn as nn 
# import torchvision as tv
# class ExtractFeature_(nn.Module):
#     def __init__(self):
#         super(ExtractFeature_, self).__init__()
#         self.densenet = tv.models.densenet121(pretrained=True)
#         self.final_pool = nn.AdaptiveAvgPool2d(1)
#         # self.final_pool = torch.nn.MaxPool2d(3, 2)

#     def forward(self,x):
#         x = self.densenet.conv0(x)
#         x = self.densenet.norm0(x)
#         x = self.densenet.relu0(x)  
#         x = self.final_pool(x).squeeze()
#         # x = x.flatten(start_dim=1)
#         return x

# x = torch.Tensor(3,224,224)
# model = ExtractFeature_()
# model(x)

# ----------------------------------------------------------------
#TODO [0*5]
print([0.0]*5)