import numpy as np
import torch
import os
import glob


# torch.cat
# a = torch.Tensor([1,2,3]).unsqueeze(0)
# b = torch.Tensor([4,5,6]).unsqueeze(0)
# c = torch.cat((a,b),dim=1)
# print(c)
dataset_path =r"F:\Dataset\Linear_lesion"
img_path = dataset_path + '/img'
fold = sorted(os.listdir(img_path))
# print(fold)
img_list = []
label_list = []
fold_r = fold
fold_r.remove('f1')  # remove testdata
for item in fold_r:
    img_list += glob.glob(os.path.join(img_path, item) + '/*.png')
    # print(len(img_list))
label_list = [x.replace('img', 'mask').split('.')[0] + '.png' for x in img_list]
print(img_list)
print("")
print(label_list)