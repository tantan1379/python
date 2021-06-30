import sys
sys.path.append("..")
import glob
import xlrd
import torch
from cv2 import cv2
import os
import pandas as pd
import numpy as np
import random
from PIL import Image
from config import DefaultConfig
from torchvision import transforms
from torch.utils.data import Dataset



class CNVDataset(Dataset):
    def __init__(self, data_path, k_fold_test=1, mode='train'):
        self.mode = mode
        self.img_list, self.label_list = self.get_files(
            data_path, k_fold_test=k_fold_test)

    def __getitem__(self, index):
        # img : string, labels : tensor
        imgs, labels = self.img_list[index], self.label_list[index]
        trans = {
            'train': transforms.Compose([
                lambda x: Image.open(x).convert('L'),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.485, std=
                                     0.229)
            ]),
            'val': transforms.Compose([
                lambda x: Image.open(x).convert('L'),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.485, std=
                                     0.229)
            ])
        }
        if self.mode=="train":

            for i,img in enumerate(imgs):
                img = trans['train'](img)
                if i==0:
                    out_img = img[:]
                else:
                    out_img = torch.cat((out_img,img),dim=0)

        elif self.mode=="val":
            for i,img in enumerate(imgs):
                img = trans['val'](img)
                if i==0:
                    out_img = img[:]
                else:
                    out_img = torch.cat((out_img,img),dim=0)

        label = torch.tensor(labels)
        # print(label)
        return out_img, label

    def __len__(self):
        return len(self.img_list)

    def get_files(self, root, k_fold_test=1):
        # 读取表格
        sheet_path = "F:/MyGit/Project/AMD_classification_v3/dataset/0305AMD应答标注.xlsx"
        rbook = xlrd.open_workbook(sheet_path)
        rsheet = rbook.sheet_by_index(0)
        index_list = list()
        label_list = list()
        for index in rsheet.col_values(0, start_rowx=2, end_rowx=144):
            index_list.append(index)

        for label in rsheet.col_values(25, start_rowx=2, end_rowx=144):
            label_list.append(int(label))
        # 建立病人和标签的字典
        index_to_label = dict()
        for i in range(len(index_list)):
            index_to_label[index_list[i]] = label_list[i]
        # 返回数据和标签
        img_list = list()
        target_list = list()
        fold = sorted(os.listdir(root))
        if self.mode == 'train':
            fold_r = fold
            fold_r.remove('f'+str(k_fold_test))
            for item in fold_r:
                for pat in os.listdir(os.path.join(root, item)):
                    block = []
                    for pic in os.listdir(os.path.join(root, item, pat)):
                        block.append(os.path.join(root, item, pat, pic))
                    img_list.append(block)
            for i in range(len(img_list)):
                target = index_to_label[img_list[i][0].split(os.sep)[-2][0:3]]
                target_list.append(target-1)

        elif self.mode == 'val':
            fold_s = fold[k_fold_test-1]
            for pat in os.listdir(os.path.join(root,fold_s)):
                block = []
                for pic in os.listdir(os.path.join(root,fold_s,pat)):
                    block.append(os.path.join(root,fold_s,pat,pic))
                img_list.append(block)
            for i in range(len(img_list)):
                target = index_to_label[img_list[i][0].split(os.sep)[-2][0:3]]
                target_list.append(target-1)

        # print(img_list)
        # print(len(img_list))
        # print(target_list)
        # print(len(target_list))
        return img_list, target_list


def collate_fn(batch):
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), label


if __name__ == "__main__":
    args = DefaultConfig()
    mydataset = CNVDataset(data_path=args.datapath,
                           k_fold_test=1, mode="train")
    print(mydataset[0][0].shape)
