'''
@File    :   CNV_AND_SRF.py
@Time    :   2021/06/20 13:20:29
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.
'''

import torch
import glob
import os
from torchvision import transforms
import torch.utils.data as data
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
import random


class CNV_AND_SRF(data.Dataset):
    def __init__(self, dataset_path, scale, k_fold_test=1, mode='train'):
        super().__init__()
        self.mode = mode
        self.img_path = dataset_path+'/img'
        self.mask_path = dataset_path+'/mask'
        self.image_lists, self.label_lists = self.read_list(
            self.img_path, k_fold_test=k_fold_test)
        self.seq = iaa.SomeOf(n=(2, 4), children=[  # 数据增强序列（同时适用于img和mask)
            iaa.Dropout([0.05, 0.2]),               # 随机失活
            iaa.Fliplr(0.5),                        # 水平翻转
            iaa.Flipud(0.5),                        # 垂直翻转
            iaa.Affine(rotate=(-30, 30)),           # 随机旋转
            iaa.AdditiveGaussianNoise(scale=(0.0, 0.08*255))], random_order=True)  # 高斯噪声
        # self.resize_label = transforms.Resize(scale,Image.NEAREST)               # 标签缩放（最近邻插值）[对于标签不需要过高质量]
        # self.resize_img = transforms.Resize(scale, Image.BILINEAR)               # 图像缩放（双线性插值）
        self.to_tensor = transforms.ToTensor()                                     # Image对象转Tensor

    def __getitem__(self, index):
        # 读图（转为Image对象）
        img = Image.open(self.image_lists[index])
        # 图像转numpy矩阵
        img = np.array(img)
        labels = self.label_lists[index]
        if self.mode != 'test':
            # 读标签（转为Image对象）
            label = Image.open(self.label_lists[index])
            # 标签转numpy矩阵
            label = np.array(label)
            # 标签归一化
            label[label != 255] = 0
            label[label == 255] = 1
            # 训练时对图像和标签数据增强
            if(self.mode == 'train'):
                # 创建数据增强的序列
                seq_det = self.seq.to_deterministic()
                # 将分割结果转换为SegmentationMapOnImage类型，方便后面可视化
                segmap = ia.SegmentationMapsOnImage(label, shape=label.shape)
                # 对图像进行数据增强
                img = seq_det.augment_image(img)
                # 将数据增强应用在分割标签上，并且转换成np类型
                label = seq_det.augment_segmentation_maps([segmap])[0].get_arr().astype(
                    np.uint8)      

            label = np.reshape(label, (1,)+label.shape)
            label = torch.from_numpy(label.copy()).float()
            labels = label
        img = np.reshape(img, img.shape+(1,))
        img = self.to_tensor(img.copy()).float()
        return img, labels

    def __len__(self):
        return len(self.image_lists)

    def read_list(self, image_path, k_fold_test=1):
        fold = sorted(os.listdir(image_path))
        img_list = list()
        label_list = list()
        if self.mode == 'train':
            fold_r = fold
            fold_r.remove('f'+str(k_fold_test))
            for item in fold_r:
                img_list += glob.glob(os.path.join(image_path, item)+'/*.png')
            label_list = [x.replace('img', 'mask') for x in img_list]
        elif self.mode == 'val' or self.mode == 'test':
            fold_s = fold[k_fold_test-1]
            img_list = glob.glob(os.path.join(image_path, fold_s)+'/*.png')
            label_list = [x.replace('img', 'mask') for x in img_list]
        return img_list, label_list


if __name__ == '__main__':
    dataset_path = r"F:\Dataset\CNV_Segmentation\png_split"
    dataset = CNV_AND_SRF(dataset_path, scale=(512, 256))
    # print(len(dataset))
    for i in range(len(dataset)):
        dataset[i]
