'''
@File    :   CNV_AND_SRF_2d5.py
@Time    :   2021/06/27 14:26:12
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.
'''

import sys
sys.path.append("..")
import torch
import cv2
import glob
import os
from torchvision import transforms
import torch.utils.data as data
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
from model.unet_multi import UNet
import utils.loss as LS
import imgaug as ia
import random


class CNV_AND_SRF_2d5(data.Dataset):
    def __init__(self, dataset_path, scale, mode='train'):
        super().__init__()
        self.mode = mode
        self.img_path = dataset_path+'/img'
        self.images_list, self.labels_list = self.read_list(self.img_path)
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5),                                                 # 水平翻转
            iaa.SomeOf(n=(0,2),children=[                                   
                iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},                # 尺度缩放
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # 平移
                    rotate=(-10, 10)),                                       # 旋转
                iaa.OneOf([
                    iaa.GaussianBlur((0, 1.0)),
                    iaa.AverageBlur(k=(0, 3)),
                    iaa.MedianBlur(k=(1, 3))]),
                iaa.AdditiveGaussianNoise(scale=(0.0, 0.06 * 255)),  # 高斯噪声
                iaa.contrast.LinearContrast((0.9, 1.1))
            ],random_order=True)
        ])
        # self.resize_label = transforms.Resize(scale, Image.NEAREST)   # 标签缩放（最近邻插值）[对于标签不需要过高质量]
        # self.resize_img = transforms.Resize(scale, Image.BILINEAR)    # 图像缩放（双线性插值）
        self.to_tensor = transforms.ToTensor()                          # Image对象转Tensor

    def __getitem__(self, index):
        # load image
        img_cur = cv2.imread(self.images_list[index],cv2.IMREAD_GRAYSCALE)
        length = len(self.images_list)

        # 2.5D
        pre_index = 0 if index == 0 else index-1
        next_index = index if index >= length-1 else index+1

        if pre_index == index:
            img_pre = img_cur
        else:
            if self.images_list[pre_index].split('/')[-2] == self.images_list[index].split('/')[-2]: # 要求这三张图像属于同一病人的同一时间段，如都属于chen_L_time001
                img_pre = cv2.imread(self.images_list[pre_index], cv2.IMREAD_GRAYSCALE)
            else:
                img_pre = img_cur

        if next_index == index:
            img_next = img_cur
        else:
            if self.images_list[next_index].split('/')[-2] == self.images_list[index].split('/')[-2]:
                img_next = cv2.imread(self.images_list[next_index], cv2.IMREAD_GRAYSCALE)
            else:
                img_next = img_cur

        img = np.stack((img_pre, img_cur, img_next), axis=2).astype(np.uint8) # 3pic->1pic
        labels = self.labels_list[index]
        # load label
        label_origin = cv2.imread(self.labels_list[index],cv2.IMREAD_GRAYSCALE)
        label = np.zeros(shape=(label_origin.shape[0],label_origin.shape[1]),dtype=np.uint8)
        label = np.array(label)

        # convert RGB to one hot 
        label = np.where(label_origin == 128, 1, label)
        label = np.where(label_origin == 255, 2, label)

        # augment image and label
        if self.mode == 'train':  # 训练时对图像和标签数据增强
            seq_det = self.seq.to_deterministic()  # 创建数据增强的序列
            segmap = ia.SegmentationMapsOnImage(label, shape=label.shape) # 将分割结果转换为SegmentationMapOnImage类型，方便后面可视化
            img = seq_det.augment_image(img) # 对图像进行数据增强
            label = seq_det.augment_segmentation_maps([segmap])[0].get_arr().astype(np.uint8) # 将数据增强应用在分割标签上，并且转换成np类型
            # label = np.reshape(label, (1,)+label.shape)
            # label = torch.from_numpy(label.copy()).float()
            labels = label

        elif self.mode=='val':
            # label = np.reshape(label, (1,)+label.shape)
            # label = torch.from_numpy(label.copy()).float()
            labels = label

        elif self.mode=='test':
            # label = np.reshape(label, (1,)+label.shape)
            # label = torch.from_numpy(label.copy()).float()
            labels = [label,labels]

        # img = self.to_tensor(img.copy()).float()
        img = self.to_tensor(img.copy())
        return img, labels

    def __len__(self):
        return len(self.images_list)

    def read_list(self, image_path):
        img_list = list()
        label_list = list()
        if self.mode == 'train':
            for f in os.listdir(os.path.join(image_path,'train')):
                img_list += sorted(glob.glob(os.path.join(image_path, 'train',f)+'/*.png'))
            label_list = [x.replace('img', 'all_mask') for x in img_list]
            # print(label_list)

        elif self.mode == 'val':
            for f in os.listdir(os.path.join(image_path,'val')):
                img_list += sorted(glob.glob(os.path.join(image_path, 'val',f)+'/*.png'))
            label_list = [x.replace('img', 'all_mask') for x in img_list]

        elif self.mode == 'test':
            for f in os.listdir(os.path.join(image_path,'test')):
                img_list += sorted(glob.glob(os.path.join(image_path, 'test',f)+'/*.png'))
            label_list = [x.replace('img', 'all_mask') for x in img_list]

        return img_list, label_list


if __name__ == '__main__':
    dataset_path = r"F:\Dataset\CNV_Seg\png_split_tvt"
    dataset = CNV_AND_SRF_2d5(dataset_path, scale=(512, 256), mode='train')
    # print(len(dataset))
    dataloader = data.DataLoader(dataset,batch_size=2, shuffle=False)
    criterion = LS.SD_Loss()
    model = UNet(in_channels=3, n_classes=3).cuda()
    for image,label in dataloader:
        print(image.shape)
        print(label.shape)
        image = image.cuda()
        label = label.cuda().long()
        output = model(image)
        loss = criterion(output,label)
        print(output.shape)
        print(loss.item())
        print("")
