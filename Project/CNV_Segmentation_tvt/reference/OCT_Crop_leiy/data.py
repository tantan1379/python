import torch
import glob
import os
import sys

from torchvision import transforms
from torchvision.transforms import functional as F
import cv2
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
import random
import skimage.io as io
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def augmentation():
    # augment images with spatial transformation: Flip, Affine, Rotation, etc...
    # see https://github.com/aleju/imgaug for more details
    pass

def augmentation_pixel():
    # augment images with pixel intensity transformation: GaussianBlur, Multiply, etc...
    pass

class SegOCT(Dataset):
    def __init__(self, base_path, scale = (512, 512), mode='train',label_number = "label_1", k_fold_test = 1):
        super().__init__()
        self.mode = mode
        self.image_path = base_path + '/data_image'
        self.label_path = base_path + '/data_label/' + label_number
        self.label_number = label_number
        self.images_list, self.labels_list = self.read_list(self.image_path, k_fold_test=k_fold_test)
        self.scale = scale

        self.flip = iaa.SomeOf((1, 4), [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-15, 15)), # 仿射变换
            iaa.ContrastNormalization((0.9, 1.1))], random_order=True) # 图像对比度
        # resize
        self.resize_label = transforms.Resize(scale, Image.BILINEAR)
        self.resize_img = transforms.Resize(scale, Image.BILINEAR)
        # normalization
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # load image and crop
        # print(self.images_list[index])
        img_cur = cv2.resize(cv2.imread(self.images_list[index],cv2.IMREAD_GRAYSCALE),self.scale)
        length = len(self.images_list)

        if index == 0:
            pre_index = 0
        else:
            pre_index = index - 1

        if index >= length - 1:
            next_index = index
        else:
            next_index = index + 1

        if self.images_list[pre_index].split('/')[-1] == self.images_list[index].split('/')[-1]:
            img_pre = img_cur
        else:
            if self.images_list[pre_index].split('/')[-1].split('_')[0] == self.images_list[index].split('/')[-1].split('_')[0]:
                img_pre = cv2.resize(cv2.imread(self.images_list[pre_index], cv2.IMREAD_GRAYSCALE),self.scale)
            else:
                img_pre = img_cur
        if self.images_list[next_index].split('/')[-1] == self.images_list[index].split('/')[-1]:
            img_next = img_cur
        else:
            if self.images_list[index].split('/')[-1].split('_')[0] == self.images_list[next_index].split('/')[-1].split('_')[0]:
                img_next = cv2.resize(cv2.imread(self.images_list[next_index], cv2.IMREAD_GRAYSCALE),self.scale)
            else:
                img_next = img_cur

        img = np.stack((img_pre, img_cur, img_next), axis=2).astype(np.uint8)  # 2.5D
        labels = self.labels_list[index]


        label_ori = np.uint8(cv2.resize(cv2.imread(self.labels_list[index], cv2.IMREAD_GRAYSCALE), self.scale))
        label = np.zeros(shape=(label_ori.shape[0], label_ori.shape[1]), dtype=np.uint8)

        # convert RGB  to one hot
        label = np.where(label_ori == 63, 1, label)
        label = np.where(label_ori == 255, 2, label)

        # cv2.imshow('image_pre', img[:, :, 0])
        # cv2.imshow('image_cur', img[:, :, 1])
        # cv2.imshow('image_nex', img[:, :, 2])
        # cv2.imshow('label_0', np.uint8(np.where(label == 0, 255, label)))
        # cv2.imshow('label_1', np.uint8(np.where(label == 1, 255, label)))
        # cv2.imshow('label_2', np.uint8(np.where(label == 2, 255, label)))
        # cv2.waitKey()
        # label = torch.from_numpy(label.copy()).float()

        # load label 0:背景 1:CME 2:MH

        if self.mode == 'train' :
            # augment image and label
            seq_det = self.flip.to_deterministic()  # 固定变换
            segmap = ia.SegmentationMapOnImage(label, shape=label.shape)
            img = seq_det.augment_image(img)
            label = seq_det.augment_segmentation_maps([segmap])[0].get_arr_int().astype(np.uint8)

        # imgs = img.transpose(2, 0, 1)
        # img = torch.from_numpy(imgs.copy()).float()  # self.to_tensor(img.copy()).float()
        img = self.to_tensor(img.copy())
        # label = self.to_tensor(label.copy())[0]

        return img, label

    def __len__(self):
        return len(self.images_list)

    def read_list(self, image_path, k_fold_test = 1):
        fold = sorted(os.listdir(image_path))
        image_list = []
        if self.mode == 'train':
            fold_r = fold
            fold_r.remove('f' + str(k_fold_test))  # remove testdata

            # Mac系统
            if '.DS_Store' in fold_r:
                fold_r.remove('.DS_Store')

            for item in fold_r:
                for dir in os.listdir(os.path.join(image_path, item)):
                    image_list += sorted(sorted(glob.glob(os.path.join(image_path, item, dir) + '/*.png')),
                                         key=lambda i: len(i))
            label_list = [x.replace('data_image', 'data_label/' + self.label_number) for x in image_list]

        elif self.mode == 'val' or self.mode == 'train_eval' or self.mode == 'test':
            fold_s = fold[k_fold_test - 1]
            # Mac系统
            if '.DS_Store' in fold_s:
                fold_r.remove('.DS_Store')

            for dir in sorted(os.listdir(os.path.join(image_path, fold_s))):
                image_list += sorted(sorted(glob.glob(os.path.join(image_path, fold_s, dir) + '/*.png')),
                                     key=lambda i: len(i))
            label_list = [x.replace('data_image', 'data_label/' + self.label_number) for x in image_list]

        return image_list, label_list

if __name__ == '__main__':
    path = '/home/leiy/DLProjects/Graduation_Design/OCT_Crop_Doctor/data_crop'
    segOCT_dataset = SegOCT(path,(512,512), mode='train_eval')

    dataloader_test = DataLoader(
        segOCT_dataset,
        # this has to be 1
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    for i,(data,label) in enumerate(dataloader_test):
        for index in range(data.shape[1]):
            img_0 = np.uint8(data[0,0,:,:] * 255)
            img_1 = np.uint8(data[0,1,:,:] * 255)
            img_2 = np.uint8(data[0,2,:,:] * 255)
            lab = np.uint8(label[0])
            lab = np.where(lab == 1, 63, lab).astype(np.uint8)
            lab = np.where(lab == 2, 255, lab).astype(np.uint8)

            cv2.imshow('img_0',img_0)
            cv2.imshow('img_1',img_1)
            cv2.imshow('img_2',img_2)

            cv2.imshow('label',lab)
            cv2.waitKey()







