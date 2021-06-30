#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: Pokemon.py
# author: twh
# time: 2020/11/17 11:50
import glob
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import csv
from PIL import Image
import random


# Note: Class inherited from Dataset must have __len__(self) & __getitem__(self,index)
class Pokemon(Dataset):
    def __init__(self, root, resize, mode):  # mode = train/val/test
        super(Pokemon, self).__init__()
        self.root = root
        self.resize = resize
        self.name2label = {}
        for name in sorted(os.listdir(self.root)):
            if not os.path.isdir(os.path.join(self.root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        self.images, self.labels = self.load_csv('images&labels.csv')
        # Devide the whole dataset into three modes
        assert (mode == 'train' or mode == 'val' or mode == 'test'), 'invalid mode'
        if mode == 'train':  # dataset(0%~70%) as train_set
            self.images = self.images[:int(0.7 * len(self.images))]
            self.labels = self.labels[:int(0.7 * len(self.images))]
        elif mode == 'val':  # dataset(70%~85%) as validation_set
            self.images = self.images[int(0.7 * len(self.images)):int(0.85 * len(self.images))]
            self.labels = self.labels[int(0.7 * len(self.images)):int(0.85 * len(self.images))]
        elif mode == 'test':  # dataset(85%~100%) as test_set
            self.images = self.images[int(0.85 * len(self.images)):]
            self.labels = self.labels[int(0.85 * len(self.images)):]
        else:
            pass

    def get_label(self):
        return self.name2label

    # Create csv file if file does not exist in current directory
    # Save the path instead of the origin picture in order to avoid memory explosion
    def load_csv(self, filename):
        # If the csv file doesn't exist, create it; Instead, continue
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                # glob.glob() return to all the file paths under the file in bracket
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            print(len(images))
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]  # os.sep = '\'
                    label = self.name2label[name]
                    writer.writerow([img, label])
                print('image paths and labels have been writen into csv file:', filename)

        # Read in data
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]  # img : string, labels : tensor
        # Data augment to avoid overfitting
        trans = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),
            transforms.Resize((int(self.resize*1.25),int(self.resize*1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = trans(img)
        label = torch.tensor(label)
        return img, label


# train_db = Pokemon(r'F:\Database\pokemon', 224, 'train')  # dataset(0%~70%) as train_set
# val_db = Pokemon(r'F:\Database\pokemon', 224, 'val')  # dataset(70%~85%) as validation_set
# test_db = Pokemon(r'F:\Database\pokemon', 224, 'test')  # dataset(85%~100%) as test_set