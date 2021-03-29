#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: dataset.py
# author: twh
# time: 2021/3/11 21:30
from torch.utils import data
import os
import glob
import random
import csv
import torch
from torchvision import transforms
from PIL import Image


class AMD_CL(data.Dataset):
    def __init__(self, root, resize, mode, transform):
        super(AMD_CL, self).__init__()
        self.root = root
        self.transform = transform
        self.resize = resize
        self.images, self.labels = self.load_csv('images&labels.csv')
        assert (mode == "train" or mode == "val" or mode == "test"), "invalid mode input"
        if mode == "train":
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == "val":
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        elif mode == "train" :
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]
        else:
            pass

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in os.listdir(self.root):
                images += glob.glob(os.path.join(self.root, name, "*.jpg"))
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline="") as f:
                writer = csv.writer(f)
                for img in images:
                    label = img.split(os.sep)[-2]
                    writer.writerow([img, label])
                print('image paths and labels have been writen into csv file:', filename)
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                images.append(img)
                labels.append(int(label) - 1)
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]  # img : string, labels : tensor
        trans = {
            'train': transforms.Compose([
                lambda x: Image.open(x).convert("RGB"),
                transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
                transforms.RandomRotation(15),
                transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            ]),
            'val': transforms.Compose([
                lambda x: Image.open(x).convert("RGB"),
                transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
                transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }
        if self.transform:
            img = trans['train'](img)
        else:
            img = trans['val'](img)
        label = torch.tensor(label)
        return img, label


AMD_CL(r"F:\Lab\AMD_CL\preprocessed", 224, 'train', True)