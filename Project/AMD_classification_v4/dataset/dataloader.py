from torch.utils.data import Dataset
from torchvision import transforms
from config import config
from PIL import Image
from itertools import chain
import glob
from tqdm import tqdm
from .augmentations import get_train_transform, get_test_transform
import random
import numpy as np
import pandas as pd
import os
from cv2 import cv2
import torch

# 1.set random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)


# 2.define dataset
class ChaojieDataset(Dataset):
    def __init__(self,label_list,transform):
        imgs = []
        for _, row in label_list.iterrows():
            imgs.append((row["filename"], row["label"]))
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.imgs[index] # img : string, labels : tensor
        trans = {
            'train': transforms.Compose([
                lambda x: Image.open(x).convert("L"),
                transforms.Resize((int(config.img_height * 1.5), int(config.img_weight * 1.5))),
                transforms.RandomRotation(15),
                transforms.CenterCrop((config.img_height,config.img_weight)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.CenterCrop((config.img_height,config.img_weight)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }
        if self.transform:
            img = trans['train'](img)
        else:
            img = trans['val'](img)
        label = torch.tensor(label)
        # print(img.shape)
        # print(label.shape)
        return img, label

    def __len__(self):
        return len(self.imgs)


def collate_fn(batch):
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), label


def get_files(root):
    all_images, labels = [], []
    # image_folders = list(map(lambda x: root + x, os.listdir(root)))
    for f in os.listdir(root):
        all_images += glob.glob(os.path.join(root, f, "*.jpg"))
        # all_images += glob.glob(os.path.join(root,f, "*.png"))
    while " " in all_images:
        all_images.remove(" ")
    print("loading dataset")
    # for file in tqdm(all_images):
    for image in all_images:
        labels.append(int(image.split(os.sep)[-2])-1)
    all_files = pd.DataFrame({"filename": all_images, "label": labels})
    return all_files

