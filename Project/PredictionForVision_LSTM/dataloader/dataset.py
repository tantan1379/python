import sys
sys.path.append("..")
import torch
import random
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from config import config
import numpy as np
import random


random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

class TreatmentRequirement(Dataset):
    def __init__(self, datapath,mode,ki,K):
        self.patnum = len(os.listdir(datapath))
        self.imgs, self.labels = self.get_img_label(datapath)
        self.mode = mode
        self.seq_len = config.seq_len
        every_z_len = self.patnum // K
        if mode=="val":
            self.imgs = self.imgs[every_z_len*ki:every_z_len*(ki+1)]
            self.labels = self.labels[every_z_len*ki:every_z_len*(ki+1)]
        elif mode=="train":
            self.imgs = np.vstack((self.imgs[:every_z_len*ki],self.imgs[every_z_len*(ki+1):]))
            self.labels = np.vstack((self.labels[:every_z_len*ki],self.labels[every_z_len*(ki+1):]))

    def get_img_label(self, dataPath):
        imgs,labels = [],[]
        for pat in os.listdir(dataPath):
            patpath = dataPath + pat
            frames = []
            for i in range(1, config.seq_len+1):
                imgname = '{}-V{}-OCT.jpg'.format(pat, i)
                frames.append(patpath+os.sep+imgname)
            imgs.append(frames)
            labelPath = patpath + os.sep + 'label.txt'
            tx = open(labelPath)
            str1 = tx.read()
            tx.close()
            labels.append([int(str1)-1])
        imgs = np.array(imgs)
        labels = np.array(labels)
        return imgs, labels

    def __getitem__(self,index):
        img, label = self.imgs[index], self.labels[index]
        nimg = torch.zeros(self.seq_len,3,config.img_height,config.img_width)
        # trans = {
        #     "train": transforms.Compose([
        #         lambda x: Image.open(x).convert("RGB"),
        #         transforms.Resize((int(config.img_height * 1.25),
        #                            int(config.img_width * 1.25))),
        #         transforms.RandomRotation(15),
        #         transforms.CenterCrop((config.img_height, config.img_width)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        #             0.229, 0.224, 0.225])
        #     ]),
        #     "val": transforms.Compose([
        #         lambda x: Image.open(x).convert("RGB"),
        #         transforms.Resize((int(config.img_height * 1.25),
        #             int(config.img_width * 1.25))),
        #         transforms.CenterCrop((config.img_height, config.img_width)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        #                              0.229, 0.224, 0.225])
        #     ])}

        trans = transforms.Compose([
                lambda x: Image.open(x).convert("RGB"),
                transforms.Resize((int(config.img_height),
                    int(config.img_width))),
                # transforms.CenterCrop((config.img_height, config.img_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
        ])

        # if self.mode=="train":
        #     for f in range(self.seq_len):
        #         single_img = img[f]
        #         single_img = trans["train"](single_img)
        #         nimg[f] = single_img
        # else:
        #     for f in range(self.seq_len):
        #         single_img = img[f]
        #         single_img = trans["val"](single_img)
        #         nimg[f] = single_img

        for f in range(self.seq_len):
                single_img = img[f]
                single_img = trans(single_img)
                nimg[f] = single_img

        label = torch.Tensor(label)
        return nimg, label

    def __len__(self):
        return len(self.imgs)