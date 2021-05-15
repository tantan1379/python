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
import glob

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
            frames_one_pat = []
            # -------------------------------
            # 6 channels data
            # -------------------------------
            # for one_v in os.listdir(patpath):
            #     if not os.path.isdir(os.path.join(patpath,one_v)):
            #         continue
            #     vpath = patpath + os.sep + one_v
            #     frames_one_v = []
            #     for one_pic in os.listdir(vpath):
            #         picpath = vpath+os.sep+one_pic
            #         frames_one_v.append(picpath)
            #     frames_one_pat.append(frames_one_v)
            # -------------------------------
            # 18 channels data
            # -------------------------------
            for one_v in os.listdir(patpath):
                if not os.path.isdir(os.path.join(patpath,one_v)):
                        continue
                vpath = patpath + os.sep + one_v
                for one_pic in os.listdir(vpath):
                    frames_one_pat.append(os.path.join(patpath,one_v,one_pic))
            imgs.append(frames_one_pat)
            labelPath = patpath + os.sep + 'label.txt'
            tx = open(labelPath)
            str1 = tx.read()
            tx.close()
            labels.append([int(str1)-1])
        imgs = np.array(imgs)
        labels = np.array(labels)
        # print(labels)
        return imgs, labels

    def __getitem__(self,index):
        img, label = self.imgs[index], self.labels[index]
        # trans = {
        #     "train": transforms.Compose([
        #         lambda x: Image.open(x),
        #         # transforms.RandomRotation(15),
        #         transforms.CenterCrop((config.center_crop_height, config.center_crop_width)),
        #         transforms.Resize((int(config.img_height),
        #             int(config.img_width))),
        #         transforms.ToTensor(),
        #         transforms.Normalize(0.4630,0.2163)
        #     ]),
        #     "val": transforms.Compose([
        #         lambda x: Image.open(x),
        #         transforms.CenterCrop((config.center_crop_height, config.center_crop_width)),
        #         transforms.Resize((int(config.img_height),
        #             int(config.img_width))),
        #         transforms.ToTensor(),
        #         transforms.Normalize(0.4630,0.2163)
        #     ])}

        # -------------------------------
        # no transform
        # -------------------------------
        trans = transforms.Compose([
                lambda x: Image.open(x),
                transforms.Resize((int(config.img_height),
                    int(config.img_width))),
                # transforms.CenterCrop((config.img_height, config.img_width)),
                transforms.ToTensor(),
                transforms.Normalize(0.4630,0.2163)
        ])

        # -------------------------------
        # 6 channels proprecess 
        # -------------------------------
        # if self.mode=="train":
        #     all_v_img = torch.zeros(config.seq_len,config.input_channel,config.img_height,config.img_width)

        #     for one_v in range(config.seq_len):
        #         merged_img = torch.zeros(config.input_channel,config.img_height,config.img_width)
        #         for one_pic in range(config.input_channel):
        #             single_img = img[one_v,one_pic]
        #             single_img = trans["train"](single_img)
        #             merged_img[one_pic]=single_img
        #         all_v_img[one_v] = merged_img
        #
        # else:
        #     all_v_img = torch.zeros(config.seq_len,config.input_channel,config.img_height,config.img_width)
        #     for one_v in range(config.seq_len):
        #         merged_img = torch.zeros(config.input_channel,config.img_height,config.img_width)
        #         for one_pic in range(config.input_channel):
        #             single_img = img[one_v,one_pic]
        #             single_img = trans["val"](single_img)
        #             merged_img[one_pic]=single_img
        #         all_v_img[one_v] = merged_img

        # ------------------------------
        # 18 channels preprocessed
        # -------------------------------
        # if self.mode=="train":
        #     all_v_img = torch.zeros(config.seq_len,config.img_height,config.img_width)
        #     for one_pic in range(config.seq_len):
        #         single_img = img[one_pic]
        #         single_img = trans["train"](single_img)
        #         all_v_img[one_pic] = single_img
        #     img = all_v_img
        # else:
        #     all_v_img = torch.zeros(config.seq_len,config.img_height,config.img_width)
        #     for one_pic in range(config.seq_len):
        #         single_img = img[one_pic]
        #         single_img = trans["val"](single_img)
        #         all_v_img[one_pic] = single_img
        #     img = all_v_img

        all_v_img = torch.zeros(config.seq_len,config.img_height,config.img_width)
        for one_pic in range(config.seq_len):
            single_img = img[one_pic]
            single_img = trans(single_img)
            all_v_img[one_pic] = single_img
        img = all_v_img
        label = torch.Tensor(label)
        return img, label

    def __len__(self):
        return len(self.imgs)


def main():
    dataset = TreatmentRequirement(config.data_path,"train",0,3)
    # imgs,_ = dataset.get_img_label(config.data_path)
    # img = imgs[0]
    # print(img.shape)
    print(dataset[0][0].shape)
    trans = {
    "train": transforms.Compose([
        lambda x: Image.open(x),
        transforms.Resize((int(config.img_height * 1.25),
                            int(config.img_width * 1.25))),
        transforms.RandomRotation(15),
        transforms.CenterCrop((config.img_height, config.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(0.4630,0.2163)
    ]),
    "val": transforms.Compose([
        lambda x: Image.open(x),
        transforms.Resize((int(config.img_height * 1.25),
            int(config.img_width * 1.25))),
        transforms.CenterCrop((config.img_height, config.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(0.4630,0.2163)
    ])}
    # all_v_img = img.unsqueeze(0)
    # # print(all_v_img.shape)
    # for one_v in range(config.seq_len):
    #     merged_img = torch.zeros(config.input_channel,config.img_height,config.img_width)
    #     for one_pic in range(config.input_channel):
    #         single_img = img[one_v,one_pic]
    #         single_img = trans["train"](single_img)
    #         merged_img[one_pic]=single_img
    #     all_v_img[one_v] = merged_img
    # print(all_v_img)

if __name__ == "__main__":
    main()