import warnings
import random
import os
import time
import torch
import torchvision as tv
from torch import nn, optim
import numpy as np
from utils import *
from dataloader.dataset import TreatmentRequirement
from config import config
from torch.utils.data import DataLoader
from model.ResNetLSTM import ExtractFeature, LSTM
from itertools import chain


if __name__ == '__main__':
    # -------------------------------
    # 设置种子
    # -------------------------------
    torch.cuda.empty_cache()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus  # 指定GPU训练
    torch.backends.cudnn.benchmark = True  # 加快卷积计算速度
    # -------------------------------
    # 相关参数设置
    # -------------------------------
    seq_len = config.seq_len
    start_epoch = 0
    K = config.K
    savedir = os.path.dirname(__file__)+os.sep+'save'+os.sep
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    # -------------------------------
    # 训练
    # -------------------------------
    train_loss_sum = AverageMeter()
    train_acc_sum = AverageMeter()
    val_loss_sum = AverageMeter()
    val_acc_sum = AverageMeter()
    for ki in range(K):
        model1 = ExtractFeature().cuda()
        model2 = LSTM().cuda()
        criteria = nn.CrossEntropyLoss()
        optimizer = optim.Adam(chain(model1.parameters(), model2.parameters()), lr=config.lr,
                               )
        # optimizer = optim.Adam([{'params': model1.parameters()},
        #                         {'params': model2.parameters()}], lr=config.lr,
#                           amsgrad=True)
        # optimizer = optim.Adam(model1.parameters()+model2.parameters(), lr=config.lr,
        # amsgrad=True)
        print("k{} fold validation starts:".format(ki+1))
        train_dataset = TreatmentRequirement(
            config.data_path, mode='train', ki=ki, K=K)
        val_dataset = TreatmentRequirement(
            config.data_path, mode='val', ki=ki, K=K)
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, pin_memory=True, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                                pin_memory=False, shuffle=False, num_workers=0)
        best_precision = 0

        for epoch in range(start_epoch, config.epochs):
            train_loss = AverageMeter()
            train_acc = AverageMeter()
            val_loss = AverageMeter()
            val_acc = AverageMeter()
            # with torch.no_grad():
            for data, target in train_loader:
                model1.train()
                model2.train()
                target = torch.from_numpy(
                    np.array(target)).squeeze(1).long().cuda()
                for one_batch in range(data.size(0)):
                    one_batch_feature = torch.zeros(
                        data.size(0), data.size(1), 2048).cuda()
                    merged_channel_feature = torch.zeros(data.size(1), 2048)
                    for one_pic in range(data.size(1)):
                        one_p_3d = torch.zeros(
                            3, config.img_height, config.img_width).cuda()
                        one_p = data[one_batch, one_pic]
                        for i in range(3):
                            one_p_3d[i] = one_p
                        one_p_3d = model1(one_p_3d.unsqueeze(0))
                        merged_channel_feature[one_pic] = one_p_3d
                    one_batch_feature[one_batch] = merged_channel_feature
                pred = model2(one_batch_feature)
                loss = criteria(pred, target)
                precision1_train = accuracy(pred, target)
                train_loss.update(loss.item(), data.size(0))
                train_acc.update(precision1_train, data.size(0))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for data, target in val_loader:
                model1.eval()
                model2.eval()
                target = torch.from_numpy(
                    np.array(target)).squeeze(1).long().cuda()
                for one_batch in range(data.size(0)):
                    one_batch_feature = torch.zeros(data.size(0), data.size(1), 2048).cuda()
                    merged_channel_feature = torch.zeros(data.size(1), 2048)
                    for one_pic in range(data.size(1)):
                        one_p_3d = torch.zeros(
                            3, config.img_height, config.img_width).cuda()
                        one_p = data[one_batch, one_pic]
                        for i in range(3):
                            one_p_3d[i] = one_p
                        one_p_3d = model1(one_p_3d.unsqueeze(0))
                        merged_channel_feature[one_pic] = one_p_3d
                    one_batch_feature[one_batch] = merged_channel_feature

                pred = model2(one_batch_feature)
                loss = criteria(pred, target)
                precision1_val = accuracy(pred, target)
                val_loss.update(loss.item(), data.size(0))
                val_acc.update(precision1_val, data.size(0))

            print('epoch {}:\ntrain loss:{}, train precision:{}\ntest loss:{}, test precision:{}\n'.format(
                epoch+1, train_loss.avg, train_acc.avg, val_loss.avg, val_acc.avg))
            # print("best_precision={} in epoch{}".format(best_precision, best_epoch))
            # print('epoch:{},train  loss:{},precision:{}'.format(
            #         epoch, train_loss.avg,train_acc.avg))
