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
from sklearn.metrics import r2_score


# def GetBatch(data, label, sampleNum, batchnum=4):
#     for i in range(0, sampleNum//batchnum):
#         low = i*batchnum
#         x = data[low:low+batchnum]
#         y = label[low:low+batchnum]
#         yield x, y


# def shuffle(data, label, num):
#     indices = np.arange(num)
#     np.random.shuffle(indices)
#     data = data[indices]
#     label = label[indices]


if __name__ == '__main__':
    # 设置种子
    torch.cuda.empty_cache()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus  # 指定GPU训练
    torch.backends.cudnn.benchmark = True  # 加快卷积计算速度

    # 相关参数设置
    num_pat = config.num_pat
    seq_len = config.seq_len
    start_epoch = 0
    K = config.K
    savedir = os.path.dirname(__file__)+os.sep+'save'+os.sep
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # 训练
    train_loss_sum = AverageMeter()
    train_acc_sum = AverageMeter()
    val_loss_sum = AverageMeter()
    val_acc_sum = AverageMeter()
    for ki in range(K):
        model1 = ExtractFeature().cuda()
        model2 = LSTM().cuda()
        criteria = nn.MSELoss()
        optimizer = optim.Adam([{'params': model1.parameters()},
                                {'params': model2.parameters()}], lr=config.lr,
                               amsgrad=True)
        print("k{} fold validation starts:".format(ki+1))
        train_dataset = TreatmentRequirement(
            config.data_path, mode='train', ki=ki, K=K)
        val_dataset = TreatmentRequirement(
            config.data_path, mode='val', ki=ki, K=K)
        train_loader = DataLoader(train_dataset,batch_size=config.batch_size,pin_memory=True,shuffle=True,num_workers=0)
        val_loader = DataLoader(val_dataset,batch_size=config.batch_size,pin_memory=False,shuffle=False,num_workers=0)

        for epoch in range(start_epoch, config.epochs):
            train_loss = AverageMeter()
            val_loss = AverageMeter()
            for data,target in train_loader:
                model1.train()
                model2.train()
                with torch.no_grad():
                    data = data.cuda()
                    target = torch.from_numpy(np.array(target)).cuda()
                optimizer.zero_grad()
                batch_data = torch.zeros(config.batch_size,config.seq_len,2048)
                for i,one_data in enumerate(data):
                    one_data = model1(one_data)
                    batch_data[i] = one_data
                batch_data = batch_data.cuda()
                pred = model2(batch_data)
                loss = torch.sqrt(criteria(pred, target))
                # with torch.no_grad():
                #     print(pred.cpu(), target.cpu())
                    # r2_score = r2_score(pred.cpu(), target.cpu())
                loss.backward()
                optimizer.step()
                train_loss.update(loss.item(), data.size(0))

            for data,target in val_loader:
                model1.eval()
                model2.eval()
                with torch.no_grad():
                    data = data.cuda()
                    target = torch.from_numpy(np.array(target)).long().cuda()
                batch_data = torch.zeros(config.batch_size,config.seq_len,2048)
                for i,one_data in enumerate(data):
                    one_data = model1(one_data)
                    batch_data[i] = one_data
                batch_data = batch_data.cuda()
                pred = model2(batch_data)
                # print(pred.shape)
                # print(target.shape)
                loss = torch.sqrt(criteria(pred, target))
                precision1_val = accuracy(pred, target)
                val_loss.update(loss.item(), data.size(0))
            print('epoch:{},train  loss:{},\ntest  loss:{}\n'.format(
                epoch, train_loss.avg, val_loss.avg ))
            # print('epoch:{},train  loss:{},precision:{}'.format(
            #         epoch, train_loss.avg,train_acc.avg))
    #     train_loss_sum.update(train_loss.avg)
    #     val_loss_sum.update(val_loss.avg)
    # print("Result of fold validation:train  loss:{}, precision:{}\ntest  loss:{},precision:{}".format(
    #     train_loss_sum.avg, train_acc_sum.avg, val_loss_sum.avg, val_acc_sum.avg))
