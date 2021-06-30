import os
import random
import time
import json
import torch
import torchvision
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from torch import nn, optim
from config import config
from collections import OrderedDict
from torch.utils.data import DataLoader
from dataset.dataloader import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from timeit import default_timer as timer
from models.model import *
from utils import *
from IPython import embed
# 1. 设置种子和初始参数
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus  # 指定GPU训练
torch.backends.cudnn.benchmark = True  # 加快卷积计算速度


# 2. 验证
def evaluate(val_loader, model, optimizer, criterion, epoch):
    # 2.1 为损失和准确率创建“标尺”（拥有计数、求和、求平均功能）
    losses = AverageMeter()
    top1 = AverageMeter()
    # 2.2 创建进度条
    val_progressor = ProgressBar(mode="Val  ", epoch=epoch, total_epoch=config.epochs, model_name=config.model_name,
                                 total=len(val_loader))
    # 2.3 验证过程
    model.cuda()
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            val_progressor.current
            input = input.cuda()
            target = torch.from_numpy(np.array(target)).long().cuda()  # 范式
            output = model(input)  # 将input输入模型得到预测输出
            loss = criterion(output, target)  # 根据预测和真实标签求损失
            # 2.3.2 计算准确率并实时更新进度条的显示内容
            precision = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top1.update(precision[0], input.size(0))
            val_progressor.current_loss = losses.avg
            val_progressor.current_top1 = top1.avg
            val_progressor.current_lr = get_learning_rate(optimizer)
            val_progressor()
        val_progressor.done()
    return [losses.avg, top1.avg]


# 4. 主函数
def main():
    fold = 0
    # 4.1 创建相应的文件夹
    if not os.path.exists(config.submit):
        os.mkdir(config.submit)
    if not os.path.exists(config.weights):
        os.mkdir(config.weights)
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    if not os.path.exists(config.weights + config.model_name + os.sep + str(fold) + os.sep):
        os.makedirs(config.weights + config.model_name +
                    os.sep + str(fold) + os.sep)
    if not os.path.exists(config.best_models + config.model_name + os.sep + str(fold) + os.sep):
        os.makedirs(config.best_models + config.model_name +
                    os.sep + str(fold) + os.sep)
    # 4.2 输入模型、优化器、损失函数
    model = get_net().cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.lr,
                           amsgrad=True, weight_decay=config.weight_decay)
    # optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6)

    # 4.3 重启参数
    start_epoch = 0
    best_precision1 = 0
    # best_precision_save = 0

    # 4.4 重启程序
    assert(config.resume == "restart" or config.resume ==
           "best" or config.resume == "last"), print("error input")
    if config.resume == "best":
        checkpoint = torch.load(
            config.best_models + config.model_name + os.sep + str(fold) + "\\model_best.pth.tar")
        start_epoch = checkpoint["epoch"]
        fold = checkpoint["fold"]
        best_precision1 = checkpoint["best_precision1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    elif config.resume == "last":
        checkpoint = torch.load(
            config.weights + config.model_name + os.sep + str(fold) + "\\_checkpoint.pth.tar")
        start_epoch = checkpoint["epoch"]
        fold = checkpoint["fold"]
        best_precision1 = checkpoint["best_precision1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # 4.5 get files and split for K-fold dataset
    # 4.5.1 读入文件列表
    train_data_list = get_files(config.train_data)
    val_data_list = get_files(config.val_data)
    # test_files = get_files(config.test_data,"test")

    """ 
    #4.5.2 split
    split_fold = StratifiedKFold(n_splits=3)
    folds_indexes = split_fold.split(X=origin_files["filename"],y=origin_files["label"])
    folds_indexes = np.array(list(folds_indexes))
    fold_index = folds_indexes[fold]

    #4.5.3 using fold index to split for train data and val data
    train_data_list = pd.concat([origin_files["filename"][fold_index[0]],origin_files["label"][fold_index[0]]],axis=1)
    val_data_list = pd.concat([origin_files["filename"][fold_index[1]],origin_files["label"][fold_index[1]]],axis=1)
    """
    # train_data_list,val_data_list = train_test_split(origin_files,test_size = 0.1,stratify=origin_files["label"])
    # 4.5.4 将文件列表加载到dataloader中
    # dataset = ChaojieDataset(train_data_list,'train')
    # print(dataset.imgs)
    train_dataloader = DataLoader(ChaojieDataset(train_data_list,'train'), batch_size=config.batch_size, shuffle=True,
                                  collate_fn=collate_fn, pin_memory=True, num_workers=0)
    val_dataloader = DataLoader(ChaojieDataset(val_data_list,'val'), batch_size=config.batch_size * 2,
                                shuffle=True, collate_fn=collate_fn, pin_memory=False, num_workers=0)
    # test_dataloader = DataLoader(ChaojieDataset(test_files,test=True),batch_size=1,shuffle=False,pin_memory=False)

    # 4.5.5 训练部分
    model.train()
    train_losses = AverageMeter()
    train_top1 = AverageMeter()
    valid_loss = [np.inf, 0, 0]
    for epoch in range(start_epoch, config.epochs):
        train_progressor = ProgressBar(mode="Train", epoch=epoch, total_epoch=config.epochs,
                                       model_name=config.model_name,total=len(train_dataloader))
        for iter, (img, target) in enumerate(train_dataloader):
            train_progressor.current = iter
            model.train()
            img = img.cuda()
            target = torch.from_numpy(np.array(target)).long().cuda()
            output = model(img)
            # print(target.shape)
            # print(output.shape)
            loss = criterion(output, target)
            precision1_train = accuracy(output, target)
            train_losses.update(loss.item(), img.size(0))  # img.size(0) = batch
            train_top1.update(precision1_train[0], img.size(0))
            train_progressor.current_loss = train_losses.avg
            train_progressor.current_top1 = train_top1.avg
            train_progressor.current_lr = optimizer.param_groups[0]["lr"]
            # BP算法（三范式）
            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播
            optimizer.step()  # 梯度更新
            train_progressor()  # 调用__call__，每batch更新并输出一次进度条
        scheduler.step(loss.item())  # 学习率衰减更新
        train_progressor.done()  # 调用进度条的done函数：（1）将进度条拉满 （2）向log文件输出当前的结果
        # 验证过程
        valid_info = evaluate(val_dataloader, model,
                              optimizer, criterion, epoch)
        # 每次检查该epoch获取的模型是否使验证集准确率更优，如果是则保存该模型
        is_best = valid_info[1] > best_precision1
        best_precision1 = max(valid_info[1], best_precision1)
        save_checkpoint({
            "epoch": epoch + 1,
            "model_name": config.model_name,
            "state_dict": model.state_dict(),
            "best_precision1": best_precision1,
            "optimizer": optimizer.state_dict(),
            "fold": fold,
            "valid_loss": valid_loss,
        }, is_best, fold)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
