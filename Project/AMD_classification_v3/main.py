import os
import random
import time
import json
import torch
import torchvision
import numpy as np
import pandas as pd
import warnings
import socket
from tensorboardX import SummaryWriter
from datetime import datetime
from torch import nn, optim
from config import DefaultConfig
from torch.utils.data import DataLoader
from dataset.dataloader import *
from models.model import *
from utils.cfm import ConfusionMatrix
import utils.progress_bar as pb
import utils.utils as u


def val(args, model,criterion, dataloader, epoch, k_fold):
    with torch.no_grad():
        model.eval()
        top1_m = u.AverageMeter()
        loss_m = u.AverageMeter()
        # precision_m = u.AverageMeter()
        # recall_m = u.AverageMeter()
        # f1_m = u.AverageMeter()
        eval_progressor = pb.Val_ProgressBar(
            mode='val', fold=k_fold, model_name=args.model_name,epoch=epoch+1,total=len(dataloader))
        for i, (data, label) in enumerate(dataloader):
            eval_progressor.current = i
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            predict = model(data)
            loss = criterion(predict,label)
            top1 = u.accuracy(predict,label)
            top1_m.update(top1[0].item(),data.size(0))
            acc = top1_m.avg
            loss_m.update(loss.item())
            loss = loss_m.avg
            eval_progressor.val = [acc,loss,0,0]
            eval_progressor()
        eval_progressor.done()
    return acc,0,0,0

def train(args, model, optimizer, criterion, dataloader_train, dataloader_val, writer, k_fold):
    best_pred, best_pre, best_rec, best_f1 = 0.0, 0.0, 0.0, 0.0
    best_epoch = 0
    step = 0
    train_loss = u.AverageMeter()
    top1_m = u.AverageMeter()
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    with open("./logs/%s.txt" % (args.model_name), "a") as f:
        print(current_time, file=f)
    for epoch in range(args.num_epochs):
        train_progressor = pb.Train_ProgressBar(mode='train', fold=k_fold, epoch=epoch, total_epoch=args.num_epochs,
                                                model_name=args.model_name, total=len(dataloader_train)*args.batch_size)
        lr = u.adjust_learning_rate(args, optimizer, epoch)
        model.train()
        for i, (data, label) in enumerate(dataloader_train):
            train_progressor.current = i*args.batch_size
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            pred = model(data)
            loss = criterion(pred,label)
            top1 = u.accuracy(pred,label)
            top1_m.update(top1[0],data.size(0))
            train_loss.update(loss.item(), data.size(0))
            train_progressor.current_loss = train_loss.avg
            train_progressor.current_lr = lr
            train_progressor.top1 = top1_m.avg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_progressor()
            step += 1
            if step % 10 == 0:
                writer.add_scalar(
                    'Train/loss_step_{}'.format(int(k_fold)), loss,step)
        train_progressor.done()
        writer.add_scalar(
            'Train/loss_epoch_{}'.format(int(k_fold)), float(train_loss.avg), epoch)
        Accuracy, Precision, Recall, F1 = val(
            args, model,criterion, dataloader_val, epoch, k_fold)
        writer.add_scalar(
            'Valid/Accuracy_val_{}'.format(int(k_fold)), Accuracy, epoch)
        writer.add_scalar(
            'Valid/Precision_val_{}'.format(int(k_fold)), Precision, epoch)
        writer.add_scalar(
            'Valid/Recall_val_{}'.format(int(k_fold)), Recall, epoch)
        writer.add_scalar(
            'Valid/F1_val_{}'.format(int(k_fold)), F1, epoch)

        is_best = Accuracy > best_pred
        if is_best:
            best_pred = max(best_pred, Accuracy)
            best_pre = max(best_pre, Precision)
            best_rec = max(best_rec, Recall)
            best_f1 = max(best_f1, F1)
            best_epoch = epoch+1
        checkpoint_dir = os.path.join(args.save_model_path, str(k_fold))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_latest_name = os.path.join(
            checkpoint_dir, 'checkpoint_latest.path.tar')
        # print(best_pred)
        u.save_checkpoint({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'best_dice': best_pred
        }, best_pred, epoch, is_best, checkpoint_dir, filename=checkpoint_latest_name)
    # 记录该折分类最好一次epoch的所有参数
    best_indicator_message = "f{} best pred in Epoch:{}\Accuracy={} Precision={} Recall={} F1={}".format(
        k_fold, best_epoch,best_pred,best_pre, best_rec, best_f1)
    with open("./logs/%s_%s_best_indicator.txt" % (args.model_name), mode='a') as f:
        print(best_indicator_message,file=f)


# 4. 主函数
def main(mode='train',args=None,writer=None,k_fold=1):
    dataset_train = CNVDataset(args.datapath,k_fold_test=k_fold,mode="train")
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dataset_val = CNVDataset(args.datapath,k_fold_test=k_fold,mode="val")
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True)
    model = get_net().cuda()
    torch.backends.cudnn.benchmark = True
    optimizer = optim.Adam(model.parameters(), lr=args.lr,amsgrad=True, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()
    train(args, model, optimizer, criterion, dataloader_train, dataloader_val, writer, k_fold)


if __name__ == "__main__":
    args = DefaultConfig()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    comments = os.getcwd().split(os.sep)[-1]
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(args.log_dirs, comments +
                            '_' + current_time + '_' + socket.gethostname())
    # print(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    for i in range(args.start_fold-1,args.k_fold):
        # print(i)
        main(mode='train', args=args, writer=writer, k_fold=int(i + 1))
