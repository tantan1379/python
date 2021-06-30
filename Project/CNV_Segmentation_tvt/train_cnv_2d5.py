'''
@File    :   train.py
@Time    :   2021/06/07 16:06:36
@Author  :   Tan Wenhao
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.
'''

# 库函数
import argparse
import torch
import os
import socket
from torch.utils.data import DataLoader
from datetime import datetime
from tensorboardX import SummaryWriter
import tqdm
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from PIL import Image
import torch.backends.cudnn as cudnn
# 内部函数
import utils.progress_bar as pb
import utils.utils as u
import utils.loss as LS
from config import cnv_single_config
from dataset.CNV import CNV
from dataset.CNV_2d5 import CNV_2d5
from model.unet import UNet
from model.resunetplusplus import ResUnetPlusPlus
from model.cpfnet import CPFNet
# from model.resunet import DeepResUNet,HybridResUNet,ONet


def val(args, model, dataloader):
    with torch.no_grad():
        model.eval()
        val_progressor = pb.Val_ProgressBar(model_name=args.save_model_path,total=len(dataloader))

        total_Dice = []
        total_Acc = []
        total_jaccard = []
        total_Sensitivity = []
        total_Specificity = []

        for i, (data, label) in enumerate(dataloader):
            val_progressor.current = i
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            # get RGB predict image
            predict = model(data)
            Dice, Acc, jaccard, Sensitivity, Specificity = u.eval_single_seg(predict, label)

            total_Dice += Dice
            total_Acc += Acc
            total_jaccard += jaccard
            total_Sensitivity += Sensitivity
            total_Specificity += Specificity
        
            dice = sum(total_Dice) / len(total_Dice)
            acc = sum(total_Acc) / len(total_Acc)
            jac = sum(total_jaccard) / len(total_jaccard)
            sen = sum(total_Sensitivity) / len(total_Sensitivity)
            spe = sum(total_Specificity) / len(total_Specificity)
            val_progressor.val=[dice,acc,jac,sen,spe]
            val_progressor()
        val_progressor.done()
            
        return dice, acc, jac, sen, spe


def train(args, model, optimizer, criterion, dataloader_train, dataloader_val, writer=None):
    best_pred, best_acc, best_jac, best_sen, best_spe = 0.0, 0.0, 0.0, 0.0, 0.0
    best_epoch = 0
    end_epoch = None
    step = 0         # tensorboard相关
    end_index = None   # 可以设为1，用于直接进入val过程，检查bug
    current_time = datetime.now().strftime('%b%d %H:%M:%S')
    with open("./logs/%s.txt" % (args.save_model_path.split('/')[-1]), "a") as f:
        print(current_time, file=f)
    for epoch in range(args.num_epochs):
        if(epoch==end_epoch):
            break
        train_loss = u.AverageMeter()
        train_progressor = pb.Train_ProgressBar(mode='train', epoch=epoch, total_epoch=args.num_epochs,save_model_path=args.save_model_path, total=len(dataloader_train)*args.batch_size)
        lr = u.adjust_learning_rate(args, optimizer, epoch)
        model.train()

        for i, (data, label) in enumerate(dataloader_train):
            if(i==end_index):
                break
            train_progressor.current = i*args.batch_size
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            output = model(data)  

            output = torch.sigmoid(output)
            loss_aux = criterion[0](output, label)
            loss_main = criterion[1](output, label)
            loss = loss_main + loss_aux
            train_loss.update(loss.item(), data.size(0))
            train_progressor.current_loss = train_loss.avg
            train_progressor.current_lr = lr
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_progressor()
            step += 1
            if step % 10 == 0:
                writer.add_scalar('Train/loss_step', loss, step)
        train_progressor.done()
        writer.add_scalar('Train/loss_epoch', float(train_loss.avg), epoch)
        Dice, Acc, jaccard, Sensitivity, Specificity = val(args, model, dataloader_val)
        writer.add_scalar('Valid/Dice_val', Dice, epoch)
        writer.add_scalar('Valid/Acc_val', Acc, epoch)
        writer.add_scalar('Valid/Jac_val', jaccard, epoch)
        writer.add_scalar('Valid/Sen_val', Sensitivity, epoch)
        writer.add_scalar('Valid/Spe_val', Specificity, epoch)

        is_best = Dice > best_pred
        if is_best:
            best_pred = max(best_pred, Dice)
            best_jac = max(best_jac, jaccard)
            best_acc = max(best_acc, Acc)
            best_sen = max(best_sen, Sensitivity)
            best_spe = max(best_spe, Specificity)
            best_epoch = epoch+1
        checkpoint_dir = os.path.join(args.save_model_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_latest_name = os.path.join(checkpoint_dir, 'checkpoint_latest.path.tar')
        u.save_checkpoint({
            'epoch': best_epoch,
            'state_dict': model.state_dict(),
            'best_dice': best_pred
        }, best_pred, epoch, is_best, checkpoint_dir, filename=checkpoint_latest_name)
    # 记录该折分割效果最好一次epoch的所有参数
    best_indicator_message = "best pred in Epoch:{}\nDice={:.4f} Accuracy={:.4f} jaccard={:.4f} Sensitivity={:.4f} Specificity={:.4f}".format(
        best_epoch, best_pred, best_acc, best_jac, best_sen, best_spe)
    end_time = datetime.now().strftime('%b%d %H:%M:%S')
    # 写入结束时间和验证的最佳指标
    with open("./logs/%s.txt" % (args.save_model_path.split('/')[-1]), "a") as f:
        print("Test time: "+end_time, file=f)
        print(best_indicator_message, file=f)



def eval(args, model, dataloader):
    print('\nStart Test!')
    num_checkpoint = len(os.listdir(args.save_model_path))
    for iter,c in enumerate(os.listdir(args.save_model_path)):
        # if(iter==num_checkpoint-1):
        #     break
        pretrained_model_path = os.path.join(args.save_model_path,c) # 最后一个模型(最好的)
    print("Load best model "+'\"'+os.path.abspath(pretrained_model_path)+'\"')
    checkpoint = torch.load(pretrained_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    with torch.no_grad():
        model.eval()
        test_progressor = pb.Test_ProgressBar(total=len(dataloader),save_model_path=args.save_model_path)

        total_Dice = []
        total_Acc = []
        total_jaccard = []
        total_Sensitivity = []
        total_Specificity = []

        for i, (data, (label,labels)) in enumerate(dataloader):
            test_progressor.current = i
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            # get RGB predict image
            predict = model(data)
            Dice, Acc, jaccard, Sensitivity, Specificity = u.eval_single_seg(predict, label)

            total_Dice += Dice
            total_Acc += Acc
            total_jaccard += jaccard
            total_Sensitivity += Sensitivity
            total_Specificity += Specificity
        
            dice = sum(total_Dice) / len(total_Dice)
            acc = sum(total_Acc) / len(total_Acc)
            jac = sum(total_jaccard) / len(total_jaccard)
            sen = sum(total_Sensitivity) / len(total_Sensitivity)
            spe = sum(total_Specificity) / len(total_Specificity)
            test_progressor.val=[dice,acc,jac,sen,spe]
            test_progressor()    
        test_progressor.done()



def main(mode='train', args=None, writer=None):
    # create dataset and dataloader
    dataset_path = os.path.join(args.data, args.dataset)
    dataset_train = CNV_2d5(dataset_path, scale=(args.crop_height, args.crop_width), mode='train')
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True)
    dataset_val = CNV_2d5(dataset_path, scale=(args.crop_height, args.crop_width), mode='val')
    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=1, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=False)
    dataset_test = CNV_2d5(dataset_path, scale=(args.crop_height, args.crop_width), mode='test')
    dataloader_test = DataLoader(
        dataset_test, 
        batch_size=1, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=False)
    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    # load model
    model_all = {'unet': UNet(in_channels=3, n_classes=args.num_classes),
                 'resunetplusplus': ResUnetPlusPlus(channel=3),
                 'cpfnet':CPFNet(),
                 }
    model = model_all[args.net_work].cuda()
    cudnn.benchmark = True

    if(args.optimizer=="SGD"):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif(args.optimizer=="Adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    
    criterion_aux = nn.BCELoss()
    criterion_main = LS.DiceLoss()
    # criterion_0 = LS.SD_Loss()
    criterion = [criterion_aux, criterion_main]
    if mode == 'train':  # tv
        train(args, model, optimizer, criterion, dataloader_train, dataloader_val, writer)
    if mode == 'test':  # 单独使用测试集
        eval(args, model, dataloader_test)
    if mode == 'train_test':
        train(args, model, optimizer, criterion, dataloader_train, dataloader_val,writer)
        eval(args, model, dataloader_test)

if __name__ == "__main__":
    seed = 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args = cnv_single_config()
    modes = args.mode

    if modes == 'train' or modes == 'train_test':
        comments = os.getcwd().split(os.sep)[-1]
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(args.log_dirs, args.net_work + '_' + current_time + '_' + socket.gethostname())
        writer = SummaryWriter(log_dir=log_dir)
        if modes=='train':
            main(mode='train', args=args, writer=writer)
        else:
            main(mode='train_test', args=args, writer=writer)
    elif modes == 'test':
        main(mode='test', args=args, writer=None)


