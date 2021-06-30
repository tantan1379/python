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
from config import cnv_and_srf_config
from dataset.CNV_AND_SRF_2d5 import CNV_AND_SRF_2d5
from model.unet_multi import UNet
from model.resunetplusplus import ResUnetPlusPlus
from model.cpfnet import CPFNet
# from model.resunet import DeepResUNet,HybridResUNet,ONet


def val(args, model, dataloader):
    with torch.no_grad():
        model.eval()
        val_progressor = pb.Val_ProgressBar_2(save_model_path=args.save_model_path,total=len(dataloader)) # 验证进度条，用于显示指标

        total_Dice_cnv = []
        total_Acc_cnv = []
        total_Jaccard_cnv = []
        total_Sensitivity_cnv = []
        total_Specificity_cnv = []

        total_Dice_srf = []
        total_Acc_srf = []
        total_Jaccard_srf = []
        total_Sensitivity_srf = []
        total_Specificity_srf = []

        cur_predict_cube = []
        cur_label_cube = []
        counter = 0
        end_flag = False

        for i, (data, label) in enumerate(dataloader):
            val_progressor.current = i
            H,W = data.shape[2], data.shape[3]
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            slice_num = 128
            # get RGB predict image
            predicts = model(data)
            predicts = torch.argmax(torch.exp(predicts),dim=1)
            batch_size = predicts.size()[0]
            counter += batch_size
            if counter <= slice_num:
                cur_predict_cube.append(predicts)
                cur_label_cube.append(label)
                if counter == slice_num:
                    end_flag = True
                    counter = 0
            else:
                last = batch_size - (counter - slice_num)
                last_p = predicts[0:last]
                last_l = label[0:last]

                first_p = predicts[last:]
                first_l = label[last:]

                cur_predict_cube.append(last_p)
                cur_label_cube.append(last_l)
                end_flag = True
                counter = counter - slice_num

            if end_flag:
                end_flag = False
                predict_cube = torch.cat(cur_predict_cube, dim=0).reshape(-1,H,W)
                label_cube = torch.cat(cur_label_cube, dim=0).reshape(-1,H,W)
                cur_predict_cube = []
                cur_label_cube = []
                if counter != 0:
                    cur_predict_cube.append(first_p)
                    cur_label_cube.append(first_l)

                # assert predict_cube.size()[0]==slice_num

                [Dice_srf, Dice_cnv], [Acc_srf, Acc_cnv], [Jaccard_srf, Jaccard_cnv], \
                    [Sensitivity_srf, Sensitivity_cnv], [Specificity_srf, Specificity_cnv] = u.eval_multi_seg(predict_cube,label_cube,args.num_classes)
                # 将每个batch的列表相加，在迭代中动态显示指标的平均值（最终值）
                total_Dice_cnv.append(Dice_cnv)
                total_Dice_srf.append(Dice_srf)
                total_Acc_cnv.append(Acc_cnv)
                total_Acc_srf.append(Acc_srf)
                total_Jaccard_cnv.append(Jaccard_cnv)
                total_Jaccard_srf.append(Jaccard_srf)
                total_Sensitivity_cnv.append(Sensitivity_cnv)
                total_Sensitivity_srf.append(Sensitivity_srf)
                total_Specificity_cnv.append(Specificity_cnv)
                total_Specificity_srf.append(Specificity_srf)

                dice_cnv = sum(total_Dice_cnv) / len(total_Dice_cnv)
                dice_srf = sum(total_Dice_srf) / len(total_Dice_srf)
                acc_cnv = sum(total_Acc_cnv) / len(total_Acc_cnv)
                acc_srf = sum(total_Acc_srf) / len(total_Acc_srf)
                jac_cnv = sum(total_Jaccard_cnv) / len(total_Jaccard_cnv)
                jac_srf = sum(total_Jaccard_srf) / len(total_Jaccard_srf)
                sen_cnv = sum(total_Sensitivity_cnv) / len(total_Sensitivity_cnv)
                sen_srf = sum(total_Sensitivity_srf) / len(total_Sensitivity_srf)
                spe_cnv = sum(total_Specificity_cnv) / len(total_Specificity_cnv)
                spe_srf = sum(total_Specificity_srf) / len(total_Specificity_srf)
                val_progressor.val=[[dice_srf,dice_cnv],[acc_srf,acc_cnv],[jac_srf,jac_cnv],[sen_srf,sen_cnv],[spe_srf,spe_cnv]]
                val_progressor()    
        val_progressor.done()
            
        return [dice_srf, dice_cnv], [acc_srf, acc_cnv], [jac_srf, jac_cnv], \
                [sen_srf, sen_cnv], [spe_srf, spe_cnv]


def train(args, model, optimizer, criterion, dataloader_train, dataloader_val, writer):
    best_pred = 0.0
    best_epoch = 0
    end_epoch = None # 可以设为1，用于直接进入test过程，检查bug
    step = 0         # tensorboard相关
    end_index = None # 可以设为1，用于直接进入val过程，检查bug
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    with open("./logs/%s.txt" % (args.save_model_path.split('/')[-1]), "a") as f:
        print(current_time, file=f)
    for epoch in range(args.num_epochs):
        if(epoch==end_epoch):
            break
        train_loss = u.AverageMeter() # 滑动平均
        train_progressor = pb.Train_ProgressBar(mode='train', epoch=epoch, total_epoch=args.num_epochs, 
            save_model_path=args.save_model_path, total=len(dataloader_train)*args.batch_size) # train进度条，用于显示loss和lr
        lr = u.adjust_learning_rate(args, optimizer, epoch) # 自动调节学习率
        model.train()

        for i, (data, label) in enumerate(dataloader_train):
            if i==end_index:
                break
            train_progressor.current = i*args.batch_size

            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                # print(data.shape)
                label = label.cuda().long()
                # print(label.shape)

            output = model(data)
            # print(output.shape)
            loss = criterion[0](output, label)
            train_loss.update(loss.item(), data.size(0)) # loss.item()表示去除张量的元素值，data.size()表示batchsize
            train_progressor.current_loss = train_loss.avg
            train_progressor.current_lr = lr
            optimizer.zero_grad() # 梯度清零
            loss.backward() # 反向传播
            optimizer.step() # 梯度更新
            train_progressor() # 显示进度条
            step += 1
            if step % 10 == 0:
                writer.add_scalar('Train/loss_step', loss, step)
        train_progressor.done() # 输出logs
        writer.add_scalar('Train/loss_epoch', float(train_loss.avg), epoch)

        # 计算指标
        [Dice_srf, Dice_cnv], [Acc_srf, Acc_cnv], [jaccard_srf, jaccard_cnv], \
        [Sensitivity_srf, Sensitivity_cnv], [Specificity_srf, Specificity_cnv] = val(args, model, dataloader_val)
        # 将计算结果保存在tensorboard中
        writer.add_scalar('Valid/Dice_srf_val', Dice_srf, epoch)
        writer.add_scalar('Valid/Acc_srf_val', Acc_srf, epoch)
        writer.add_scalar('Valid/Jac_srf_val', jaccard_srf, epoch)
        writer.add_scalar('Valid/Sen_srf_val', Sensitivity_srf, epoch)
        writer.add_scalar('Valid/Spe_srf_val', Specificity_srf, epoch)

        writer.add_scalar('Valid/Dice_cnv_val', Dice_cnv, epoch)
        writer.add_scalar('Valid/Acc_cnv_val', Acc_cnv, epoch)
        writer.add_scalar('Valid/Jac_cnv_val', jaccard_cnv, epoch)
        writer.add_scalar('Valid/Sen_cnv_val', Sensitivity_cnv, epoch)
        writer.add_scalar('Valid/Spe_cnv_val', Specificity_cnv, epoch)

        writer.add_scalar('Valid/Dice_avg_val', (Dice_srf + Dice_cnv) / 2, epoch)
        writer.add_scalar('Valid/Acc_avg_val',  (Acc_srf + Acc_cnv) / 2, epoch)
        writer.add_scalar('Valid/Jac_avg_val', (jaccard_srf + jaccard_cnv) / 2 , epoch)
        writer.add_scalar('Valid/Sen_avg_val', (Sensitivity_srf + Sensitivity_cnv) / 2, epoch)
        writer.add_scalar('Valid/Spe_avg_val', (Specificity_srf+Specificity_cnv) / 2, epoch)

        is_best = (Dice_srf + Dice_cnv) / 2 > best_pred
        if is_best:
            best_pred = max(best_pred, (Dice_srf + Dice_cnv) / 2)
            best_acc = (Acc_srf + Acc_cnv) / 2
            best_jac = (jaccard_srf + jaccard_cnv) / 2
            best_sen = (Sensitivity_srf + Sensitivity_cnv) / 2
            best_spe = (Specificity_srf + Specificity_cnv) / 2
            best_epoch = epoch + 1
        checkpoint_dir = os.path.join(args.save_model_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_latest_name = os.path.join(checkpoint_dir, 'checkpoint_latest.path.tar') # 保存最新的一个checkpoint用于中继训练
        u.save_checkpoint({
            'epoch': best_epoch,
            'state_dict': model.state_dict(),
            'best_dice': best_pred
        }, best_pred, epoch, is_best, checkpoint_dir, filename=checkpoint_latest_name) # 保存最好的
    # 记录该折分割效果最好一次epoch的所有参数
    best_indicator_message = "best pred in Epoch:{}\nDice={:.4f} Accuracy={:.4f} jaccard={:.4f} Sensitivity={:.4f} Specificity={:.4f}".format(
        best_epoch, best_pred, best_acc, best_jac, best_sen, best_spe)
    end_time = datetime.now().strftime('%b%d %H:%M:%S')
    with open("./logs/%s_test_indicator.txt" % (args.save_model_path.split('/')[-1]), mode='a') as f:
        print("Test time: "+end_time, file=f)
        print(best_indicator_message, file=f)


def eval(args, model, dataloader):
    print('\nStart Test!')
    num_checkpoints = len(os.listdir(os.path.join(args.save_model_path)))
    for iter,c in enumerate(os.listdir(os.path.join(args.save_model_path))):
        if iter == 1:
            break
        pretrained_model_path = os.path.join(args.save_model_path,c) # 最后一个模型(最好的)
    print("Load best model "+'\"'+os.path.abspath(pretrained_model_path)+'\"')
    checkpoint = torch.load(pretrained_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    with torch.no_grad():
        model.eval()
        test_progressor = pb.Test_ProgressBar_2(save_model_path=args.save_model_path,total=len(dataloader)) # 验证进度条，用于显示指标

        total_Dice_cnv = []
        total_Acc_cnv = []
        total_Jaccard_cnv = []
        total_Sensitivity_cnv = []
        total_Specificity_cnv = []

        total_Dice_srf = []
        total_Acc_srf = []
        total_Jaccard_srf = []
        total_Sensitivity_srf = []
        total_Specificity_srf = []

        cur_predict_cube = []
        cur_label_cube = []
        counter = 0
        end_flag = False

        for i, (data, [label,labels]) in enumerate(dataloader):
            test_progressor.current = i
            H,W = data.shape[2], data.shape[3]
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            slice_num = 128
            # get RGB predict image
            predicts = model(data)
            predicts = torch.argmax(torch.exp(predicts),dim=1)
            batch_size = predicts.size()[0]
            counter += batch_size
            if counter <= slice_num:
                cur_predict_cube.append(predicts)
                cur_label_cube.append(label)
                if counter == slice_num:
                    end_flag = True
                    counter = 0
            else:
                last = batch_size - (counter - slice_num)
                last_p = predicts[0:last]
                last_l = label[0:last]

                first_p = predicts[last:]
                first_l = label[last:]

                cur_predict_cube.append(last_p)
                cur_label_cube.append(last_l)
                end_flag = True
                counter = counter - slice_num

            if end_flag:
                end_flag = False
                predict_cube = torch.cat(cur_predict_cube, dim=0).reshape(-1,H,W)
                label_cube = torch.cat(cur_label_cube, dim=0).reshape(-1,H,W)
                cur_predict_cube = []
                cur_label_cube = []
                if counter != 0:
                    cur_predict_cube.append(first_p)
                    cur_label_cube.append(first_l)

                # assert predict_cube.size()[0]==slice_num

                [Dice_srf, Dice_cnv], [Acc_srf, Acc_cnv], [Jaccard_srf, Jaccard_cnv], \
                    [Sensitivity_srf, Sensitivity_cnv], [Specificity_srf, Specificity_cnv] = u.eval_multi_seg(predict_cube,label_cube,args.num_classes)
                # 将每个batch的列表相加，在迭代中动态显示指标的平均值（最终值）
                total_Dice_cnv.append(Dice_cnv)
                total_Dice_srf.append(Dice_srf)
                total_Acc_cnv.append(Acc_cnv)
                total_Acc_srf.append(Acc_srf)
                total_Jaccard_cnv.append(Jaccard_cnv)
                total_Jaccard_srf.append(Jaccard_srf)
                total_Sensitivity_cnv.append(Sensitivity_cnv)
                total_Sensitivity_srf.append(Sensitivity_srf)
                total_Specificity_cnv.append(Specificity_cnv)
                total_Specificity_srf.append(Specificity_srf)

                dice_cnv = sum(total_Dice_cnv) / len(total_Dice_cnv)
                dice_srf = sum(total_Dice_srf) / len(total_Dice_srf)
                acc_cnv = sum(total_Acc_cnv) / len(total_Acc_cnv)
                acc_srf = sum(total_Acc_srf) / len(total_Acc_srf)
                jac_cnv = sum(total_Jaccard_cnv) / len(total_Jaccard_cnv)
                jac_srf = sum(total_Jaccard_srf) / len(total_Jaccard_srf)
                sen_cnv = sum(total_Sensitivity_cnv) / len(total_Sensitivity_cnv)
                sen_srf = sum(total_Sensitivity_srf) / len(total_Sensitivity_srf)
                spe_cnv = sum(total_Specificity_cnv) / len(total_Specificity_cnv)
                spe_srf = sum(total_Specificity_srf) / len(total_Specificity_srf)
                test_progressor.val=[[dice_srf,dice_cnv],[acc_srf,acc_cnv],[jac_srf,jac_cnv],[sen_srf,sen_cnv],[spe_srf,spe_cnv]]
                test_progressor()    
        test_progressor.done()
            


def main(mode='train', args=None, writer=None):
    # create dataset and dataloader
    dataset_path = os.path.join(args.data, args.dataset)
    dataset_train = CNV_AND_SRF_2d5(dataset_path, scale=(args.crop_height, args.crop_width), mode='train')
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True)
    dataset_val = CNV_AND_SRF_2d5(dataset_path, scale=(args.crop_height, args.crop_width), mode='val')
    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=1, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=False)
    dataset_test = CNV_AND_SRF_2d5(dataset_path, scale=(args.crop_height, args.crop_width), mode='test')
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
                #  'resunetplusplus': ResUnetPlusPlus(channel=1),
                #  'cpfnet':CPFNet(),
                #  'resunet1':DeepResUNet(in_channels=args.input_channel),
                 }
    model = model_all[args.net_work].cuda()
    cudnn.benchmark = True

    # 选择优化器
    if(args.optimizer=="SGD"):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif(args.optimizer=="Adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    # 选择loss
    if args.loss == 'NLL_Loss':
        criterion_main = nn.NLLLoss()
    elif args.loss == 'SD_Loss':
        criterion_main = LS.SD_Loss()
    elif args.loss == 'ALL_Loss':
        criterion_main = LS.ALL_Loss()

    criterion = [criterion_main]

    if mode == 'train':  # tv
        train(args, model, optimizer, criterion, dataloader_train, dataloader_val, writer)
    if mode == 'test':  # 单独使用测试集
        eval(args,model, dataloader_test)
    if mode == 'train_test':
        train(args, model, optimizer, criterion, dataloader_train, dataloader_val,writer)
        eval(args, model, dataloader_test)


if __name__ == "__main__":
    seed = 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args = cnv_and_srf_config()
    modes = args.mode
    args.img_format = '2d5'

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
