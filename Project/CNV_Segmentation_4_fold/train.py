import argparse
# from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset.CNV_AND_SRF import CNV_AND_SRF
import socket
from datetime import datetime
import os
from model.unet import UNet
from model.resunet_attention import ResUnetPlusPlus
import torch
from tensorboardX import SummaryWriter
import tqdm
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from PIL import Image
# from utils import poly_lr_scheduler
# from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy,batch_intersection_union,batch_pix_accuracy
import utils.progress_bar as pb
import utils.utils as u
import utils.loss as LS
from utils.config import DefaultConfig
import torch.backends.cudnn as cudnn


def val(args, model, dataloader, k_fold, epoch):
    with torch.no_grad():
        model.eval()
        Dice_m = u.AverageMeter()
        Acc_m = u.AverageMeter()
        jaccard_m = u.AverageMeter()
        Sensitivity_m = u.AverageMeter()
        Specificity_m = u.AverageMeter()
        eval_progressor = pb.Val_ProgressBar(
            mode='val', epoch=epoch+1, fold=k_fold, model_name=args.net_work, net_index=args.net_index, total=len(dataloader))
        for i, (data, label) in enumerate(dataloader):
            eval_progressor.current = i
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            aux_predict, predict = model(data)
            # 获取评价指标
            Dice, Acc, jaccard, Sensitivity, Specificity = u.eval_single_seg(
                predict, label)
            Dice_m.update(Dice)
            Acc_m.update(Acc)
            jaccard_m.update(jaccard)
            Sensitivity_m.update(Sensitivity)
            Specificity_m.update(Specificity)
            dice, acc, jac, sen, spe = Dice_m.avg, Acc_m.avg, jaccard_m.avg, Sensitivity_m.avg, Specificity_m.avg
            eval_progressor.val = [dice, acc, jac, sen, spe]
            # 更新进度条
            eval_progressor()

        eval_progressor.done()
        # print('Dice:', dice)
        # print('Acc:', acc)
        # print('Jac:', jac)
        # print('Sen:', sen)
        # print('Spe:', spe)
        return dice, acc, jac, sen, spe


def train(args, model, optimizer, criterion, dataloader_train, dataloader_val, writer, k_fold):
    best_pred, best_acc, best_jac, best_sen, best_spe = 0.0, 0.0, 0.0, 0.0, 0.0
    best_epoch = 0
    step = 0
    train_loss = u.AverageMeter()
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    with open("./logs/%s_%s.txt" % (args.net_work, args.net_index), "a") as f:
        print(current_time, file=f)
    for epoch in range(args.num_epochs):
        train_progressor = pb.Train_ProgressBar(mode='train', fold=k_fold, epoch=epoch, total_epoch=args.num_epochs,
                                                model_name=args.net_work, total=len(dataloader_train)*args.batch_size)
        lr = u.adjust_learning_rate(args, optimizer, epoch)
        model.train()

        for i, (data, label) in enumerate(dataloader_train):
            train_progressor.current = i*args.batch_size
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            main_out = model(data)
            # get weight_map
            weight_map = torch.zeros(args.num_classes).cuda()
            for t in range(args.num_classes):
                weight_map[t] = 1/(torch.sum((label == t).float())+1.0)
            loss_aux = F.binary_cross_entropy_with_logits(
                main_out, label, weight=None)
            loss_main = criterion[1](main_out, label)
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
                writer.add_scalar(
                    'Train/loss_step_{}'.format(int(k_fold)), loss,step)
        train_progressor.done()
        writer.add_scalar(
            'Train/loss_epoch_{}'.format(int(k_fold)), float(train_loss.avg), epoch)
        Dice, Acc, jaccard, Sensitivity, Specificity = val(
            args, model, dataloader_val, k_fold, epoch)
        writer.add_scalar(
            'Valid/Dice_val_{}'.format(int(k_fold)), Dice, epoch)
        writer.add_scalar(
            'Valid/Acc_val_{}'.format(int(k_fold)), Acc, epoch)
        writer.add_scalar(
            'Valid/Jac_val_{}'.format(int(k_fold)), jaccard, epoch)
        writer.add_scalar(
            'Valid/Sen_val_{}'.format(int(k_fold)), Sensitivity, epoch)
        writer.add_scalar(
            'Valid/Spe_val_{}'.format(int(k_fold)), Specificity, epoch)

        is_best = Dice > best_pred
        if is_best:
            best_pred = max(best_pred, Dice)
            best_jac = max(best_jac, jaccard)
            best_acc = max(best_acc, Acc)
            best_sen = max(best_sen, Sensitivity)
            best_spe = max(best_spe, Specificity)
            best_epoch = epoch+1
        checkpoint_dir = os.path.join(args.save_model_path, str(k_fold))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_latest_name = os.path.join(
            checkpoint_dir, 'checkpoint_latest.path.tar')
        u.save_checkpoint({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'best_dice': best_pred
        }, best_pred, epoch, is_best, checkpoint_dir, filename=checkpoint_latest_name)
    # 记录该折分割效果最好一次epoch的所有参数
    best_indicator_message = "f{} best pred in Epoch:{}\nDice={} Accuracy={} jaccard={} Sensitivity={} Specificity={}".format(
        k_fold, best_epoch,best_pred,best_acc, best_jac, best_sen, best_spe)
    with open("./logs/%s_%s_best_indicator.txt" % (args.net_work, args.net_index), mode='a') as f:
        print(best_indicator_message,file=f)


def eval(model, dataloader, args):
    print('start test!')
    with torch.no_grad():
        model.eval()
        tq = tqdm.tqdm(total=len(dataloader)*args.batch_size)
        tq.set_description('test')
        comments = os.getcwd().split(os.sep)[-1]
        for i, (data, label_path) in enumerate(dataloader):
            tq.update(args.batch_size)
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
            aux_pred, predict = model(data)
            predict = torch.round(torch.sigmoid(aux_pred)).byte()
            pred_seg = predict.data.cpu().numpy() * 255

            for index, item in enumerate(label_path):
                save_img_path = label_path[index].replace(
                    'mask', comments+'_mask')
                if not os.path.exists(os.path.dirname(save_img_path)):
                    os.makedirs(os.path.dirname(save_img_path))
                img = Image.fromarray(pred_seg[index].squeeze(), mode='L')
                img.save(save_img_path)
                tq.set_postfix(str=str(save_img_path))
        tq.close()


def main(mode='train', args=None, writer=None, k_fold=1):
    # create dataset and dataloader
    dataset_path = os.path.join(args.data, args.dataset)
    dataset_train = CNV_AND_SRF(dataset_path, scale=(
        args.crop_height, args.crop_width), k_fold_test=k_fold, mode='train')
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dataset_val = CNV_AND_SRF(dataset_path, scale=(
        args.crop_height, args.crop_width), k_fold_test=k_fold, mode='val')
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dataset_test = CNV_AND_SRF(dataset_path, scale=(
        args.crop_height, args.crop_width), k_fold_test=k_fold, mode='test')
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=True, drop_last=True)
    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    # load model
    model_all = {'UNet': UNet(
        in_channels=args.input_channel, n_classes=args.num_classes),'ResUnetPlusPlus':ResUnetPlusPlus(channel=args.input_channel)}
    model = model_all[args.net_work].cuda()
    cudnn.benchmark = True
    # if torch.cuda.is_available() and args.use_gpu:
    #     model = torch.nn.DataParallel(model).cuda()
    if args.pretrained_model_path and model == 'test':
        print("=> loading pretrained model '{}'".format(
            args.pretrained_model_path))
        checkpoint = torch.load(args.pretrained_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print('Done!')
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion_aux = nn.BCEWithLogitsLoss(weight=None)
    criterion_main = LS.DiceLoss()
    criterion = [criterion_aux, criterion_main]
    if mode == 'train':  # 交叉验证
        train(args, model, optimizer, criterion,
              dataloader_train, dataloader_val, writer, k_fold)
    if mode == 'test':  # 单独使用测试集
        eval(model, dataloader_test, args)


if __name__ == "__main__":
    seed = 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args = DefaultConfig()
    modes = args.mode

    if modes == 'train':
        comments = os.getcwd().split(os.sep)[-1]
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(args.log_dirs, comments +
                               '_' + current_time + '_' + socket.gethostname())
        # print(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        for i in range(args.start_fold-1,args.k_fold):
            main(mode='train', args=args, writer=writer, k_fold=int(i + 1))
    elif modes == 'test':
        main(mode='test', args=args, writer=None, k_fold=args.test_fold)
