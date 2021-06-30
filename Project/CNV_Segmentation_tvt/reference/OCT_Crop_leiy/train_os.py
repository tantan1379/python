import argparse
import socket
import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
import utils.utils as u
import utils.loss as LS
import torch.backends.cudnn as cudnn
import torchsummary
import lib.transforms as transforms
import math
import torch.nn.parallel
import torch.distributed as dist
import multiprocessing
import cv2
import time

from torch.utils.data import DataLoader
from datetime import datetime
from data import *
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from utils.config import Model_choice
from utils.utils import *
from utils.loss import *
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data.distributed import DistributedSampler
from data_os import SegOCT




from model.mini_unet_v0 import Mini_Unet_V0
from model.baseline_v0 import Baseline_V0
from model.nestedunet_v0 import NestedUNet_V0
from model.mini_unet_dcm_v0 import Mini_Unet_Dcm_V0
from model.mini_unet_dcm_v1 import Mini_Unet_Dcm_V1
from model.mini_unet_dcm_v2 import Mini_Unet_Dcm_V2



'''
    指标评估
'''
def val(args, model, dataloader):
    print('\n')
    print('#-------------------------Start Validation!-------------------------#')
    model.eval()
    with torch.no_grad():
        tbar = tqdm(dataloader, desc='\r')

        total_Dice_mh = []
        total_Acc_mh = []
        total_Jaccard_mh = []
        total_Sensitivity_mh = []
        total_Specificity_mh = []

        total_Dice_cme = []
        total_Acc_cme = []
        total_Jaccard_cme = []
        total_Sensitivity_cme = []
        total_Specificity_cme = []

        cur_predict_cube = []
        cur_label_cube = []
        counter = 0
        end_flag = False


        for i, (image,label) in enumerate(tbar):
            # tbar.update()
            H, W = image.shape[2], image.shape[3]
            if torch.cuda.is_available() and args.use_gpu:
                image = image.cuda()  # [b,3,864,512]
                label = label.cuda()  # [b,864,512]

            slice_num = args.slice_num

            # get RGB predict image
            predicts = model(image)
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
                [Dice_cme, Dice_mh], [Acc_cme, Acc_mh], [Jaccard_cme, Jaccard_mh], \
                [Sensitivity_cme, Sensitivity_mh], [Specificity_cme, Specificity_mh] = u.eval_multi_seg(predict_cube,label_cube,args.num_classes)

                total_Dice_mh.append(Dice_mh)
                total_Dice_cme.append(Dice_cme)
                total_Acc_mh.append(Acc_mh)
                total_Acc_cme.append(Acc_cme)
                total_Jaccard_mh.append(Jaccard_mh)
                total_Jaccard_cme.append(Jaccard_cme)
                total_Sensitivity_mh.append(Sensitivity_mh)
                total_Sensitivity_cme.append(Sensitivity_cme)
                total_Specificity_mh.append(Specificity_mh)
                total_Specificity_cme.append(Specificity_cme)

                dice_mh = sum(total_Dice_mh) / len(total_Dice_mh)
                dice_cme = sum(total_Dice_cme) / len(total_Dice_cme)
                acc_mh = sum(total_Acc_mh) / len(total_Acc_mh)
                acc_cme = sum(total_Acc_cme) / len(total_Acc_cme)
                jac_mh = sum(total_Jaccard_mh) / len(total_Jaccard_mh)
                jac_cme = sum(total_Jaccard_cme) / len(total_Jaccard_cme)
                sen_mh = sum(total_Sensitivity_mh) / len(total_Sensitivity_mh)
                sen_cme = sum(total_Sensitivity_cme) / len(total_Sensitivity_cme)
                spe_mh = sum(total_Specificity_mh) / len(total_Specificity_mh)
                spe_cme = sum(total_Specificity_cme) / len(total_Specificity_cme)

                tbar.set_description(
                            'Dice_mh: %.4f, Dice_cme: %.4f' % (dice_mh, dice_cme))

        return dice_mh, acc_mh, jac_mh, sen_mh, spe_mh, dice_cme, acc_cme, jac_cme, sen_cme, spe_cme

def train(args, model, optimizer, criterion, dataloader_train, dataloader_val, writer, k_fold,logger_test):
    step = 0
    best_pred = 0.0
    for epoch in range(args.num_epochs):
        if args.lr_mode == 'poly':
            lr = u.adjust_learning_rate(args, optimizer, epoch)

        model.train()
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('fold %d,epoch %d' % (int(k_fold), epoch))
        loss_record = []
        train_loss = 0.0

        for i, (image,label) in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
                image = image.cuda()
                label = label.cuda().long()


            main_out = model(image)
            loss = criterion[0](main_out, label)
            loss_record.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tq.update(args.batch_size)
            train_loss += loss.item()

            tq.set_postfix(loss='%.6f' % (train_loss / (i + 1)))
            step += 1

            if step % 10 == 0:
                writer.add_scalar('Train/loss_step_{}'.format(int(k_fold)), loss, step)
            loss_record.append(loss.item())

        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('Train/loss_epoch_{}'.format(int(k_fold)), float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))

        if epoch % args.validation_step == 0:
            Dice_mh, Acc_mh, jaccard_mh, Sensitivity_mh, Specificity_mh, \
            Dice_cme, Acc_cme, jaccard_cme, Sensitivity_cme, Specificity_cme = val(args, model, dataloader_val)

            writer.add_scalar('Valid/Dice_mh_val_{}'.format(int(k_fold)), Dice_mh, epoch)
            writer.add_scalar('Valid/Acc_mh_val_{}'.format(int(k_fold)), Acc_mh, epoch)
            writer.add_scalar('Valid/Jac_mh_val_{}'.format(int(k_fold)), jaccard_mh, epoch)
            writer.add_scalar('Valid/Sen_mh_val_{}'.format(int(k_fold)), Sensitivity_mh, epoch)
            writer.add_scalar('Valid/Spe_mh_val_{}'.format(int(k_fold)), Specificity_mh, epoch)

            writer.add_scalar('Valid/Dice_cme_val_{}'.format(int(k_fold)), Dice_cme, epoch)
            writer.add_scalar('Valid/Acc_cme_val_{}'.format(int(k_fold)), Acc_cme, epoch)
            writer.add_scalar('Valid/Jac_cme_val_{}'.format(int(k_fold)), jaccard_cme, epoch)
            writer.add_scalar('Valid/Sen_cme_val_{}'.format(int(k_fold)), Sensitivity_cme, epoch)
            writer.add_scalar('Valid/Spe_cme_val_{}'.format(int(k_fold)), Specificity_cme, epoch)

            writer.add_scalar('Valid/Dice_avg_val_{}'.format(int(k_fold)), (Dice_mh+Dice_cme) / 2, epoch)
            writer.add_scalar('Valid/Acc_avg_val_{}'.format(int(k_fold)),  (Acc_mh + Acc_cme) / 2, epoch)
            writer.add_scalar('Valid/Jac_avg_val_{}'.format(int(k_fold)), (jaccard_mh+ jaccard_cme) / 2 , epoch)
            writer.add_scalar('Valid/Sen_avg_val_{}'.format(int(k_fold)), (Sensitivity_mh+Sensitivity_cme) / 2, epoch)
            writer.add_scalar('Valid/Spe_avg_val_{}'.format(int(k_fold)), (Specificity_mh+Specificity_cme) / 2, epoch)

            Dice = (Dice_cme + Dice_mh) / 2

            is_best = Dice > best_pred
            best_pred = max(best_pred, Dice)

            if is_best:
                torch.save(model,args.model_save_net_path  + str(k_fold) + '/net.pkl')
                torch.save(model.state_dict(), args.model_load_para_path + str(k_fold) +'/net_params.pkl')



def eval(model, dataloader, args,k_fold):
    print('#------------------------------ Run! -------------------------------#')
    print('start eval !!!')
    print('#-------------------------------------------------------------------#')

    with torch.no_grad():
        model.eval()
        tq = tqdm(total=len(dataloader) * args.batch_size)
        tq.set_description('eval')

        file_list = os.listdir(args.data_load_path + 'data_image/f' + str(k_fold))

        for file in file_list:
            tmp_path = os.path.join(args.image_save_path, 'f' + str(k_fold), file)
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)

            tmp_path = os.path.join(args.image_save_path.replace('logs','predict'), 'f' + str(k_fold), file)
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)

        # image_save_name
        image_save_name_list = []
        image_root_path = os.path.join(args.data_load_path,'data_image','f' + str(k_fold))
        for dir in sorted(os.listdir(image_root_path)):
            tmp_name_list = sorted(sorted(glob.glob(os.path.join(image_root_path,dir) + '/*png')),key=lambda i: len(i))
            image_save_name_list += tmp_name_list

        image_save_name_list = [x.replace('data_crop/data_image/','Source/Source/logs/' + args.net_work + '/') for x in image_save_name_list]
        predict_save_name_list = [x.replace('logs','predict') for x in image_save_name_list]

        print(image_save_name_list[0])
        print(predict_save_name_list[0])

        for i, (image,label) in enumerate(dataloader):
            tq.update(args.batch_size)
            if torch.cuda.is_available() and args.use_gpu:
                image = image.cuda()

            predicts = model(image)
            predicts = torch.argmax(torch.exp(predicts), dim = 1) # []
            image_merge(image,label,predicts,k_fold,args,i,image_save_name_list,predict_save_name_list)

        tq.close()

def train_eval(model, dataloader, args,k_fold):
    print('#------------------------------ Run! -------------------------------#')
    print('start train_eval !!!')
    print('#-------------------------------------------------------------------#')

    fold = [i + 1 for i in range((args.k_fold))]
    fold.remove(k_fold)

    with torch.no_grad():
        model.eval()
        tq = tqdm(total = (dataloader._size))
        tq.set_description('train_eval')


        # image_save_name
        image_save_name_list = sorted(sorted(glob.glob(os.path.join(args.data_load_path,'data_image','f'+str(fold[0])) + '/*.png')),key=lambda i: len(i))

        for i, (image, label) in enumerate(dataloader):
            tq.update(args.batch_size)
            if torch.cuda.is_available() and args.use_gpu:
                pass

            predict_cme,predict_mh = model(image)
            predict = torch.cat([predict_cme,predict_mh],dim=1)

            image_merge(image,label,predict,k_fold,args,i,image_save_name_list)
        tq.close()


def main(mode='train', args=None, writer=None, k_fold=1, device = None, local_rank = None):

    # 固定随机种子
    seed = 520
    torch.manual_seed(seed)  # cpu种子
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证实验结果能够复现
    torch.backends.cudnn.benchmark = True

    # 事件记录模块
    logger_test  = result_log(logger_name='test',  file_name=args.logger_test_path + '/logs_test_'  +
                                                             str(k_fold) + '.log', delete=True)

    dataset_path = args.data_load_path

    dataset_train = SegOCT(dataset_path, mode='train', label_number = "label_1",k_fold_test = k_fold)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    dataset_val = SegOCT(dataset_path, mode='val', label_number="label_1", k_fold_test = k_fold)
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    dataset_test = SegOCT(dataset_path, mode='test', label_number="label_1", k_fold_test = k_fold)
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    dataset_train_eval = SegOCT(dataset_path, mode='train_eval', label_number="label_1",k_fold_test = k_fold)
    dataloader_train_eval = DataLoader(
        dataset_train_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    model_all = {
                 # 'Test_V0': Test_V0(in_channels=args.input_channel, n_classes=args.num_classes),
                 # 'Test_V1': Test_V1(in_channels=args.input_channel, n_classes=args.num_classes),
                 # 'Mini_Unet_V0': Mini_Unet_V0(in_channels=args.input_channel, n_classes=args.num_classes),
                 # 'NestedUNet_V0': NestedUNet_V0(input_channels=3,deepsupervision=False),
                 # 'Mini_Unet_Dcm_V0': Mini_Unet_Dcm_V0(in_channels=args.input_channel, n_classes=args.num_classes),
                 # 'Mini_Unet_Dcm_V1': Mini_Unet_Dcm_V1(in_channels=args.input_channel, n_classes=args.num_classes),
                 'Mini_Unet_Dcm_V2': Mini_Unet_Dcm_V2(in_channels=args.input_channel, n_classes=args.num_classes),

                 # 'Baseline_V0': Baseline_V0(in_channels=args.input_channel, n_classes=args.num_classes),
                }

    model = model_all[args.net_work]
    torchsummary.summary(model, (1, 512, 512),device='cpu') # 输出模型的参数量


    # 大部分情况下，设置这MyDataset个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    # cudnn.benchmark = True
    if torch.cuda.is_available() and args.use_gpu:
        model.cuda()

    # 保存初始权重
    checkpoint_dir_root = args.model_load_para_path
    checkpoint_dir = os.path.join(checkpoint_dir_root, str(k_fold))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # load pretrained model if exists
    if args.load_parameters == True or mode == 'test' or mode == 'train_eval':
        # 预加载模型
        print("==> loading pretrained model '{}'".format(args.model_load_para_path + str(k_fold) + '/net_params.pkl'))
        model.load_state_dict(torch.load(args.model_load_para_path + str(k_fold) +'/net_params.pkl',map_location='cpu'))
        print('Done!')

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),  lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 选择loss
    if args.loss == 'NLL_Loss':
        criterion_main = nn.NLLLoss()
    elif args.loss == 'SD_Loss':
        criterion_main = SD_Loss()
    elif args.loss == 'ALL_Loss':
        criterion_main = ALL_Loss()

    criterion = [criterion_main]

    if mode == 'train':
        train(args, model, optimizer, criterion, dataloader_train, dataloader_val, writer, k_fold,logger_test)
    if mode == 'test':
        eval(model, dataloader_test, args,k_fold)
    if mode == 'train_test':
        train(args, model, optimizer, criterion, dataloader_train, dataloader_val)
        eval(model, dataloader_test, args)
    if mode == 'train_eval':
        train_eval(model, dataloader_train_eval, args, k_fold)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('Task', type=str, help='display an task')
    parser.add_argument('--fold', type = int, help='display an fold')
    arg = parser.parse_args()

    args = Model_choice(arg.Task)
    modes = args.mode

    # args = Model_choice()
    # modes = args.mode
    # build model
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    print('#------------------------------ Run! -------------------------------#')
    print('This is a model:%s !!!'%(args.net_work))
    print('#-------------------------------------------------------------------#')

    # 创建文件夹
    os_makedir(args.tensorboard_path,args.model_load_para_path,args.logger_train_path,args.image_save_path)

    if modes == 'train':
        writer = SummaryWriter(log_dir=args.tensorboard_path)
        if arg.fold == 3:
            for i in range(args.k_fold): # 自动交叉验证
                main(mode='train', args=args, writer=writer, k_fold=int(i+1))
        else:
            main(mode='train', args=args, writer=writer, k_fold=3)
    elif modes == 'test':
        main(mode='test', args=args, writer=None, k_fold=args.test_fold)
    elif modes == 'train_eval':
        main(mode='train_eval', args=args, writer=None, k_fold=args.test_fold)







