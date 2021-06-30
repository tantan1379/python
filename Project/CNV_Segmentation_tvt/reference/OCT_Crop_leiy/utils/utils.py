import numpy as np
import glob
import cv2
import os
import logging
import sys
import copy
import torch
import shutil
import copy
import torch.nn as nn


from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.nn import functional as F



#------------------------------------------------   对最终结果进行数据统计   ------------------------------------------------#
'''
    创建相应文件路径
'''
def os_makedir(tensorboard_path,model_save_path,logs_save_path,image_save_path):

    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(logs_save_path):
        os.makedirs(logs_save_path)
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

def save_checkpoint(state,best_pred, epoch,is_best,checkpoint_path,filename='./checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, osp.join(checkpoint_path,'model_{:03d}_{:.4f}.pth.tar'.format((epoch + 1),best_pred)))

def adjust_learning_rate(opt, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    if opt.lr_mode == 'step':
        lr = opt.lr * (0.1 ** (epoch // opt.step))
    elif opt.lr_mode == 'poly':
        lr = opt.lr * (1 - epoch / opt.num_epochs) ** 0.9
        sge_lr = 1e-2 * (1 - epoch / opt.num_epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(opt.lr_mode))

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = sge_lr
    # optimizer.param_groups[0]['lr'] = lr

    # for param_group in optimizer.param_groups:
    #     print(param_group['lr'])

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


'''
    文件名索引
'''
def indexread(path):
    result = []
    fd = open('/home/leiy/DLProjects/fundus/'+ path )
    for line in fd.readlines():
        result.append(line.split('\n')[0])
    result = np.array(result)
    return result



#########################################   评估指标 &  损失函数   ###########################################
def compute_score_single(predict, target, forground=1, smooth=1):
    score = 0
    count = 0
    target[target != forground] = 0
    predict[predict != forground] = 0
    assert (predict.shape == target.shape)
    overlap = ((predict == forground) * (target == forground)).sum()  # TP
    union = (predict == forground).sum() + (target == forground).sum() - overlap  # FP+FN+TP
    FP = (predict == forground).sum() - overlap  # FP
    FN = (target == forground).sum() - overlap  # FN
    TN = target.shape[0] * target.shape[1] * target.shape[2] - union  # TN

    # print('overlap:',overlap)
    dice = (2 * overlap + smooth) / (union + overlap + smooth)

    precsion = ((predict == target).sum() + smooth) / (target.shape[0] * target.shape[1] * target.shape[2] + smooth)

    jaccard = (overlap + smooth) / (union + smooth)

    Sensitivity = (overlap + smooth) / ((target == forground).sum() + smooth)

    Specificity = (TN + smooth) / (FP + TN + smooth)

    return dice, precsion, jaccard, Sensitivity, Specificity

def eval_single_seg(predict, target, forground=1):
    pred_seg = torch.round((predict)).int()
    pred_seg = pred_seg.data.cpu().numpy()
    label_seg = target.data.cpu().numpy().astype(dtype=np.int)
    assert (pred_seg.shape == label_seg.shape)

    Dice = []
    Precsion = []
    Jaccard = []
    Sensitivity = []
    Specificity = []

    n = pred_seg.shape[0]

    for i in range(n):
        dice, precsion, jaccard, sensitivity, specificity = compute_score_single(pred_seg[i], label_seg[i])
        Dice.append(dice)
        Precsion.append(precsion)
        Jaccard.append(jaccard)
        Sensitivity.append(sensitivity)
        Specificity.append(specificity)

    return Dice, Precsion, Jaccard, Sensitivity, Specificity


def eval_multi_seg(predict, target, num_classes):
    # pred_seg=torch.argmax(torch.exp(predict),dim=1).int()
    smooth = 0.1
    pred_seg = predict.data.cpu().numpy()
    label_seg = target.data.cpu().numpy().astype(dtype=np.int)
    assert (pred_seg.shape == label_seg.shape) # [b,32, 128,128]


    Dice = []
    Acc = []
    Jaccard = []
    Sensitivity=[]
    Specificity=[]

    # n = pred_seg.shape[0]
    Dice = []
    for classes in range(1, num_classes):
        overlap = ((pred_seg == classes) * (label_seg == classes)).sum()
        union = (pred_seg == classes).sum() + (label_seg == classes).sum()

        FP = (pred_seg == classes).sum() - overlap  # FP
        FN = (label_seg == classes).sum() - overlap  # FN
        TN = label_seg.shape[0] * label_seg.shape[1] * label_seg.shape[2] - union + overlap # TN

        dice = (2 * overlap + 0.1) / (union + 0.1)
        acc = ((pred_seg == label_seg).sum() + smooth) / (label_seg.shape[0] * label_seg.shape[1] * label_seg.shape[2] + smooth)
        jaccard = (overlap + smooth) / (union + smooth - overlap)
        sensitivity = (overlap + smooth) / ((label_seg == classes).sum() + smooth)
        specificity = (TN + smooth) / (FP + TN + smooth)

        Dice.append(dice)
        Acc.append(acc)
        Jaccard.append(jaccard)
        Sensitivity.append(sensitivity)
        Specificity.append(specificity)

    return Dice, Acc, Jaccard, Sensitivity, Specificity


def dice_coef(y_true,y_pred):
    smooth = 1.
    dice_sum = 0

    for items in range(y_true.shape[0]):
        y_true_f = y_true[items]
        y_pred_f = y_pred[items]

        y_true_f = y_true_f.contiguous().view(-1)
        y_pred_f = y_pred_f.contiguous().view(-1)

        intersection = (y_true_f * y_pred_f).sum()
        dice = (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)
        dice = float(dice.data.cpu().numpy())

        dice_sum = dice_sum + dice

    return dice_sum



def dice_coef_loss(y_true, y_pred):

    return 1 - dice_coef(y_true, y_pred)


def evaluation_index(y_true,y_pred):
    smooth = 1
    sensitivity_sum = 0.
    specificity_sum = 0.
    precision_sum = 0.
    dice_sum = 0.

    for items in range(y_true.shape[0]):
        y_true_f = y_true[items]
        y_pred_f = y_pred[items]

        y_true_f = y_true_f.contiguous().view(-1)
        y_pred_f = y_pred_f.contiguous().view(-1)

        # 计算敏感度
        TP = (y_true_f * y_pred_f).sum()
        TP_FN = y_true_f.sum()
        sensitivity = (TP / (TP_FN + smooth))

        # 计算特异性
        TN = ((1-y_true_f) * (1 - y_pred_f)).sum()
        TN_FP = (1 - y_true_f).sum()
        specificity = (TN / (TN_FP + smooth))

        # 计算precision
        TP_FP = y_pred_f.sum()
        precision = (TP / (TP_FP + smooth))

        # 计算戴斯
        intersection = (y_true_f * y_pred_f).sum()
        dice = (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)
        dice = float(dice.data.cpu().numpy())
        dice_sum = dice_sum + dice

        sensitivity_sum = sensitivity_sum + sensitivity
        specificity_sum = specificity_sum + specificity
        precision_sum = precision_sum + precision

    sensitivity_sum = sensitivity_sum.cpu().numpy()
    specificity_sum = specificity_sum.cpu().numpy()
    precision_sum = precision_sum.cpu().numpy()


    return sensitivity_sum,specificity_sum,precision_sum,dice_sum


'''
    结果记录
'''

'''
    训练结果记录
'''

def get_logger(logger_name='Unet', file_name='logs.log'):

    logger = logging.getLogger(logger_name) # 实例化logger 使用接口debug,info,warn,error,critical之前必须创建Logger实例
    logger.setLevel(logging.DEBUG) # 设置日志级别 只有级别大于DEBUG日志才会输出

    fh = logging.FileHandler(file_name) #实例化Handler属性
    fh.setLevel(logging.DEBUG) #设置日志级别 低于WARN级别日志将会被忽视

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO) #设置日志级别 低于INFO级别日志将会被忽视

    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter) # 设置一个格式化器formatter
    ch.setFormatter(formatter) # 设置一个格式化器formatter

    logger.addHandler(fh) #为logger实例增加一个处理器
    logger.addHandler(ch) #为logger实例增加一个处理器

    return logger



def result_log(logger_name = 'Unet',file_name = 'logs.log',delete = True):
    if delete == True:
        if os.path.exists(file_name):
            os.remove(file_name)
        logger = get_logger(logger_name = logger_name,file_name = file_name)
        logger.info("--------------------------------------------")
    else:
        logger = get_logger(logger_name = logger_name, file_name = file_name)
        logger.info("--------------------------------------------")

    return logger















########################################   图像的预处理部分   ########################################


'''
    制作出一张fundus图片的mask
'''
def fundus_image_mask(train_dir):
    # 随机不同的图片路径
    path_image = os.path.join(train_dir,'image')
    path_image_all = sorted(sorted(glob.glob(path_image + '/*.png')),key = lambda i:len(i))

    for items in range(len(path_image_all)):
        images = cv2.imread(path_image_all[items])
        gray = cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 17 , 255, cv2.THRESH_BINARY)

        cv2.imwrite('IDRiD_'+str(items+1)+'_mask.png',binary)

        # cv2.imshow('binary',binary)

        # _,contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(images,contours,-1,(0,0,255),1)

        # cv2.imshow('image',binary)
        # cv2.waitKey()

'''
    使用生成的mask对原来的图像进行过滤
'''
def fundus_image_filter(train_dir):
    # 随机不同的图片路径
    path_image = os.path.join(train_dir,'image')
    path_image_all = sorted(sorted(glob.glob(path_image + '/*.png')),key = lambda i:len(i))

    path_image_mask = os.path.join(train_dir,'mask')
    path_image_mask_all = sorted(sorted(glob.glob(path_image_mask + '/*.png')),key = lambda i:len(i))

    for items in range(len(path_image_all)):
        images = cv2.imread(path_image_all[items])
        mask   = cv2.imread(path_image_mask_all[items])

        images = images / 255
        mask = mask / 255

        images = np.where(mask > 0,images,mask)

        cv2.imwrite('IDRiD_' + str(items + 55) +'.png',images * 255)

        # images = cv2.resize(images,(1072, 712))
        # cv2.imshow('image',images)
        # cv2.waitKey()




def train_label_cvtgray(train_dir):

    path_image = os.path.join(train_dir,'label','Haemorrhages')
    path_image_all = sorted(sorted(glob.glob(path_image + '/*.tif')),key = lambda i:len(i))

    for items in range(len(path_image_all)):
        images = cv2.imread(path_image_all[items])
        gray = cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)

        cv2.imwrite('IDRiD_' + str(items + 1) + '_EX.png', binary)


        # cv2.imshow('image',binary)
        # cv2.waitKey()

'''
    将图片进行裁剪
'''
def image_crop_resize(path):

    path_image = os.path.join(path,'image')
    path_image_all = sorted(sorted(glob.glob(path_image + '/*.png')),key = lambda i:len(i))
    path_label = os.path.join(path, 'label','Haemorrhages')
    path_label_all = sorted(sorted(glob.glob(path_label +'/*.png')), key=lambda i: len(i))
    path_mask = os.path.join(path,'mask')
    path_mask_all = sorted(sorted(glob.glob(path_mask + '/*.png')), key=lambda i: len(i))

    for item in tqdm(range(len(path_image_all))):

        images = np.array(cv2.imread(path_image_all[item]))
        labels = np.array(cv2.imread(path_label_all[item]))
        masks  = np.array(cv2.imread(path_mask_all[item], cv2.IMREAD_GRAYSCALE))


        # 寻找最左边与最右边的点
        masks = masks / 255

        masks_y_min = float('inf')
        masks_y_max = 0

        for x in range(masks.shape[0]):
            for y in range(masks.shape[1]):
                if masks[x,y] == 1:
                    if masks_y_min > y:
                        masks_y_min = y
                    if masks_y_max < y:
                        masks_y_max = y
        # 寻找最中心点的坐标
        masks_center = int((masks_y_min + masks_y_max) / 2)

        # crop
        masks = masks[:,masks_center - 1899:masks_center + 1899]
        images = images[:,masks_center - 1899:masks_center + 1899,:]
        labels = labels[:,masks_center - 1899:masks_center + 1899,:]


        # 加载图像
        cv2.imwrite('/home/leiy/DLProjects/fundus/data_2/train/mask/' +'IDRiD_'+str(item+1)+'_mask.png',masks * 255)
        cv2.imwrite('/home/leiy/DLProjects/fundus/data_2/train/image/'+'IDRiD_'+str(item+1)+'.png',images)
        cv2.imwrite('/home/leiy/DLProjects/fundus/data_2/train/label/'+'IDRiD_'+str(item+1) + '_EX.png',labels)


'''
    对图像进行增强
'''
def Illumination_balance_channel(image_channel,block_size):

    # 原图的平均灰度值
    global_mean_gray = np.mean(image_channel)

    # 获取块的数目
    row = int(np.ceil(image_channel.shape[0] / block_size[0]))
    col = int(np.ceil(image_channel.shape[1] / block_size[0]))

    #
    image_E = np.zeros((row,col))
    image_R = np.zeros((image_channel.shape[0],image_channel.shape[1]))

    for i in range(row):
        for j in range(col):
            row_min = i * block_size[0]
            row_max = (i + 1) * block_size[0]
            col_min = j * block_size[1]
            col_max = (j + 1) * block_size[1]

            if row_max > image_channel.shape[0]:
                row_max = image_channel.shape[0]
            if col_max > image_channel.shape[1]:
                col_max = image_channel.shape[1]

            image_crop = image_channel[row_min:row_max,col_min:col_max]

            # 一个block的平均灰度值
            image_crop_mean_gray = np.mean(image_crop)

            # 子块亮度差值
            image_crop_mean_gray = image_crop_mean_gray - global_mean_gray

            image_E[i,j] = image_crop_mean_gray

    image_R = cv2.resize(image_E,(image_channel.shape[1],image_channel.shape[0]))
    result = image_channel - image_R

    # 对阀值进行限定
    result = np.where(result>0,result,0)
    result = np.where(result>255,255,result)

    return result


def Illumination_balance(image,mask,filter_size):
    mask = cv2.resize(mask, (1024, 768))
    mask = np.where(mask > 127, 255, 0).astype(np.uint8)


    # 分离图像
    image_b, image_g, image_r = cv2.split(image)

    image_result_b = Illumination_balance_channel(image_b, [64, 64])
    image_result_g = Illumination_balance_channel(image_g, [64, 64])
    image_result_r = Illumination_balance_channel(image_r, [64, 64])

    image_result = cv2.merge((image_result_b, image_result_g, image_result_r))
    image_result = image_result.astype(np.uint8)

    image_result = image_result / 255
    mask = mask / 255
    image_result = np.where(mask > 0, image_result, mask)


    # 高斯滤波
    image_result = cv2.GaussianBlur(image_result, (3, 3), 0)
    image_result = image_result * 255
    image_result = image_result.astype(np.uint8)

    image_lab = cv2.cvtColor(image_result, cv2.COLOR_BGR2Lab)

    image_L, image_A, image_B = cv2.split(image_lab)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(60, 60))
    image_L = clahe.apply(image_L)
    result = cv2.merge((image_L, image_A, image_B))
    image_result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


    return image_result


'''
    对图像进行合并
'''

# def image_merge(image_original,label_original,predict_original,k_fold,args,i,image_save_name_list,predict_mh_save_name_list,predict_cme_save_name_list):
#
#     fold = [i+1 for i in range((args.k_fold))]
#     fold.remove(k_fold)
#
#     # 先做一次深拷贝
#     images  = copy.deepcopy(image_original)
#     labels  = copy.deepcopy(label_original)
#     predicts = copy.deepcopy(predict_original)
#
#     # 对图像进行预处理
#     # 0-1 => 0-2550
#     images  = (images.data.cpu().numpy() * 255).astype(np.uint8)[:,1,:,:]     # [2, 864, 512] 只取中间一张
#     labels  = (labels.data.cpu().numpy()).astype(np.uint8)              # [2, 864, 512]
#     # print(np.unique(labels))
#     predicts = (predicts.data.cpu().numpy()).astype(np.uint8)           # [2, 864, 512]
#     # print(np.unique(predicts))
#
#     # images   = np.transpose(images,(0,2,3,1))
#     # labels   = np.transpose(labels,(0,2,3,1))
#     # predicts = np.transpose(predicts,(0,2,3,1))
#
#     if args.mode == 'test':
#         dir = os.path.join(args.image_save_path, 'f' + str(k_fold))
#         if not os.path.exists(dir):
#             os.makedirs(dir)
#     else:
#         dir = os.path.join(args.image_save_path, str(fold[0]) + '+' + str(fold[1]))
#         if not os.path.exists(dir):
#             os.makedirs(dir)
#
#     for items in range(images.shape[0]):
#         image_overlap   = images[items]
#         label   = labels[items]
#         predict = predicts[items]
#
#         predict_mh = np.where(predict == 2, 255, 0).astype(np.uint8)
#         predict_cme = np.where(predict == 1,255,0).astype(np.uint8)
#
#         label_mh = np.where(label == 2, 255, 0).astype(np.uint8)
#         label_cme= np.where(label == 1, 255, 0).astype(np.uint8)
#
#         # cv2.imshow('image_overlap', image_overlap)
#         # cv2.imshow('label_mh',label_mh)
#         # cv2.imshow('label_cme',label_cme)
#         # cv2.imshow('predict_mh',predict_mh)
#         # cv2.imshow('predict_cme',predict_cme)
#         # cv2.waitKey()
#
#         # cvtColor
#         image_overlap = cv2.cvtColor(image_overlap, cv2.COLOR_GRAY2BGR)
#         label_mh   = cv2.cvtColor(label_mh,cv2.COLOR_GRAY2BGR)
#         label_cme  = cv2.cvtColor(label_cme,cv2.COLOR_GRAY2BGR)
#         predict_mh = cv2.cvtColor(predict_mh,cv2.COLOR_GRAY2BGR)
#         predict_cme = cv2.cvtColor(predict_cme,cv2.COLOR_GRAY2BGR)
#
#         # 压和label_mh  Blue
#         condition = label_mh[:, :, 0] > 127
#         image_overlap[:,:,0] = np.where(condition,255,image_overlap[:,:,0])
#         image_overlap[:,:,1] = np.where(condition,0,image_overlap[:,:,1])
#         image_overlap[:,:,2] = np.where(condition,0,image_overlap[:,:,2])
#
#         # 压和label_cme Green
#         condition = label_cme[:, :, 0] > 127
#         image_overlap[:, :, 0] = np.where(condition, 0, image_overlap[:, :, 0])
#         image_overlap[:, :, 1] = np.where(condition, 255,image_overlap[:, :,1])
#         image_overlap[:, :, 2] = np.where(condition, 0, image_overlap[:, :, 2])
#
#         # # 压和predict_mh
#         condition = predict_mh[:, :, 0] > 127
#         image_overlap[:,:,1] = np.where(condition,0,image_overlap[:,:,1])
#         image_overlap[:,:,2] = np.where(condition,255,image_overlap[:,:,2])
#
#         # predict_cme
#         condition = predict_cme[:, :, 0] > 127
#         image_overlap[:,:,0] = np.where(condition,255,image_overlap[:,:,0])
#         image_overlap[:,:,2] = np.where(condition,0,image_overlap[:,:,0])
#
#         predict_mh = np.where(predict_mh > 127, 255, 0).astype(np.uint8)
#         predict_cme = np.where(predict_cme > 127, 255, 0).astype(np.uint8)
#
#         if args.mode == 'test':
#             cv2.imwrite(image_save_name_list[args.batch_size * i  + items], image_overlap)
#             cv2.imwrite(predict_mh_save_name_list[args.batch_size * i  + items], predict_mh)
#             cv2.imwrite(predict_cme_save_name_list[args.batch_size * i  + items], predict_cme)
#         else:
#             cv2.imwrite(image_save_name_list[args.batch_size * i  + items], image_overlap)
#             cv2.imwrite(predict_mh_save_name_list[args.batch_size * i  + items], predict_mh)
#             cv2.imwrite(predict_cme_save_name_list[args.batch_size * i  + items], predict_cme)


def image_merge(image_original,label_original,predict_original,k_fold,args,i,image_save_name_list,predict_save_name_list):

    fold = [i+1 for i in range((args.k_fold))]
    fold.remove(k_fold)

    # 先做一次深拷贝
    images  = copy.deepcopy(image_original)
    labels  = copy.deepcopy(label_original)
    predicts = copy.deepcopy(predict_original)

    # 对图像进行预处理
    # 0-1 => 0-2550
    images  = (images.data.cpu().numpy() * 255).astype(np.uint8)[:,1,:,:]     # [2, 864, 512] 只取中间一张
    labels  = (labels.data.cpu().numpy()).astype(np.uint8)              # [2, 864, 512]
    predicts = (predicts.data.cpu().numpy()).astype(np.uint8)           # [2, 864, 512]

    if args.mode == 'test':
        dir = os.path.join(args.image_save_path, 'f' + str(k_fold))
        if not os.path.exists(dir):
            os.makedirs(dir)
    else:
        dir = os.path.join(args.image_save_path, str(fold[0]) + '+' + str(fold[1]))
        if not os.path.exists(dir):
            os.makedirs(dir)

    for items in range(images.shape[0]):
        image_overlap   = images[items]
        label   = labels[items]
        predict = predicts[items]

        predict_mh = np.where(predict == 2, 255, 0).astype(np.uint8)
        predict_cme = np.where(predict == 1,255,0).astype(np.uint8)

        label_mh = np.where(label == 2, 255, 0).astype(np.uint8)
        label_cme= np.where(label == 1, 255, 0).astype(np.uint8)

        image_mask = np.zeros_like(image_overlap).astype(np.uint8)
        # cvtColor
        image_overlap = cv2.cvtColor(image_overlap, cv2.COLOR_GRAY2BGR)

        condition = (label_mh == 255) * (predict_mh == 255)
        image_mask = np.where(condition, 55, image_mask)
        condition = (label_mh == 255) * (predict_mh == 0)
        image_mask = np.where(condition, 75, image_mask)
        condition = (label_mh == 0) * (predict_mh == 255)
        image_mask = np.where(condition, 125, image_mask)

        condition = (label_cme == 255) * (predict_cme == 255)
        image_mask = np.where(condition, 155, image_mask)
        condition = (label_cme == 255) * (predict_cme == 0)
        image_mask = np.where(condition, 175, image_mask)
        condition = (label_cme == 0) * (predict_cme == 255)
        image_mask = np.where(condition, 225, image_mask)

        image_mask = cv2.cvtColor(image_mask, cv2.COLOR_GRAY2BGR)
        image_overlap = np.where(image_mask == 55,[255,0,255],image_overlap).astype(np.uint8) # 紫色 TP
        image_overlap = np.where(image_mask == 75,[255,0,0],image_overlap).astype(np.uint8)   # 蓝色 FN
        image_overlap = np.where(image_mask == 125,[0,0,255],image_overlap).astype(np.uint8)  # 红色 FP
        image_overlap = np.where(image_mask == 155,[0,255,255],image_overlap).astype(np.uint8) # 黄色 TP
        image_overlap = np.where(image_mask == 175,[0,255,0],image_overlap).astype(np.uint8)   # 绿色 FN
        image_overlap = np.where(image_mask == 225,[255,255,0],image_overlap).astype(np.uint8) # 青色 FP

        predict = np.where(predict == 2, 255, predict).astype(np.uint8)
        predict = np.where(predict == 1, 63, predict).astype(np.uint8)
        if args.mode == 'test':
            cv2.imwrite(image_save_name_list[args.batch_size * i  + items], image_overlap)
            cv2.imwrite(predict_save_name_list[args.batch_size * i  + items], predict)
        else:
            cv2.imwrite(image_save_name_list[args.batch_size * i  + items], image_overlap)
            cv2.imwrite(predict_mh_save_name_list[args.batch_size * i  + items], predict)






'''
    提取图像边缘并进行高斯模糊
'''

def edge_label(path,fold):

    path_new = '/home/leiy/DLProjects/MICCAI/data/data_skin/raw_data'
    path_label = os.path.join(path,'label',fold)
    path_label_all = sorted(sorted(glob.glob(path_label + '/*.png')), key=lambda i: len(i))

    path_label_new = [x.split('/')[-1] for x in path_label_all]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for items in range(len(path_label_all)):

        label = cv2.imread(path_label_all[items])

        # 结构元素匹配

        image_dilation = cv2.dilate(label, kernel)
        image_erode = cv2.erode(label,kernel)

        image = image_dilation - image_erode

        # image = cv2.GaussianBlur(image, (9, 9), 0)

        path = os.path.join(path_new, 'edge_label', fold ,path_label_new[items])
        cv2.imwrite(os.path.join(path_new, 'edge_label', fold ,path_label_new[items]),image)


def grid2contour(path,grid):
    plt.switch_backend('agg')
    '''
    grid--image_grid used to show deform field
    type: numpy ndarray, shape： (h, w, 2), value range：(-1, 1)
    '''
    assert grid.ndim == 3
    x = np.arange(-1, 1, 2 / grid.shape[1])
    y = np.arange(-1, 1, 2 / grid.shape[0])
    X, Y = np.meshgrid(x, y)
    Z1 = grid[:, :, 0] + 2  # remove the dashed line
    Z1 = Z1[::-1]  # vertical flip
    Z2 = grid[:, :, 1] + 2


    plt.figure()
    plt.contour(X, Y, Z1, 15, colors='k')
    plt.contour(X, Y, Z2, 15, colors='k')
    plt.xticks(()), plt.yticks(())  # remove x, y ticks
    plt.savefig(path)

class SpatialTransformation(nn.Module):
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        super(SpatialTransformation, self).__init__()

    def meshgrid(self, height, width):
        x_t = torch.matmul(torch.ones([height, 1]), torch.transpose(torch.unsqueeze(torch.linspace(0.0, width -1.0, width), 1), 1, 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height - 1.0, height), 1), torch.ones([1, width]))

        x_t = x_t.expand([height, width])
        y_t = y_t.expand([height, width])
        if self.use_gpu==True:
            x_t = x_t.cuda()
            y_t = y_t.cuda()

        # x与y相互
        x = x_t.cpu().numpy()
        y = y_t.cpu().numpy()

        return x_t, y_t

    def repeat(self, x, n_repeats):
        rep = torch.transpose(torch.unsqueeze(torch.ones(n_repeats), 1), 1, 0)
        rep = rep.long()
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        if self.use_gpu:
            x = x.cuda()
        return torch.squeeze(torch.reshape(x, (-1, 1)))


    def interpolate(self, im, x, y):

        im = F.pad(im, (0,0,1,1,1,1,0,0))

        batch_size, height, width, channels = im.shape

        batch_size, out_height, out_width = x.shape

        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        x = x + 1
        y = y + 1

        max_x = width - 1
        max_y = height - 1

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)

        dim2 = width
        dim1 = width*height
        base = self.repeat(torch.arange(0, batch_size)*dim1, out_height*out_width)

        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = torch.reshape(im, [-1, channels])
        im_flat = im_flat.float()
        dim, _ = idx_a.transpose(1,0).shape
        Ia = torch.gather(im_flat, 0, idx_a.transpose(1,0).expand(dim, channels))
        Ib = torch.gather(im_flat, 0, idx_b.transpose(1,0).expand(dim, channels))
        Ic = torch.gather(im_flat, 0, idx_c.transpose(1,0).expand(dim, channels))
        Id = torch.gather(im_flat, 0, idx_d.transpose(1,0).expand(dim, channels))

        # and finally calculate interpolated values
        x1_f = x1.float()
        y1_f = y1.float()

        dx = x1_f - x
        dy = y1_f - y

        wa = (dx * dy).transpose(1,0)
        wb = (dx * (1-dy)).transpose(1,0)
        wc = ((1-dx) * dy).transpose(1,0)
        wd = ((1-dx) * (1-dy)).transpose(1,0)

        output = torch.sum(torch.squeeze(torch.stack([wa*Ia, wb*Ib, wc*Ic, wd*Id], dim=1)), 1)
        output = torch.reshape(output, [-1, out_height, out_width, channels])
        output = output.permute(0,3,1,2)
        return output

    def forward(self, moving_image, deformation_matrix):


        dx = deformation_matrix[:, :, :, 0]
        dy = deformation_matrix[:, :, :, 1]

        batch_size, height, width = dx.shape

        x_mesh, y_mesh = self.meshgrid(height, width) # 生成网格点的坐标矩阵

        moving_image = moving_image.expand([batch_size, height, width,1])
        x_mesh = x_mesh.expand([batch_size, height, width])
        y_mesh = y_mesh.expand([batch_size, height, width])

        x_new = dx + x_mesh  # 对网格点进行形变
        y_new = dy + y_mesh

        return self.interpolate(moving_image, x_new, y_new)  #

'''
    是否需要更新梯度
'''
def make_trainable(model, val):
    for p in model.parameters():
        p.requires_grad = val

if __name__ == '__main__':
    path = '/home/leiy/DLProjects/MICCAI/data/data_skin/raw_data'

    edge_label(path,'f5')





















