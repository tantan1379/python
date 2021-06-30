'''
@File    :   utils.py
@Time    :   2021/06/07 16:24:59
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.
'''

import torch
from torch.nn import functional as F
import numpy as np
import pandas as pd
import os.path as osp
import shutil


class AverageMeter(object):  # 标尺类
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, best_pred, epoch, is_best, checkpoint_path, filename='./checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print("Saving best model to model_{:03d}_{:.4f}.pth.tar\n".format((epoch + 1), best_pred))
        shutil.copyfile(filename, osp.join(
            checkpoint_path, 'model_{:03d}_{:.4f}.pth.tar'.format((epoch + 1), best_pred)))


# 调整学习率（包括step模式和poly模式，一般采用poly模式）
def adjust_learning_rate(args, optimizer, epoch):
    if args.lr_mode == 'step': # Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.num_epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def one_hot_it(label, label_info):
# return semantic_map -> [H, W, num_classes]
    semantic_map = []
    for info in label_info:
        color = label_info[info]
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def eval_single_seg(predict, target, foreground = 1):
    # print(predict.shape)
    # print(target.shape)
    pred_seg = torch.round(torch.sigmoid(predict)).int()
    pred_seg = pred_seg.data.cpu().numpy()
    label_seg = target.data.cpu().numpy().astype(dtype=np.int)
    assert(pred_seg.shape == label_seg.shape)

    Dice = []
    Precision = []
    Jaccard = []
    Sensitivity=[]
    Specificity=[]

    n = pred_seg.shape[0]
    
    for i in range(n):
        dice,precision,jaccard,sensitivity,specificity = compute_score_single(pred_seg[i],label_seg[i])
        Dice.append(dice)
        Precision.append(precision)
        Jaccard.append(jaccard)
        Sensitivity.append(sensitivity)
        Specificity.append(specificity)

    return Dice,Precision,Jaccard,Sensitivity,Specificity


def compute_score_single(predict, target, foreground = 1,smooth=1):
    target[target!=foreground]=0
    predict[predict!=foreground]=0
    assert(predict.shape == target.shape)
    overlap = ((predict == foreground)*(target == foreground)).sum() #TP
    union = (predict == foreground).sum() + (target == foreground).sum()-overlap #FP+FN+TP
    FP = (predict == foreground).sum()-overlap #FP
    FN = (target == foreground).sum()-overlap #FN
    TN = target.shape[0]*target.shape[1]*target.shape[2]-union #TN

    dice=(2*overlap +smooth)/ (union+overlap+smooth)
    
    precision=((predict == target).sum()+smooth) / (target.shape[0]*target.shape[1]*target.shape[2]+smooth)
    
    jaccard=(overlap+smooth) / (union+smooth)

    Sensitivity=(overlap+smooth) / ((target == foreground).sum()+smooth)

    Specificity=(TN+smooth) / (FP+TN+smooth)
    

    return dice,precision,jaccard,Sensitivity,Specificity


def eval_multi_seg(predict, target, num_classes):
    smooth = 0.1
    # 转为numpy形式
    pred_seg = predict.data.cpu().numpy() 
    label_seg = target.data.cpu().numpy().astype(dtype=np.int)
    # 确保预测图片和标签形状一致
    assert (pred_seg.shape == label_seg.shape) # 128*512*256

    Dice = []
    Acc = []
    Jaccard = []
    Sensitivity=[]
    Specificity=[]

    for classes in range(1, num_classes): # 对每一类分别进行指标计算（跳过背景）
        overlap = ((pred_seg == classes) * (label_seg == classes)).sum()
        union = (pred_seg == classes).sum() + (label_seg == classes).sum() # TP+FN+FP

        FP = (pred_seg == classes).sum() - overlap  # FP
        FN = (label_seg == classes).sum() - overlap  # FN
        TN = label_seg.shape[0] * label_seg.shape[1] * label_seg.shape[2] - union # TN

        dice = (2 * overlap + smooth) / (union + smooth)
        acc = ((pred_seg == label_seg).sum() + smooth) / (label_seg.shape[0] * label_seg.shape[1] * label_seg.shape[2] + smooth)
        jaccard = (overlap + smooth) / (union - overlap + smooth)
        sensitivity = (overlap + smooth) / ((label_seg == classes).sum() + smooth)
        specificity = (TN + smooth) / (FP + TN + smooth)

        Dice.append(dice)
        Acc.append(acc)
        Jaccard.append(jaccard)
        Sensitivity.append(sensitivity)
        Specificity.append(specificity)
    # print(len(Dice))
    # print(len(Acc))
    # print(len(Jaccard))
    # print(len(Sensitivity))
    # print(len(Specificity))
    return Dice, Acc, Jaccard, Sensitivity, Specificity


def batch_pix_accuracy(pred, label, nclass=1):
    if nclass == 1:
        pred = torch.round(torch.sigmoid(pred)).int()
        pred = pred.cpu().numpy()
    else:
        pred = torch.max(pred, dim=1)
        pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    pixel_labeled = np.sum(label >= 0)
    pixel_correct = np.sum(pred == label)

    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"

    return pixel_correct, pixel_labeled


def batch_intersection_union(predict, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int),note: not include background
    """
    if nclass == 1:
        pred = torch.round(torch.sigmoid(predict)).int()
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        area_inter = np.sum(pred*target)
        area_union = np.sum(pred)+np.sum(target)-area_inter

        return area_inter, area_union

    if nclass > 1:
        _, predict = torch.max(predict, 1)
        mini = 1
        maxi = nclass
        nbins = nclass
        predict = predict.cpu().numpy() + 1
        target = target.cpu().numpy() + 1
        # target = target + 1

        predict = predict * (target > 0).astype(predict.dtype)
        intersection = predict * (predict == target)
        # areas of intersection and union
        area_inter, _ = np.histogram(
            intersection, bins=nbins-1, range=(mini+1, maxi))
        area_pred, _ = np.histogram(
            predict, bins=nbins-1, range=(mini+1, maxi))
        area_lab, _ = np.histogram(target, bins=nbins-1, range=(mini+1, maxi))
        area_union = area_pred + area_lab - area_inter
        assert (area_inter <= area_union).all(), \
            "Intersection area should be smaller than Union area"
        return area_inter, area_union


def pixel_accuracy(im_pred, im_lab):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(im_lab > 0)
    pixel_correct = np.sum((im_pred == im_lab) * (im_lab > 0))
    #pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return pixel_correct, pixel_labeled


def reverse_one_hot(image):
	"""
	Transform a 2D array in one-hot format (depth is num_classes),
	to a 2D array with only 1 channel, where each pixel value is
	the classified class key.

	# Arguments
		image: The one-hot format image

	# Returns
		A 2D array with the same width and height as the input, but
		with a depth size of 1, where each pixel value is the classified
		class key.
	"""
	# w = image.shape[0]
	# h = image.shape[1]
	# x = np.zeros([w,h,1])

	# for i in range(0, w):
	#     for j in range(0, h):
	#         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
	#         x[i, j] = index
	image = image.permute(1, 2, 0)
	x = torch.argmax(image, dim=-1)
	return x