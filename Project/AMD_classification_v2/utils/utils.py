import shutil
import torch
import sys
import os
from config import config
import re

class AverageMeter(object): # 标尺类
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


def save_checkpoint(state, is_best, fold):  # state是一个记录训练结果的字典
    filename = config.weights + config.model_name + \
        os.sep + str(fold) + os.sep + "_checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best:
        message = config.best_models + config.model_name + \
            os.sep + str(fold) + os.sep + 'model_best.pth.tar'
        print("Get Better top1 : %s saving weights to %s" %
              (state["best_precision1"], message))
        with open("./logs/%s.txt" % config.model_name, "a") as f:
            print("Get Better top1 : %s saving weights to %s" %
                  (state["best_precision1"], message), file=f)
        shutil.copyfile(filename, message)


def accuracy(output, target,topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) 返回输入张量指定dim的k个最大值
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target)
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]

    # assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

