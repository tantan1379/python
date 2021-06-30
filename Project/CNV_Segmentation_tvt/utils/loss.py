'''
@File    :   loss.py
@Time    :   2021/06/07 16:25:39
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.
'''

import torch
import torch.nn as nn

class SD_Loss(nn.Module):
    def __init__(self,weight = 0.5):
        super(SD_Loss, self).__init__()
        self.weight = weight
        self.ce_loss = nn.NLLLoss()
        self.dice_loss = Multi_DiceLoss()

    def forward(self, y_pred, y_true):
        ce_loss = self.ce_loss(y_pred,y_true)
        dice_loss = self.dice_loss(y_pred,y_true)
        return ce_loss + dice_loss

class DiceLoss(nn.Module):
    def __init__(self,smooth=0.001):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self,output,target):
        # output = torch.sigmoid(output)
        Dice = torch.Tensor([0]).float().cuda()
        intersect = (output*target).sum()
        union = torch.sum(output)+torch.sum(target)
        Dice = (2*intersect+self.smooth)/(union + self.smooth)
        dice_loss = 1-Dice
        return dice_loss
    

class Multi_DiceLoss(nn.Module):
    def __init__(self, class_num=3, smooth=0.001):
        super(Multi_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self, input, target):
        input = torch.exp(input)
        Dice = torch.Tensor([0]).float().cuda()
        for i in range(0, self.class_num):
            input_i = input[:, i, :, :]
            target_i = (target == i).float()
            intersect = (input_i * target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += dice
        dice_loss = 1 - Dice / (self.class_num)
        return dice_loss


class SD_3D_Loss(nn.Module):
    def __init__(self,weight = 0.5):
        super(SD_3D_Loss, self).__init__()
        self.weight = weight
        self.ce_loss = nn.NLLLoss()
        self.dice_loss = Multi_DiceLoss()

    def forward(self, y_pred, y_true):
        y_true =  y_true.squeeze()
        ce_loss = self.ce_loss(y_pred,y_true)
        dice_loss = self.dice_loss(y_pred,y_true)

        return ce_loss + dice_loss


class ALL_Loss(nn.Module):
    def __init__(self,weight = 0.5):
        super(ALL_Loss, self).__init__()
        self.weight = weight
        self.ce_loss = nn.NLLLoss()
        self.dice_loss = Multi_DiceLoss()
        self.entropy_loss = Entropy_Loss()

    def forward(self, y_pred, y_true):
        ce_loss = self.ce_loss(y_pred,y_true)
        dice_loss = self.dice_loss(y_pred,y_true)
        entropy_loss = self.entropy_loss(y_pred)
        return ce_loss + dice_loss + 0.75 * entropy_loss


class Entropy_Loss(nn.Module):
    def __init__(self, class_num=3, smooth=0.001):
        super(Entropy_Loss, self).__init__()
        self.smooth = smooth

    def forward(self, input):
        input = torch.exp(input)
        loss = -(input * torch.log2(input))
        loss = torch.mean(loss)
        return loss


class EL_DiceLoss(nn.Module):
    def __init__(self, class_num=4, smooth=1, gamma=0.5):
        super(EL_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num
        self.gamma = gamma

    def forward(self, input, target):
        input = torch.exp(input)
        self.smooth = 0.
        Dice = torch.Tensor([0]).float().cuda()
        for i in range(1, self.class_num):
            input_i = input[:, i, :, :]
            target_i = (target == i).float()
            intersect = (input_i * target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = torch.Tensor([1]).float().cuda()
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += (-torch.log(dice)) ** self.gamma
        dice_loss = Dice / (self.class_num - 1)
        return dice_loss

class LS_loss(nn.Module):
    def __init__(self):
        super(LS_loss, self).__init__()
    def forward(self, y_pred, image_input):

        # print('y_pred_max = %s, y_pred_min = %s'%(torch.max(y_pred),torch.min(y_pred)))
        # print('y_imag_max = %s, y_imag_min = %s'%(torch.max(image_input),torch.min(image_input)))

        # 计算前景均值和图像前景值
        y_avg_front = torch.mean(y_pred * image_input)
        a = (y_pred * image_input - y_avg_front)
        b = (y_pred * image_input - y_avg_front) ** 2
        y_front = torch.sum((y_pred * image_input - y_avg_front) ** 2)

        # 计算背景均值
        y_avg_back  = torch.mean((1 - y_pred) * image_input)
        y_back = torch.sum(((1 - y_pred) * image_input - y_avg_back)**2)

        loss = y_front + y_back
        return loss

class SL_loss(nn.Module):
    def __init__(self,weight = 0.0001):
        super(SL_loss, self).__init__()
        self.weight = torch.nn.Parameter(torch.cuda.FloatTensor(1), requires_grad=True)
        self.weight.data.fill_(weight)
        self.bce_loss = nn.BCELoss()
        self.ls_loss = LS_loss()

    def forward(self, y_pred, y_true,image_input):
        bce_loss = self.bce_loss(y_pred,y_true)
        ls_loss = self.ls_loss(y_pred,image_input)
        return bce_loss + self.weight * ls_loss


class Tversky_loss(nn.Module):
    def __init__(self,smooth = 0.01, alpha = 0.25, beta = 0.75):
        super(Tversky_loss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self,input,target):
        tver = torch.Tensor([0]).float().cuda()
        intersect = (input * target).sum()
        fp = (input * (1 - target)).sum()
        fn = ((1 - input) * target).sum()
        tver = intersect / (intersect + self.alpha * fp + self.beta * fn + self.smooth)

        return 1 - tver