import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self,smooth=0.01):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self,output,target):
        output = torch.sigmoid(output)
        Dice = torch.Tensor([0]).float().cuda()
        intersect = (output*target).sum()
        union = torch.sum(output)+torch.sum(target)
        Dice = (2*intersect+self.smooth)/(union + self.smooth)
        dice_loss = 1-Dice
        return dice_loss
    
class Multi_DiceLoss(nn.Module):
    def __init__(self,class_num=4,smooth=0.001):
        super(Multi_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self,output,target):
        Dice = torch.Tensor([0]).float().cuda()
        for i in range(0,self.class_num):
            output_i = output[:,i,:,:]
            target_i = (target==i).float()
            intersect = (output_i*target*i).sum()
            union = torch.sum(output_i)+torch.sum(target_i)
            dice = (2*intersect+self.smooth) / (union + self.smooth)
            Dice+=dice
        dice_loss = 1-Dice/(self.class_num)
        return dice_loss


class EL_DiceLoss(nn.Module):
    def __init__(self,class_num=4,smooth=1,gamma=0.5):
        super(EL_DiceLoss,self).__init__()
        self.smooth = smooth
        self.class_num = class_num
        self.gamma = gamma

    def forward(self,output,target):
        output = torch.exp(output)
        Dice = torch.Tensor([0]).float().cuda()
        for i in range(1,self.class_num):
            output_i = output[:,i,:,:]
            target_i = (target==i).float()
            intersect = (output_i*target_i).sum()
            union = torch.sum(output_i)+torch.sum(target_i)
            if target_i.sum()==0:
                dice = torch.Tensor([1]).float().cuda()
            else:
                dice = (2*intersect+self.smooth) / (union + self.smooth)
            Dice += (-torch.log(dice))**self.gamma
        dice_loss = Dice/(self.class_num-1)
        return dice_loss