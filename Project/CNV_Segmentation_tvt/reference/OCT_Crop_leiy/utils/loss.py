import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
import torchvision

from matplotlib import pyplot as plt
from torch.autograd import Variable




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











class Multi_DiceLoss(nn.Module):
    def __init__(self, class_num=3, smooth=0.001):
        super(Multi_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self, input, target):
        input = torch.exp(input)
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(0, self.class_num):
            input_i = input[:, i, :, :]
            target_i = (target == i).float()
            intersect = (input_i * target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += dice
        dice_loss = 1 - Dice / (self.class_num)
        return dice_loss

class EL_DiceLoss(nn.Module):
    def __init__(self, class_num=4, smooth=1, gamma=0.5):
        super(EL_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num
        self.gamma = gamma

    def forward(self, input, target):
        input = torch.exp(input)
        self.smooth = 0.
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(1, self.class_num):
            input_i = input[:, i, :, :]
            target_i = (target == i).float()
            intersect = (input_i * target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = Variable(torch.Tensor([1]).float()).cuda()
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
        tver = Variable(torch.Tensor([0]).float()).cuda()
        intersect = (input * target).sum()
        fp = (input * (1 - target)).sum()
        fn = ((1 - input) * target).sum()
        tver = intersect / (intersect + self.alpha * fp + self.beta * fn + self.smooth)

        return 1 - tver









if __name__ == '__main__':
    import torch
    np.random.seed(520)
    target = torch.from_numpy(np.ones((1,1,2,2)).astype(np.float32))
    predit = torch.from_numpy(np.random.rand(1,1,2,2))
    print(target)
    print(predit)

    loss = FocalLoss()
    a = loss(predit,target)
    print(a)










#------------------------------------------------   GHMC_Loss   ------------------------------------------------#
# 二次函数_0.52
# class GHMC_loss(nn.Module):
#     def __init__(self,
#                  bins = 10,
#                  momentum = 0.2,
#                  loss_weight = 1.0,
#                  coefficient = 0.2,
#                  density = [],
#                  weight = [],
#                  gradient_norm_contribution = [],
#                  debug_mode = 'tensorboard'):
#
#         super(GHMC_loss,self).__init__()
#         self.bins = bins
#         self.momentum = momentum
#         # self.coefficient = Parameter(torch.zeros(1))
#         self.debug_mode = debug_mode
#         self.gd_weight = weight
#
#         self.edges = torch.arange(bins + 1).float().cuda() / bins
#         self.edges[-1] += 1e-6
#         if momentum > 0:
#             self.acc_sum = torch.zeros(bins).cuda()
#         self.loss_weight = loss_weight
#         self.density = torch.zeros(density)
#         self.gradient_norm_contribution = torch.zeros(gradient_norm_contribution)
#         self.min_contribution_result_hard_ratio = 1
#         self.gamma = 1
#
#     def forward(self, pred,target,para = []):
#         # 产生整个梯度模长的分区区间
#         edges = self.edges
#         mmt = self.momentum
#         weights = torch.zeros_like(pred)
#         contribution_result = [0 for i in range(self.bins)]
#         weights_result = [0 for i in range(self.bins)]
#
#         # 计算梯度模长 gradient len3gth
#         g = torch.abs(pred.detach() - target).cuda()
#
#         # 计算整个样本的总数
#         tot = pred.numel()
#         n = 0
#         for i in range(self.bins):
#             inds = (g >= edges[i]) & (g < edges[i+1])
#             num_in_bin = inds.sum().item() # 求和每个区间的样本总数             -4 * ((((edges[i] + edges[i + 1]) / 2) - 0.5)) ** 2) + 1
#
#             if num_in_bin > 0:  # 求解梯度贡献
#                 if mmt > 0:
#                     self.acc_sum[i] = mmt * self.acc_sum[i] \
#                                       + (1 - mmt) * num_in_bin  # 使用滑动平均的方法来统计梯度密度,让loss更加稳定
#
#                     self.acc_sum[i] = torch.log10(self.acc_sum[i])  # 取对数缩小数据差异
#                     weights[inds] = (1 / self.acc_sum[i]) * ((-4 * ((((edges[i] + edges[i + 1]) / 2) - 0.52) ** 2)) + 1)
#                     contribution_result[i] = self.acc_sum[i] * ((edges[i] + edges[i + 1]) / 2)   # 获取当前梯度贡献
#                     weights_result[i] = (1 / self.acc_sum[i]) * ((-4 * ((((edges[i] + edges[i + 1]) / 2) - 0.5) ** 2)) + 1)
#                 else:
#                     num_in_bin = torch.log10(num_in_bin)
#                     weights[inds] = (1 / num_in_bin) * ((-4 * ((((edges[i] + edges[i + 1]) / 2) - 0.52) ** 2)) + 1)
#                     contribution_result[i] = num_in_bin * ((edges[i] + edges[i + 1]) / 2)        # 获取当前梯度贡献
#                     weights_result[i] = (1 / self.acc_sum[i]) * ((-4 * ((((edges[i] + edges[i + 1]) / 2) - 0.5) ** 2)) + 1)
#
#                 n += 1
#         # if n > 0:
#             # weights = weights / n
#             # weights = torch.log10(self.coefficient * weights + 1)
#             # print(self.coefficient)
#
#         # 取
#
#         # 计算focal loss的参数,计算各个分布
#         contribution_result_easy = np.sum(np.array(contribution_result[0:5]))
#         contribution_result_hard = np.sum(np.array(contribution_result[5:10]))
#         contribution_result_hard_ratio = contribution_result_hard / (contribution_result_easy + contribution_result_hard)
#         # para[6].info('contribution_result_easy = %s,contribution_result_hard = %s' % (np.sum(np.array(contribution_result[0:5])),contribution_result_hard))
#
#         para[2].add_scalar('Train/contribution_result_easy_{}'.format(para[3]), contribution_result_easy,para[1])
#         para[2].add_scalar('Train/contribution_result_hard_{}'.format(para[3]), contribution_result_hard,para[1])
#         para[2].add_scalar('Train/contribution_result_ratio_{}'.format(para[3]), contribution_result_hard / (contribution_result_easy + contribution_result_hard), para[1])
#
#         # if contribution_result_hard_ratio < self.min_contribution_result_hard_ratio:
#         #     self.min_contribution_result_hard_ratio = contribution_result_hard_ratio
#         #
#         #     self.gamma = self.gamma + para[5].gamma_step
#         #     para[6].info('self.gamma = %s' % self.gamma)
#
#
#
#         #
#         # if self.gamma > 10:
#         #     self.gamma = 10
#         #
#         #
#         # pred = pred.view(-1,1)
#         # target = target.view(-1,1)
#         #
#         # pred_numpy = pred.data.cpu().numpy()
#         # target_numpy = target.data.cpu().numpy()
#
#         # pred = pred.clamp(min = 0.0001, max = 1.0)
#
#         # # loss_1
#         # loss_1 = -target * (torch.pow((1 - pred), self.gamma)) * torch.log(pred)
#         # # loss_0
#         # loss_0 = -(1 - target) * (torch.pow(pred, self.gamma)) * torch.log(1 - pred)
#         # # focal loss函数
#         # loss = loss_0 + loss_1
#         #
#         # loss_1_numpy = loss_1.data.cpu().numpy()
#         # loss_0_numpy = loss_0.data.cpu().numpy()
#
#         # loss = loss.sum() / tot
#
#         # 统计整个数据集上g的分布
#         if para[0] == 0:
#             self.density = g
#             self.gd_weight = weights_result
#             self.gradient_norm_contribution = contribution_result
#         else:
#             self.density = torch.cat([self.density,g],0)
#             # self.gradient_norm_regularization = torch.cat([self.gradient_norm_regularization,weights_result],0)
#             self.gd_weight = np.array(np.array(self.gd_weight) + np.array(weights_result))
#             self.gradient_norm_contribution = np.array(np.array(self.gradient_norm_contribution) + np.array(contribution_result))
#
#
#         if self.debug_mode == 'tensorboard':
#             if (para[0] + 1) == para[4] :
#                 self.density = self.density.data.cpu().numpy()
#                 para[2].add_histogram('Train/gradient_norm_distribution_{}'.format(para[3]),self.density,para[1])
#                 para[2].add_histogram('Train/gradient_norm_contribution_{}'.format(para[3]),self.gradient_norm_contribution,para[1])
#                 para[2].add_histogram('Train/weights_{}'.format(para[3]),weights, para[1])
#                 # para[2].add_scalar('Train/gradient_norm_0.5_sum_{}'.format(para[3]), np.sum((self.density < 0.5)),para[1])
#
#         elif self.debug_mode == 'matplotlib':
#             if (para[0] + 1) == para[4]:
#                 self.density   = self.density.data.cpu().numpy().flatten()
#                 # self.gd_weight = self.gd_weight.data.cpu().numpy().flatten()
#                 self.gradient_norm_contribution = np.array(self.gradient_norm_contribution)
#
#                 # 绘制density图像
#                 plt.figure()
#                 plt.hist(self.density,100,color = 'red',log = True,width=0.01)
#                 plt.grid(True)
#                 plt.savefig(para[5].image_save_path + str(para[3]) + '/' + para[5].net_work + '_gradient_norm_distribution_' + str(para[3]) + '_' + str(para[1]) + '.png')
#
#                 # 绘制权重图像
#                 plt.figure()
#                 plt.bar(np.arange(0,1,0.1),self.gd_weight,width = 0.02,color = 'red',align = 'center',log = True)
#                 plt.grid(True)
#                 plt.savefig(para[5].image_save_path + str(para[3]) + '/' + para[5].net_work + '_gradient_norm_weights_' + str(para[3]) + '_' + str(para[1])+  '.png')
#
#                 # 绘制梯度贡献图像
#                 plt.figure()
#                 plt.bar(np.arange(0,1,0.1),self.gradient_norm_contribution,width = 0.02,color = 'red',align = 'center',log = True)
#                 plt.grid(True)
#                 plt.savefig(para[5].image_save_path + str(para[3]) + '/' + para[5].net_work + '_gradient_norm_contribution_' + str(para[3]) + '_' + str(para[1])+ '.png')
#
#                 # plt.show()
#
#         # the output will be summed
#         loss = F.binary_cross_entropy(
#             pred,target,weights,reduction='sum') / tot
#
#
#
#         return loss * self.loss_weight








# #------------------------------------------------   GHMC_Loss   ------------------------------------------------#
# # 使用log函数缩小数据样本之间的差异
# class GHMC_loss(nn.Module):
#     def __init__(self,
#                  bins = 10,
#                  momentum = 0.2,
#                  loss_weight = 1.0,
#                  coefficient = 0.2,
#                  density = [],
#                  weight = [],
#                  gradient_norm_contribution = [],
#                  debug_mode = 'tensorboard'):
#
#         super(GHMC_loss,self).__init__()
#         self.bins = bins
#         self.momentum = momentum
#         # self.coefficient = Parameter(torch.zeros(1))
#         self.debug_mode = debug_mode
#         self.gd_weight = weight
#
#         self.edges = torch.arange(bins + 1).float().cuda() / bins
#         self.edges[-1] += 1e-6
#         if momentum > 0:
#             self.acc_sum = torch.zeros(bins).cuda()
#         self.loss_weight = loss_weight
#         self.density = torch.zeros(density)
#         self.gradient_norm_contribution = torch.zeros(gradient_norm_contribution)
#         self.min_contribution_result_hard_ratio = 1
#         self.gamma = 1
#
#     def forward(self, pred,target,para = []):
#         # 产生整个梯度模长的分区区间
#         edges = self.edges
#         mmt = self.momentum
#         weights = torch.zeros_like(pred)
#         contribution_result = [0 for i in range(self.bins)]
#         weights_result = [0 for i in range(self.bins)]
#
#         # 计算梯度模长 gradient len3gth
#         g = torch.abs(pred.detach() - target).cuda()
#
#         # 计算整个样本的总数
#         tot = pred.numel()
#         n = 0
#         for i in range(self.bins):
#             inds = (g >= edges[i]) & (g < edges[i+1])
#             num_in_bin = inds.sum().item() # 求和每个区间的样本总数
#
#             if num_in_bin > 0:  # 求解梯度贡献
#                 if mmt > 0:
#                     self.acc_sum[i] = mmt * self.acc_sum[i] \
#                                       + (1 - mmt) * num_in_bin  # 使用滑动平均的方法来统计梯度密度,让loss更加稳定
#
#                     self.acc_sum[i] = torch.log10(self.acc_sum[i])  # 取对数缩小数据差异
#                     weights[inds] = tot / self.acc_sum[i]
#                     contribution_result[i] = self.acc_sum[i] * ((edges[i] + edges[i + 1]) / 2)   # 获取当前梯度贡献
#                 else:
#                     num_in_bin = torch.log10(num_in_bin)
#                     weights[inds] = tot / num_in_bin
#                     contribution_result[i] = num_in_bin * ((edges[i] + edges[i + 1]) / 2)        # 获取当前梯度贡献
#                 n += 1
#         if n > 0:
#             weights = weights / n
#             # weights = torch.log10(self.coefficient * weights + 1)
#             # print(self.coefficient)
#
#         # 取
#
#         # 计算focal loss的参数,计算各个分布
#         contribution_result_easy = np.sum(np.array(contribution_result[0:5]))
#         contribution_result_hard = np.sum(np.array(contribution_result[5:10]))
#         contribution_result_hard_ratio = contribution_result_hard / (contribution_result_easy + contribution_result_hard)
#         # para[6].info('contribution_result_easy = %s,contribution_result_hard = %s' % (np.sum(np.array(contribution_result[0:5])),contribution_result_hard))
#
#         para[2].add_scalar('Train/contribution_result_easy_{}'.format(para[3]), contribution_result_easy,para[1])
#         para[2].add_scalar('Train/contribution_result_hard_{}'.format(para[3]), contribution_result_hard,para[1])
#         para[2].add_scalar('Train/contribution_result_ratio_{}'.format(para[3]), contribution_result_hard / (contribution_result_easy + contribution_result_hard), para[1])
#
#         # if contribution_result_hard_ratio < self.min_contribution_result_hard_ratio:
#         #     self.min_contribution_result_hard_ratio = contribution_result_hard_ratio
#         #
#         #     self.gamma = self.gamma + para[5].gamma_step
#         #     para[6].info('self.gamma = %s' % self.gamma)
#
#
#
#         #
#         # if self.gamma > 10:
#         #     self.gamma = 10
#         #
#         #
#         # pred = pred.view(-1,1)
#         # target = target.view(-1,1)
#         #
#         # pred_numpy = pred.data.cpu().numpy()
#         # target_numpy = target.data.cpu().numpy()
#
#         # pred = pred.clamp(min = 0.0001, max = 1.0)
#
#         # # loss_1
#         # loss_1 = -target * (torch.pow((1 - pred), self.gamma)) * torch.log(pred)
#         # # loss_0
#         # loss_0 = -(1 - target) * (torch.pow(pred, self.gamma)) * torch.log(1 - pred)
#         # # focal loss函数
#         # loss = loss_0 + loss_1
#         #
#         # loss_1_numpy = loss_1.data.cpu().numpy()
#         # loss_0_numpy = loss_0.data.cpu().numpy()
#
#         # loss = loss.sum() / tot
#
#         # 统计整个数据集上g的分布
#         if para[0] == 0:
#             self.density = g
#             self.gd_weight = weights_result
#             self.gradient_norm_contribution = contribution_result
#         else:
#             self.density = torch.cat([self.density,g],0)
#             # self.gradient_norm_regularization = torch.cat([self.gradient_norm_regularization,weights_result],0)
#             self.gd_weight = np.array(np.array(self.gd_weight) + np.array(weights_result))
#             self.gradient_norm_contribution = np.array(np.array(self.gradient_norm_contribution) + np.array(contribution_result))
#
#
#         if self.debug_mode == 'tensorboard':
#             if (para[0] + 1) == para[4] :
#                 self.density = self.density.data.cpu().numpy()
#                 para[2].add_histogram('Train/gradient_norm_distribution_{}'.format(para[3]),self.density,para[1])
#                 para[2].add_histogram('Train/gradient_norm_contribution_{}'.format(para[3]),self.gradient_norm_contribution,para[1])
#                 para[2].add_histogram('Train/weights_{}'.format(para[3]),weights, para[1])
#                 # para[2].add_scalar('Train/gradient_norm_0.5_sum_{}'.format(para[3]), np.sum((self.density < 0.5)),para[1])
#
#         elif self.debug_mode == 'matplotlib':
#             if (para[0] + 1) == para[4]:
#                 self.density   = self.density.data.cpu().numpy().flatten()
#                 # self.gd_weight = self.gd_weight.data.cpu().numpy().flatten()
#                 self.gradient_norm_contribution = np.array(self.gradient_norm_contribution)
#
#                 # 绘制density图像
#                 plt.figure()
#                 plt.hist(self.density,100,color = 'red',log = True,width=0.01)
#                 plt.grid(True)
#                 plt.savefig(para[5].image_save_path + str(para[3]) + '/' + para[5].net_work + '_gradient_norm_distribution_' + str(para[3]) + '_' + str(para[1]) + '.png')
#
#                 # 绘制权重图像
#                 plt.figure()
#                 plt.bar(np.arange(0,1,0.1),self.gd_weight,width = 0.02,color = 'red',align = 'center',log = True)
#                 plt.grid(True)
#                 plt.savefig(para[5].image_save_path + str(para[3]) + '/' + para[5].net_work + '_gradient_norm_weights_' + str(para[3]) + '_' + str(para[1])+  '.png')
#
#                 # 绘制梯度贡献图像
#                 plt.figure()
#                 plt.bar(np.arange(0,1,0.1),self.gradient_norm_contribution,width = 0.02,color = 'red',align = 'center',log = True)
#                 plt.grid(True)
#                 plt.savefig(para[5].image_save_path + str(para[3]) + '/' + para[5].net_work + '_gradient_norm_contribution_' + str(para[3]) + '_' + str(para[1])+ '.png')
#
#                 # plt.show()
#
#         # the output will be summed
#         loss = F.binary_cross_entropy(
#             pred,target,weights,reduction='sum') / tot
#
#
#
#         return loss * self.loss_weight






#------------------------------------------------   GHMC_Loss   ------------------------------------------------#
# 按照梯度贡献来分配权重
# class GHMC_loss(nn.Module):
#     def __init__(self,
#                  bins = 10,
#                  momentum = 0.2,
#                  loss_weight = 1.0,
#                  coefficient = 0.2,
#                  density = [],
#                  weight = [],
#                  gradient_norm_contribution = [],
#                  debug_mode = 'matplotlib'):
#
#         super(GHMC_loss,self).__init__()
#         self.bins = bins
#         self.momentum = momentum
#         # self.coefficient = Parameter(torch.zeros(1))
#         self.debug_mode = debug_mode
#         self.gd_weight = weight
#
#         self.edges = torch.arange(bins + 1).float().cuda() / bins
#         self.edges[-1] += 1e-6
#         if momentum > 0:
#             self.acc_sum = torch.zeros(bins).cuda()
#         self.loss_weight = loss_weight
#         self.density = torch.zeros(density)
#         self.gradient_norm_contribution = torch.zeros(gradient_norm_contribution)
#
#     def forward(self, pred,target,para = []):
#         # 产生整个梯度模长的分区区间
#         edges = self.edges
#         mmt = self.momentum
#         weights = torch.zeros_like(pred)
#         contribution_result = [0 for i in range(self.bins)]
#         weights_result = [0 for i in range(self.bins)]
#
#         # 计算梯度模长 gradient len3gth
#         g = torch.abs(pred.detach() - target).cuda()
#
#         # 计算整个样本的总数
#         tot = pred.numel()
#         n = 0
#         for i in range(self.bins):
#             inds = (g >= edges[i]) & (g < edges[i+1])
#             num_in_bin = inds.sum().item() # 求和每个区间的样本总数
#
#             if num_in_bin > 0:  # 求解梯度贡献
#                 if mmt > 0:
#                     self.acc_sum[i] = mmt * self.acc_sum[i] \
#                                       + (1 - mmt) * num_in_bin  # 使用滑动平均的方法来统计梯度密度,让loss更加稳定
#                     weights[inds] = (tot / (self.acc_sum[i] * ((edges[i] + edges[i + 1]) / 2)))
#                     contribution_result[i] = self.acc_sum[i] * ((edges[i] + edges[i + 1]) / 2)
#                     weights_result[i] = (tot / (self.acc_sum[i] * ((edges[i] + edges[i + 1]) / 2)))
#
#                 else:
#                     weights[inds] = (tot / (num_in_bin * ((edges[i] + edges[i + 1]) / 2)))
#                     contribution_result[i] = num_in_bin * ((edges[i] + edges[i + 1]) / 2)
#                     weights_result[i] = (tot / (self.acc_sum[i] * ((edges[i] + edges[i + 1]) / 2)))
#                 n += 1
#
#
#         if n > 0:
#             weights = weights / n
#             # weights = torch.log10(self.coefficient * weights + 1)
#             # print(self.coefficient)
#
#
#         # 统计整个数据集上g的分布
#         if para[0] == 0:
#             self.density = g
#             self.gd_weight = weights_result
#             self.gradient_norm_contribution = contribution_result
#         else:
#             self.density = torch.cat([self.density,g],0)
#             # self.gradient_norm_regularization = torch.cat([self.gradient_norm_regularization,weights_result],0)
#             self.gd_weight = np.array(np.array(self.gd_weight) + np.array(weights_result))
#             self.gradient_norm_contribution = np.array(np.array(self.gradient_norm_contribution) + np.array(contribution_result))
#
#
#         if self.debug_mode == 'tensorboard':
#             if (para[0] + 1) == para[4] :
#                 self.density = self.density.data.cpu().numpy()
#                 para[2].add_histogram('Train/gradient_norm_distribution_{}'.format(para[3]),self.density,para[1])
#                 para[2].add_histogram('Train/gradient_norm_contribution_{}'.format(para[3]),self.gradient_norm_contribution,para[1])
#                 para[2].add_histogram('Train/weights_{}'.format(para[3]),weights, para[1])
#                 para[2].add_scalar('Train/gradient_norm_0.5_sum_{}'.format(para[3]), np.sum((self.density < 0.5)),para[1])
#
#         elif self.debug_mode == 'matplotlib':
#             if (para[0] + 1) == para[4]:
#                 self.density   = self.density.data.cpu().numpy().flatten()
#                 # self.gd_weight = self.gd_weight.data.cpu().numpy().flatten()
#                 self.gradient_norm_contribution = np.array(self.gradient_norm_contribution)
#
#                 # 绘制density图像
#                 plt.figure()
#                 plt.hist(self.density,100,color = 'red',log = True,width=0.01)
#                 plt.grid(True)
#                 plt.savefig(para[5].image_save_path + str(para[3]) + '/' + para[5].net_work + '_gradient_norm_distribution_' + str(para[3]) + '_' + str(para[1]) + '.png')
#
#                 # 绘制权重图像
#                 plt.figure()
#                 plt.bar(np.arange(0,1,0.1),self.gd_weight,width = 0.02,color = 'red',align = 'center',log = True)
#                 plt.grid(True)
#                 plt.savefig(para[5].image_save_path + str(para[3]) + '/' + para[5].net_work + '_gradient_norm_weights_' + str(para[3]) + '_' + str(para[1])+  '.png')
#
#                 # 绘制梯度贡献图像
#                 plt.figure()
#                 plt.bar(np.arange(0,1,0.1),self.gradient_norm_contribution,width = 0.02,color = 'red',align = 'center',log = True)
#                 plt.grid(True)
#                 plt.savefig(para[5].image_save_path + str(para[3]) + '/' + para[5].net_work + '_gradient_norm_contribution_' + str(para[3]) + '_' + str(para[1])+ '.png')
#
#                 # plt.show()
#
#         # the output will be summed
#         loss = F.binary_cross_entropy(
#             pred,target,weights,reduction='sum') / tot
#
#         return loss * self.loss_weight


# #------------------------------------------------   GHMC_Loss   ------------------------------------------------#
#
# class GHMC_loss(nn.Module):
#     def __init__(self,
#                  bins = 10,
#                  momentum = 0.99,
#                  loss_weight = 1.0,
#                  coefficient = 0.2,
#                  density = [],
#                  gradient_norm_regularization = [],
#                  debug_mode = 'tensorboard'):
#
#         super(GHMC_loss,self).__init__()
#         self.bins = bins
#         self.momentum = momentum
#         self.coefficient = Parameter(torch.zeros(1))
#         self.debug_mode = debug_mode
#
#         self.edges = torch.arange(bins + 1).float().cuda() / bins
#         self.edges[-1] += 1e-6
#         if momentum > 0:
#             self.acc_sum = torch.zeros(bins).cuda()
#         self.loss_weight = loss_weight
#         self.density = torch.zeros(density)
#         self.gradient_norm_regularization = torch.zeros(gradient_norm_regularization)
#
#     def forward(self, pred,target,para = []):
#         # 产生整个梯度模长的分区区间
#         edges = self.edges
#         mmt = self.momentum
#         weights = torch.zeros_like(pred)
#         weights_result = torch.zeros_like(pred)    # 绘制权重图
#
#         # 计算梯度模长 gradient len3gth
#         g = torch.abs(pred.detach() - target).cuda()
#
#         # 计算整个样本的总数
#         tot = pred.numel()
#         n = 0
#         for i in range(self.bins):
#             inds = (g >= edges[i]) & (g < edges[i+1])
#             num_in_bin = inds.sum().item() # 求和每个区间的样本总数
#
#             if num_in_bin > 0:  # 求解梯度贡献
#                 if mmt > 0:
#                     self.acc_sum[i] = mmt * self.acc_sum[i] \
#                                       + (1 - mmt) * num_in_bin  # 使用滑动平均的方法来统计梯度密度,让loss更加稳定
#                     weights[inds] = (tot / self.acc_sum[i]) * torch.log10((1e-5)* self.acc_sum[i] + 1)  # 计算整个样本的协调系数
#                     # weights[inds] = (tot / self.acc_sum[i])
#                 else:
#                     weights[inds] = (tot / num_in_bin) * torch.log10((1e-5) * num_in_bin + 1)  # 不使用滑动平均
#                     # weights[inds] = (tot / num_in_bin)
#                 n += 1
#                 weights_result[inds] = edges[i]
#
#         if n > 0:
#             weights = weights / n
#             # weights = torch.log10(self.coefficient * weights + 1)
#             # print(self.coefficient)
#
#
#
#
#         # 统计整个数据集上g的分布
#         if para[0] == 0:
#             self.density = g
#             self.gradient_norm_regularization = weights_result
#         else:
#             self.density = torch.cat([self.density,g],0)
#             self.gradient_norm_regularization = torch.cat([self.gradient_norm_regularization,weights_result],0)
#
#
#         if self.debug_mode == 'tensorboard':
#             if (para[0] + 1) == para[4] :
#                 self.density = self.density.data.cpu().numpy()
#                 para[2].add_histogram('Train/gradient_norm_distribution_{}'.format(para[3]),self.density,para[1])
#                 para[2].add_histogram('Train/gradient_norm_regularization_{}'.format(para[3]),self.gradient_norm_regularization,para[1])
#                 para[2].add_histogram('Train/weights_{}'.format(para[3]),weights, para[1])
#                 para[2].add_scalar('Train/gradient_norm_0.5_sum_{}'.format(para[3]), np.sum((self.density < 0.5)),para[1])
#
#         elif self.debug_mode == 'matplotlib':
#             if (para[0] + 1) == para[4]:
#                 self.density = self.density.data.cpu().numpy()
#
#                 # 绘制density图像
#                 plt.figure()
#                 plt.bar(np.arange(0,1,float(1 / self.bins)),height = self.density,color = 'red',width = 0.007,log = True)
#                 plt.grid(True)
#
#         # the output will be summed
#         loss = F.binary_cross_entropy(
#             pred,target,weights,reduction='sum') / tot
#
#         return loss * self.loss_weight




# class GHMC_loss(nn.Module):
#     def __init__(self,
#                  bins = 200,
#                  momentum = 0.5,
#                  use_sigmoid = True,
#                  loss_weight = 1.0):
#         super(GHMC_loss,self).__init__()
#         self.bins = bins
#         self.momentum = momentum
#
#         self.edges = torch.arange(bins + 1).float().cuda() / bins
#         self.edges[-1] += 1e-6
#         if momentum > 0:
#             self.acc_sum = torch.zeros(bins).cuda()
#         self.use_sigmoid = use_sigmoid
#         if not self.use_sigmoid:
#             raise NotImplementedError
#         self.loss_weight = loss_weight
#
#     def forward(self, pred,target):
#         # 产生整个梯度模长的分区区间
#         edges = self.edges
#         mmt = self.momentum
#         weights = torch.zeros_like(pred)
#
#         # 计算梯度模长 gradient length
#         g = torch.abs(pred.detach() - target).cuda()
#
#         # 绘制图
#         density = []
#         weights_result = []
#
#         # 计算整个样本的总数
#         tot = pred.numel()
#         n = 0
#         for i in range(self.bins):
#             inds = (g >= edges[i]) & (g < edges[i+1])
#             num_in_bin = inds.sum().item()   # 统计梯度在某一个区间内的总数
#             # print('i = %s, num = %s'%(i,num_in_bin))
#             density.append(num_in_bin)
#
#             if num_in_bin > 0:
#                 if mmt > 0:
#                     self.acc_sum[i] = mmt * self.acc_sum[i] \
#                                       + (1 - mmt) * num_in_bin  # 使用滑动平均的方法来统计梯度密度,让loss更加稳定
#                     weights[inds] = tot / self.acc_sum[i]  # 计算整个样本的协调系数
#                 else:
#                     weights[inds] = tot / num_in_bin  # 不使用滑动平均
#                 n += 1
#                 weights_result.append(tot / num_in_bin)
#
#         if n > 0:
#             weights = weights / n
#             weights_result = np.array(weights_result) / n
#
#         # print('n = %s'%n)
#         # density = np.array(density)
#         #
#         # plt.figure()
#         # plt.bar(np.arange(0,1,0.01),height = density,color = 'red',width = 0.007,log = True)
#         # plt.grid(True)
#         #
#         #
#         # plt.figure()
#         # plt.bar(np.arange(0,0.84,0.01), height=weights_result, color='red',width = 0.007,log = True)
#         # plt.grid(True)
#         #
#         # plt.figure()
#         # plt.bar(np.arange(0,0.84,0.01), height = weights_result / tot, color = 'red',width = 0.007, log = True)
#         # plt.grid(True)
#         #
#         #
#         # plt.figure()
#         # plt.bar(np.arange(0,0.84,0.01),height =  np.arange(0,0.84,0.01) * weights_result, color = 'red',width = 0.007,log = True)
#         # plt.grid(True)
#         #
#         # plt.figure()
#         # plt.bar(np.arange(0,0.84,0.01),height = (np.arange(0,0.84,0.01) * weights_result) * density[0:84], color = 'red',width = 0.007,log = True)
#         # plt.grid(True)
#         # plt.show()
#
#
#         # weights_show = weights
#         # weights_show = weights_show.cpu().numpy()
#         #
#         # plt.figure()
#         # a = np.log10(weights_show[0, 0, :, :].flatten())
#         # b = np.log10(weights_show[1, 0, :, :].flatten())
#         # c = np.log10(weights_show[2, 0, :, :].flatten())
#         # d = np.log10(weights_show[3, 0, :, :].flatten())
#         #
#         # plt.subplot(221)
#         # plt.hist(a,50,color='red',log = True)
#         # plt.subplot(222)
#         # plt.hist(b,50,color='blue',log = True)
#         # plt.subplot(223)
#         # plt.hist(c,50,color='red',log = True)
#         # plt.subplot(224)
#         # plt.hist(d,50,color='blue',log = True)
#         # plt.show()
#
#
#         # print(np.max(weights_show[0, 0, :, :]))
#         # print(np.max(weights_show[1, 0, :, :]))
#         # print(np.max(weights_show[2, 0, :, :]))
#         # print(np.max(weights_show[3, 0, :, :]))
#
#
#         # cv2.imshow('image0', weights_show[0, 0, :, :])
#         # cv2.imshow('image1', weights_show[1, 0, :, :])
#         # cv2.imshow('image2', weights_show[2, 0, :, :])
#         # cv2.imshow('image3', weights_show[3, 0, :, :])
#         # cv2.waitKey()
#
#
#         # the output will be summed
#         loss = F.binary_cross_entropy(
#             pred,target,weights,reduction='sum') / tot
#
#         return loss * self.loss_weight
