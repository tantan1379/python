from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import config
from utils import *
import os
import numpy as np
from models.UNet import UNet3D
from collections import OrderedDict
from dataset.dataset_lits import Lits_DataSet


def test(model, test_loader, criterion, n_labels):
    model.eval()
    test_loss = metrics.LossAverage()
    test_dice = metrics.DiceAverage(n_labels)
    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss.update(loss.item(), data.size(0))
            test_dice.update(output, target)
            # plt.subplot(131)
            # plt.imshow(data[0,0,0])
            # plt.subplot(132)
            # plt.imshow(target[0,1,0])
            # plt.subplot(133)
            # plt.imshow(output[0,1,0])
            # plt.show()
    if n_labels == 2:
        return OrderedDict({'Test Loss': test_loss.avg, 'Test dice0': test_dice.avg[0],
                        'Test dice1': test_dice.avg[1]})
    else:
        return OrderedDict({'Test Loss': test_loss.avg, 'Test dice0': test_dice.avg[0],
                        'Test dice1': test_dice.avg[1],'Test dice2': test_dice.avg[2]})
    
def create_list(test_path):
    test_list = os.listdir(test_path)
    f = open("./test/test_name_list.txt", 'w')
    for i in range(len(test_list)):
        f.write(str(test_list[i]) + "\n")
    f.close()


if __name__ == "__main__":
    args = config.args
    device = torch.device('cpu' if args.cpu else 'cuda')
    # 设定相关路径
    test_path = "F:\\Dataset\\gliomas\\batch1\\data\\"
    result_save_path = './output/{}/result'.format(args.save)
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    # 创建测试文件列表
    if not os.path.exists("./test/test_name_list.txt"):
        create_list(test_path)
    # 加载数据集并打包
    test_set = Lits_DataSet(args.crop_size, args.test_resize_scale, args.test_dataset_path, mode='test')
    test_loader = DataLoader(dataset=test_set,batch_size=1,num_workers=args.n_threads, shuffle=False)
    # 加载模型、损失函数和训练的结果
    model = UNet3D(in_channels=1, filter_num_list=[16, 32, 48, 64, 96], class_num=args.n_labels).to(device)
    loss=loss.DiceLoss(weight=np.array([0.3,0.7]))
    ckpt = torch.load('./output/{}/best_model.pth'.format(args.save))
    model.load_state_dict(ckpt['net'])
    # 创建log对象用于记录测试结果
    log = logger.Test_Logger('./output/{}'.format(args.save),"test_log")
    test_log = test(model,test_loader,loss,args.n_labels)
    log.update(test_log)
    
