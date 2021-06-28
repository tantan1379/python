'''
@File    :   config.py
@Time    :   2021/06/07 16:25:14
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.
'''

class DefaultConfig(object):
    num_epochs = 60
    epoch_start_i = 0
    checkpoint_step = 4
    crop_height = 512
    crop_width = 256
    batch_size = 8                  # *
    # input_channel = 1               # 输入的图像通道 *
    # 优化器相关
    optimizer = "SGD"               # Adam/SGD
    lr = 0.01                       # 学习率 *
    lr_mode = 'poly'                # poly优化策略
    step = 30                       # step模式时的衰减周期
    momentum = 0.9                  # 优化器动量
    weight_decay = 1e-4             # L2正则化系数
    # 训练相关
    net_work = 'unet'          # 可选网络 * unet/cpfnet/resunetplusplus/resunet1
    mode = 'train_test'                   # 训练模式 * train/test/train_test
    num_workers = 0                 # dataloader设置
    num_classes = 1                 # 分割类别数 类别数+加背景 *
    cuda = '0'                      # GPU id选择 *
    use_gpu = True
    # 路径相关
    data = r'F:/Dataset/CNV_Seg'    # 数据存放的根目录 *
    dataset = r'png_split_tvt'      # 数据库名字(需修改成自己的数据名字) *
    log_dirs = './results/save/'          # 存放tensorboard log的文件夹() *
    save_model_path = './checkpoints/'+net_work  # 保存模型的文件夹
    result_path = './results'+'/img_seg/'
    # pretrained_model_path = r"F:\MyGit\Project\CNV_Segmentation_tvt\checkpoints"+r"\unet\model_022_0.8236.pth.tar"