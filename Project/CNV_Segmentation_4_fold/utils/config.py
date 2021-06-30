"""
Created on Sun May 30 12:42 2021

@author:twh
"""

class DefaultConfig(object):
    num_epochs = 60
    epoch_start_i = 0
    checkpoint_step = 4
    crop_height = 512
    crop_width = 256
    batch_size = 2  # *
    input_channel = 1  # 输入的图像通道 *
    # 路径相关
    data = r'F:/Dataset/CNV_Seg'  # 数据存放的根目录 *
    dataset = r'png_split'  # 数据库名字(需修改成自己的数据名字) *
    log_dirs = './result'  # 存放tensorboard log的文件夹() *
    # 优化器相关
    lr = 0.01  # 学习率 *
    lr_mode = 'poly'  # poly优化策略
    net_work = 'ResUnetPlusPlus'  # 可选网络 *
    net_index = '0' # 尝试同一模型的次数
    momentum = 0.9  # 优化器动量
    weight_decay = 1e-4  # L2正则化系数
    # 训练相关
    mode = 'train'  # 训练模式 *
    # mode = 'test'  # 测试模式 *
    k_fold = 4   # 交叉验证折数 *
    test_fold = 4  # 测试时候需要选择的第几个文件夹
    start_fold = 1
    num_workers = 1
    num_classes = 1  # 分割类别数，二类分割设置为1，多类分割设置成 类别数+加背景 *
    cuda = '0'  # GPU id选择 *
    use_gpu = True
    pretrained_model_path = r'F:\MyGit\Project\CNV_Segmentation\checkpoints\4\model_054_0.9104.pth.tar'    # test的时候模型文件的选择（当mode='test'的时候用）
    save_model_path = './checkpoints'  # 保存模型的文件夹