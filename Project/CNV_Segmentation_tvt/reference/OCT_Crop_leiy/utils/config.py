

def Model_choice(model = 'NestedUNet_V0'):

    if model == 'Test_V0':
        args = Test_V0_Config()
    if model == 'Mini_Unet_V0':
        args = Mini_Unet_V0_Config()
    if model == 'Mini_Unet_Dcm_V0':
        args = Mini_Unet_Dcm_V0_Config()
    if model == 'Mini_Unet_Dcm_V1':
        args = Mini_Unet_Dcm_V1_Config()
    if model == 'Mini_Unet_Dcm_V2':
        args = Mini_Unet_Dcm_V2_Config()
    if model == 'Mini_Unet_Dcm_V3':
        args = Mini_Unet_Dcm_V3_Config()

    if model == 'Mini_Unet_Dcm_For_Roi_V0':
        args = Mini_Unet_Dcm_For_Roi_V0_Config()

    if model == 'Baseline_V0':
        args = Baseline_V0_Config()

    if model == 'NestedUNet_V0':
        args = NestedUNet_V0_Config()

    if model == 'UNet_3D':
        args = UNet_3D_Config()

    return args

class Test_V0_Config(object):
    #---------------------------------------   网络参数   ---------------------------------------#
    host_name = 'leiy'
    net_work = 'Test_V0'
    loss = 'ALL_Loss'
    mode = 'train'  # 训练模式
    optimizer = 'sgd'
    num_epochs = 10000
    slice_num = 128
    lr = 2e-2
    lr_mode = 'poly' # poly优化策略
    k_fold = 3  # 交叉验证折数
    test_fold = 3  # 测试时候需要选择的第几个文件夹
    save_cp = True
    load_parameters = False
    batch_size = 1
    validation_step = 1    # 每训练几个epoch进行一次验证
    cuda = '0'  # GPU id选择        **
    use_gpu = True

    input_channel = 3
    num_classes = 3  # 分割类别数，二类分割设置为1，多类分割设置成 类别数+加背景
    momentum = 0.9   # 优化器动量选择
    weight_decay = 0  # L2正则化系数
    num_workers = 4 # 多线程读取数据

    #---------------------------------------   文件路径   ---------------------------------------#
    ## 数据加载路径
    data_load_path = '/home/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/data_crop/'

    ## tensorboard模块路径
    file_name = net_work

    tensorboard_path = '/home/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/result/' + file_name

    ## log模块文件路径
    logger_train_path = '/home/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/result/' + file_name
    logger_test_path  = '/home/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/result/' + file_name

    ## 模型加载/保存路径
    model_load_para_path = '/home/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/model_save/'+ file_name + '/'
    model_save_net_path =  '/home/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/model_save/'+ file_name + '/'

    ## predict图片保存路径
    image_save_path = '/home/' + host_name + '/DLProjects/Graduation_Design/OCT/Source/Source/logs/' + file_name + '/'

class Mini_Unet_V0_Config(object):
    #---------------------------------------   网络参数   ---------------------------------------#
    host_name = 'leiy'
    net_work = 'Mini_Unet_V0'
    loss = 'SD_Loss'
    mode = 'train'  # 训练模式
    optimizer = 'adam'
    num_epochs = 120
    slice_num = 128
    lr = 4e-4
    lr_mode = 'None' # poly优化策略
    k_fold = 3  # 交叉验证折数
    test_fold = 3  # 测试时候需要选择的第几个文件夹
    save_cp = True
    load_parameters = False
    batch_size = 16
    validation_step = 1    # 每训练几个epoch进行一次验证
    cuda = '0'  # GPU id选择        **
    use_gpu = True

    input_channel = 3
    num_classes = 3  # 分割类别数，二类分割设置为1，多类分割设置成 类别数+加背景
    momentum = 0.9   # 优化器动量选择
    weight_decay = 0  # L2正则化系数
    num_workers = 4 # 多线程读取数据

    #---------------------------------------   文件路径   ---------------------------------------#
    ## 数据加载路径
    data_load_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/data_crop/'

    ## tensorboard模块路径
    file_name = net_work

    tensorboard_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/result/' + file_name

    ## log模块文件路径
    logger_train_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/result/' + file_name
    logger_test_path  = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/result/' + file_name

    ## 模型加载/保存路径
    model_load_para_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/model_save/'+ file_name + '/'
    model_save_net_path =  '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/model_save/'+ file_name + '/'

    ## predict图片保存路径
    image_save_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/logs/' + file_name + '/'

class Mini_Unet_Dcm_V0_Config(object):
    #---------------------------------------   网络参数   ---------------------------------------#
    host_name = 'leiy'
    net_work = 'Mini_Unet_Dcm_V0'
    loss = 'SD_Loss'
    mode = 'test'  # 训练模式
    optimizer = 'adam'
    num_epochs = 120
    slice_num = 128
    lr = 4e-4
    lr_mode = 'None' # poly优化策略
    k_fold = 3  # 交叉验证折数
    test_fold = 1  # 测试时候需要选择的第几个文件夹
    save_cp = True
    load_parameters = False
    batch_size = 16
    validation_step = 1    # 每训练几个epoch进行一次验证
    cuda = '0'  # GPU id选择        **
    use_gpu = True

    input_channel = 3
    num_classes = 3  # 分割类别数，二类分割设置为1，多类分割设置成 类别数+加背景
    momentum = 0.9   # 优化器动量选择
    weight_decay = 0  # L2正则化系数
    num_workers = 4 # 多线程读取数据

    #---------------------------------------   文件路径   ---------------------------------------#
    ## 数据加载路径
    data_load_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/data_crop/'

    ## tensorboard模块路径
    file_name = net_work

    tensorboard_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/result/' + file_name

    ## log模块文件路径
    logger_train_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/result/' + file_name
    logger_test_path  = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/result/' + file_name

    ## 模型加载/保存路径
    model_load_para_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/model_save/'+ file_name + '/'
    model_save_net_path =  '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/model_save/'+ file_name + '/'

    ## predict图片保存路径
    image_save_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/logs/' + file_name + '/'

class Mini_Unet_Dcm_V1_Config(object):
    #---------------------------------------   网络参数   ---------------------------------------#
    host_name = 'leiy'
    net_work = 'Mini_Unet_Dcm_V1'
    loss = 'SD_Loss'
    mode = 'train'  # 训练模式
    optimizer = 'sgd'
    num_epochs = 120
    slice_num = 128
    lr = 4e-2
    lr_mode = 'poly' # poly优化策略
    k_fold = 3  # 交叉验证折数
    test_fold = 3  # 测试时候需要选择的第几个文件夹
    save_cp = True
    load_parameters = False
    batch_size = 16
    validation_step = 1    # 每训练几个epoch进行一次验证
    cuda = '0'  # GPU id选择        **
    use_gpu = True

    input_channel = 3
    num_classes = 3  # 分割类别数，二类分割设置为1，多类分割设置成 类别数+加背景
    momentum = 0.9   # 优化器动量选择
    weight_decay = 0  # L2正则化系数
    num_workers = 4 # 多线程读取数据

    #---------------------------------------   文件路径   ---------------------------------------#
    ## 数据加载路径
    data_load_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/data_crop/'

    ## tensorboard模块路径
    file_name = net_work

    tensorboard_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/result/' + file_name

    ## log模块文件路径
    logger_train_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/result/' + file_name
    logger_test_path  = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/result/' + file_name

    ## 模型加载/保存路径
    model_load_para_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/model_save/'+ file_name + '/'
    model_save_net_path =  '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/model_save/'+ file_name + '/'

    ## predict图片保存路径
    image_save_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/logs/' + file_name + '/'

class Mini_Unet_Dcm_V2_Config(object):
    #---------------------------------------   网络参数   ---------------------------------------#
    host_name = 'leiy'
    net_work = 'Mini_Unet_Dcm_V2'
    loss = 'SD_Loss'
    mode = 'train'  # 训练模式
    optimizer = 'adam'
    num_epochs = 120
    slice_num = 128
    lr = 4e-4
    lr_mode = 'poly' # poly优化策略
    k_fold = 3  # 交叉验证折数
    test_fold = 3  # 测试时候需要选择的第几个文件夹
    save_cp = True
    load_parameters = False
    batch_size = 16
    validation_step = 1    # 每训练几个epoch进行一次验证
    cuda = '0'  # GPU id选择        **
    use_gpu = True

    input_channel = 1
    num_classes = 3  # 分割类别数，二类分割设置为1，多类分割设置成 类别数+加背景
    momentum = 0.9   # 优化器动量选择
    weight_decay = 0  # L2正则化系数
    num_workers = 4 # 多线程读取数据

    #---------------------------------------   文件路径   ---------------------------------------#
    ## 数据加载路径
    data_load_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/data_crop/'

    ## tensorboard模块路径
    file_name = net_work

    tensorboard_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/result/' + file_name

    ## log模块文件路径
    logger_train_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/result/' + file_name
    logger_test_path  = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/result/' + file_name

    ## 模型加载/保存路径
    model_load_para_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/model_save/'+ file_name + '/'
    model_save_net_path =  '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/model_save/'+ file_name + '/'

    ## predict图片保存路径
    image_save_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/logs/' + file_name + '/'

class Mini_Unet_Dcm_V3_Config(object):
    #---------------------------------------   网络参数   ---------------------------------------#
    host_name = 'leiy'
    net_work = 'Mini_Unet_Dcm_V3'
    loss = 'SD_Loss'
    mode = 'train'  # 训练模式
    optimizer = 'sgd'
    num_epochs = 120
    slice_num = 128
    lr = 4e-2
    lr_mode = 'poly' # poly优化策略
    k_fold = 3  # 交叉验证折数
    test_fold = 3  # 测试时候需要选择的第几个文件夹
    save_cp = True
    load_parameters = False
    batch_size = 8
    validation_step = 1    # 每训练几个epoch进行一次验证
    cuda = '0'  # GPU id选择        **
    use_gpu = True

    input_channel = 3
    num_classes = 3  # 分割类别数，二类分割设置为1，多类分割设置成 类别数+加背景
    momentum = 0.9   # 优化器动量选择
    weight_decay = 0  # L2正则化系数
    num_workers = 4 # 多线程读取数据

    #---------------------------------------   文件路径   ---------------------------------------#
    ## 数据加载路径
    data_load_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/data_crop/'

    ## tensorboard模块路径
    file_name = net_work

    tensorboard_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/result/' + file_name

    ## log模块文件路径
    logger_train_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/result/' + file_name
    logger_test_path  = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/result/' + file_name

    ## 模型加载/保存路径
    model_load_para_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/model_save/'+ file_name + '/'
    model_save_net_path =  '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/model_save/'+ file_name + '/'

    ## predict图片保存路径
    image_save_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/logs/' + file_name + '/'

class Mini_Unet_Dcm_For_Roi_V0_Config(object):
    #---------------------------------------   网络参数   ---------------------------------------#
    host_name = 'leiy'
    net_work = 'Mini_Unet_Dcm_For_Roi_V0'
    loss = 'SD_Loss'
    mode = 'train'  # 训练模式
    optimizer = 'adam'
    num_epochs = 60
    slice_num = 128
    lr = 4e-4
    lr_mode = 'None' # poly优化策略
    k_fold = 3  # 交叉验证折数
    test_fold = 2  # 测试时候需要选择的第几个文件夹
    save_cp = True
    load_parameters = False
    batch_size = 16
    validation_step = 1    # 每训练几个epoch进行一次验证
    cuda = '0'  # GPU id选择        **
    use_gpu = True

    input_channel = 3
    num_classes = 3  # 分割类别数，二类分割设置为1，多类分割设置成 类别数+加背景
    momentum = 0.9   # 优化器动量选择
    weight_decay = 0  # L2正则化系数
    num_workers = 4 # 多线程读取数据

    #---------------------------------------   文件路径   ---------------------------------------#
    ## 数据加载路径
    data_load_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/data_dcm/'

    ## tensorboard模块路径
    file_name = net_work

    tensorboard_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/result/' + file_name

    ## log模块文件路径
    logger_train_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/result/' + file_name
    logger_test_path  = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/result/' + file_name

    ## 模型加载/保存路径
    model_load_para_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/model_save/'+ file_name + '/'
    model_save_net_path =  '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/model_save/'+ file_name + '/'

    ## predict图片保存路径
    image_save_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/logs/' + file_name + '/'

class Baseline_V0_Config(object):
    #---------------------------------------   网络参数   ---------------------------------------#
    host_name = 'leiy'
    net_work = 'Baseline_V0'
    loss = 'SD_Loss'
    mode = 'train'  # 训练模式
    optimizer = 'sgd'
    num_epochs = 60
    slice_num = 128
    lr = 2e-2
    lr_mode = 'poly' # poly优化策略
    k_fold = 3  # 交叉验证折数
    test_fold = 3  # 测试时候需要选择的第几个文件夹
    save_cp = True
    load_parameters = False
    batch_size = 16
    validation_step = 1    # 每训练几个epoch进行一次验证
    cuda = '0'  # GPU id选择        **
    use_gpu = True

    input_channel = 3
    num_classes = 3  # 分割类别数，二类分割设置为1，多类分割设置成 类别数+加背景
    momentum = 0.9   # 优化器动量选择
    weight_decay = 0  # L2正则化系数
    num_workers = 4 # 多线程读取数据

    #---------------------------------------   文件路径   ---------------------------------------#
    ## 数据加载路径
    data_load_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/data_crop/'

    ## tensorboard模块路径
    file_name = net_work

    tensorboard_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/result/' + file_name

    ## log模块文件路径
    logger_train_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/result/' + file_name
    logger_test_path  = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/result/' + file_name

    ## 模型加载/保存路径
    model_load_para_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/model_save/'+ file_name + '/'
    model_save_net_path =  '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/model_save/'+ file_name + '/'

    ## predict图片保存路径
    image_save_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/logs/' + file_name + '/'

class NestedUNet_V0_Config(object):
    #---------------------------------------   网络参数   ---------------------------------------#
    host_name = 'leiy'
    net_work = 'NestedUNet_V0'
    loss = 'SD_Loss'
    mode = 'train'  # 训练模式
    optimizer = 'adam'
    num_epochs = 120
    slice_num = 128
    lr = 4e-4
    lr_mode = 'None' # poly优化策略
    k_fold = 3  # 交叉验证折数
    test_fold = 3  # 测试时候需要选择的第几个文件夹
    save_cp = True
    load_parameters = False
    batch_size = 8
    validation_step = 1    # 每训练几个epoch进行一次验证
    cuda = '0'  # GPU id选择        **
    use_gpu = True

    input_channel = 3
    num_classes = 3  # 分割类别数，二类分割设置为1，多类分割设置成 类别数+加背景
    momentum = 0.9   # 优化器动量选择
    weight_decay = 0  # L2正则化系数
    num_workers = 4 # 多线程读取数据

    #---------------------------------------   文件路径   ---------------------------------------#
    ## 数据加载路径
    data_load_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/data_crop/'

    ## tensorboard模块路径
    file_name = net_work

    tensorboard_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/result/' + file_name

    ## log模块文件路径
    logger_train_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/result/' + file_name
    logger_test_path  = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/result/' + file_name

    ## 模型加载/保存路径
    model_load_para_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/model_save/'+ file_name + '/'
    model_save_net_path =  '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/model_save/'+ file_name + '/'

    ## predict图片保存路径
    image_save_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/logs/' + file_name + '/'

class UNet_3D_Config(object):
    #---------------------------------------   网络参数   ---------------------------------------#
    host_name = 'leiy'
    net_work = 'UNet_3D'
    loss = 'SD_3D_Loss'
    mode = 'train'  # 训练模式
    optimizer = 'sgd'
    crop_size = [32, 128, 128]
    scale = None
    num_epochs = 60
    slice_num = 128
    lr = 2e-2
    lr_mode = 'poly' # poly优化策略
    k_fold = 3  # 交叉验证折数
    test_fold = 3  # 测试时候需要选择的第几个文件夹
    save_cp = True
    load_parameters = False
    batch_size = 8
    validation_step = 1    # 每训练几个epoch进行一次验证
    cuda = '0'  # GPU id选择        **
    use_gpu = True

    input_channel = 1
    num_classes = 3  # 分割类别数，二类分割设置为1，多类分割设置成 类别数+加背景
    momentum = 0.9   # 优化器动量选择
    weight_decay = 0  # L2正则化系数
    num_workers = 4 # 多线程读取数据

    #---------------------------------------   文件路径   ---------------------------------------#
    ## 数据加载路径
    data_load_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/data_crop/'

    ## tensorboard模块路径
    file_name = net_work

    tensorboard_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/result/' + file_name

    ## log模块文件路径
    logger_train_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/result/' + file_name
    logger_test_path  = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/result/' + file_name

    ## 模型加载/保存路径
    model_load_para_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/model_save/'+ file_name + '/'
    model_save_net_path =  '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/model_save/'+ file_name + '/'

    ## predict图片保存路径
    image_save_path = '/data1/' + host_name + '/DLProjects/Graduation_Design/OCT_Crop_Doctor/Source/Source/logs/' + file_name + '/'


