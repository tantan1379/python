class DefaultConfigs(object):
    # 1.string parameters
    data_path = "F:\\Dataset\\AMD_TimeSeries_CL\\1_to_16_single\\"
    gpus = '0'
    model_name = 'ResNetLSTM'
    # 2.numeric parameters
    epochs = 30
    lr = 1e-3
    # weight_decay = 1e-3
    seed = 2021
    seq_len = 3
    batch_size = 1
    img_height = 224
    img_width = 224
    num_pat = 104
    K = 3

config = DefaultConfigs()