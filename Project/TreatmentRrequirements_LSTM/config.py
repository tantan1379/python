class DefaultConfigs(object):
    # ----------------------------------------------------------------
    # 1.string parameters
    # ----------------------------------------------------------------
    data_path = "F:\\Dataset\\AMD\\AMD_TimeSeries_CL\\4_to_9_single\\"
    gpus = '0'
    model_name = 'ResNetLSTM'
    # ----------------------------------------------------------------
    # 2.numeric parameters
    # ----------------------------------------------------------------
    epochs = 100
    lr = 1e-3
    # weight_decay = 1e-3
    seed = 2021
    # num_pat = 104
    seq_len = 18
    # input_channel = 6
    batch_size = 2
    img_height = 224
    img_width = 224
    center_crop_height = 112
    center_crop_width = 112
    K = 3

config = DefaultConfigs()