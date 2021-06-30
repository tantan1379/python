class DefaultConfigs(object):
    # 1.string parameters
    train_data = "F:\\Dataset\\AMD\\AMD_Origin_2d_v1_cropped_tv\\train\\"
    # test_data = "F:\\\Dataset\\AMD\\AMD_CL\\split\\test\\"
    val_data = "F:\\Dataset\\AMD\\AMD_Origin_2d_v1_cropped_tv\\val\\"
    model_name = "resnet18"
    weights = ".\\checkpoints\\"
    best_models = weights + "best_model\\"
    submit = ".\\submit\\"
    logs = ".\\logs\\"
    gpus = "0"
    resume = "restart" # restart, best, last
    augmen_level = None  # "light","hard","hard2"

    # 2.numeric parameters
    epochs = 50
    batch_size = 32
    img_height = 300
    img_weight = 300
    num_classes = 3
    seed = 888
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4


config = DefaultConfigs()
