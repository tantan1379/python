class DefaultConfig(object):
    # 1.string parameters
    datapath = r"F:\Dataset\AMD\AMD_Origin_2d_v1_cropped_split"
    model_name = "resnet18"
    save_model_path = ".\\checkpoints\\"
    best_models = save_model_path + "best_model\\"
    submit = ".\\submit\\"
    log_dirs = ".\\logs\\"
    lr_mode = 'poly'
    gpus = "0"
    augmen_level = 'light'  # "light","hard","hard2"
    use_gpu = True

    # 2.numeric parameters
    num_epochs = 200
    batch_size = 8
    img_height = 300
    img_weight = 300
    num_classes = 3
    num_workers=0
    seed = 2021
    k_fold = 4
    start_fold = 1
    lr = 1e-3
    lr_decay = 1e-4
    weight_decay = 1e-4
