class DefaultConfigs(object):
    # 1.string parameters
    train_data = "F:/Dataset/traffic-sign/train/"
    test_data = ""
    val_data = "F:/Dataset/traffic-sign/test/"
    model_name = "resnet18"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/"
    gpus = "0"
    augmen_level = "medium"  # "light","hard","hard2"

    # 2.numeric parameters
    epochs = 40
    batch_size = 50
    img_height = 300
    img_weight = 300
    num_classes = 62
    seed = 888
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4


config = DefaultConfigs()
