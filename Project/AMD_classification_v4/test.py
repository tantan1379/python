import os
import random
import torch
import numpy as np
import warnings
from config import config
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.dataloader import *
from models.model import *
from utils import *


# set random.seed and cudnn performance
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')


def evaluate(model, loader):
    model.eval()
    top1 = AverageMeter()
    for _, (input, target) in enumerate(loader):
        input = input.cuda()
        target = torch.from_numpy(np.array(target)).long().cuda()
        output = model(input)
        precision, _ = accuracy(output, target, topk=(1, 2))
        top1.update(precision[0], input.size(0))
    return top1.avg.item()


def main():
    fold = 0
    # mkdirs
    if not os.path.exists(config.submit):
        os.mkdir(config.submit)
    if not os.path.exists(config.weights):
        os.mkdir(config.weights)
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    if not os.path.exists(config.weights + config.model_name + os.sep + str(fold) + os.sep):
        os.makedirs(config.weights + config.model_name +
                    os.sep + str(fold) + os.sep)
    if not os.path.exists(config.best_models + config.model_name + os.sep + str(fold) + os.sep):
        os.makedirs(config.best_models + config.model_name +
                    os.sep + str(fold) + os.sep)
    # get model and optimizer
    model = get_net().cuda()
    # model = torch.nn.DataParallel(model)
    # get files and split for K-fold dataset
    test_files = get_files(config.test_data)
    # load dataset
    test_dataloader = DataLoader(ChaojieDataset(
        test_files,'val'), batch_size=10, shuffle=False, pin_memory=False)
    # best_model = torch.load(
    #     "checkpoints/best_model/%s/0/model_best.pth.tar" % config.model_name)
    # model.load_state_dict(best_model["state_dict"])
    precision = evaluate(model, test_dataloader)
    # print(precision)
    labels = [1, 2, 3]
    confusion = ConfusionMatrix(num_classes=len(labels), labels=labels)
    model.eval()
    with torch.no_grad():
        for val_data in test_dataloader:
            val_images, val_labels = val_data
            # print(val_images)
            # print("real label",val_labels)
            # print(val_images.shape)
            outputs = model(val_images.cuda())
            # outputs = torch.softmax(outputs, dim=1)
            # print(outputs.shape)
            outputs = torch.argmax(outputs, dim=1)
            # print("prediction",outputs)
            confusion.update(outputs.to("cpu").numpy(),
                             val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()
    # draw confusion matrix


if __name__ == "__main__":
    main()
