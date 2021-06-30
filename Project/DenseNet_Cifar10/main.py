#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: main.py
# author: twh
# time: 2020/12/21 21:10
from densenet import DenseNet
import numpy as np
import torch
from torch import nn
from datetime import datetime
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torch.utils import data
import matplotlib.pyplot as plt
import argparse
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp


# 数据增强
def data_tf(image):
    image = image.resize((96, 96), 2)  # 将图片放大到 96x96
    image = np.array(image, dtype='float32') / 255
    image = (image - 0.5) / 0.5  # 标准化
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    return image


# 准确率评估
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


# 训练+测试
def train(model, _train_data, valid_data, epochs, optimizer, criterion):
    loss_list = []
    train_acc_list = []
    test_acc_list = []
    best_train_acc = 0
    best_valid_acc = 0
    best_train_epoch = 0
    best_test_epoch = 0
    if torch.cuda.is_available():
        model = model.cuda()
    prev_time = datetime.now()
    for epoch in range(epochs):
        # 训练
        train_loss = 0
        train_acc = 0
        model = model.train()
        for im, label in train_data:
            if torch.cuda.is_available():
                im = Variable(im.cuda())
                label = Variable(label.cuda())
            else:
                im = Variable(im)
                label = Variable(label)
            output = model(im)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += get_acc(output, label)
        # 记录最小损失和最优准确率
        avg_train_acc = train_acc / len(train_data)
        avg_train_loss = train_loss / len(train_data)
        if avg_train_acc > best_train_acc:
            best_train_acc = avg_train_acc
            best_train_epoch = epoch + 1
        loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        # 记录时间
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        # 测试
        valid_loss = 0
        valid_acc = 0
        model = model.eval()
        for im, label in valid_data:
            with torch.no_grad():
                im = Variable(im.cuda())
                label = Variable(label.cuda())
            print(label.shape)
            output = model(im)
            print(output.shape)
            loss = criterion(output, label)
            valid_loss += loss.item()
            valid_acc += get_acc(output, label)
        avg_valid_loss = valid_loss / len(valid_data)
        avg_valid_acc = valid_acc / len(valid_data)
        if avg_valid_acc > best_valid_acc:
            best_valid_acc = avg_valid_acc
            best_test_epoch = epoch + 1
            torch.save(model.state_dict(), r'densenet_cifar10.pth')
        test_acc_list.append(avg_valid_acc)
        epoch_str = (
            "Epoch {}: Train Loss: {:.6f}, Train Acc: {:.2f}%, Valid Loss: {:.6f}, Valid Acc: {:.2f}%, "
                .format(epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100))
        prev_time = cur_time
        print(epoch_str + time_str)

    # 显示准确率
    print("")
    print("Best train accuracy = {:.2f}% when in Epoch {}\nBest test accuracy = {:.2f}% when in Epoch {}\n"
          .format(best_train_acc * 100, best_train_epoch, best_valid_acc * 100, best_test_epoch))

    # 显示各分类的准确度
    class_correct = [0] * 10
    class_total = [0] * 10
    for data in test_data:
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        output = model(images)
        _, predicted = torch.max(output, 1)
        c = (predicted == labels).squeeze()
        for i in range(10):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    return loss_list, train_acc_list, test_acc_list


def draw_ROC(net, _data, _num_class):
    # 绘制 ROC曲线
    score_list = []  # 存储预测得分
    label_list = []  # 存储真实标签
    for i, (inputs, labels) in enumerate(_data):
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = net(inputs)
        # prob_tmp = torch.nn.Softmax(dim=1)(outputs) # (batchsize, nclass)
        score_tmp = outputs  # (batchsize, nclass)

        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())

    score_array = np.array(score_list)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)

    print("score_array:", score_array.shape)  # (batchsize, classnum)
    print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])

    # 调用sklearn库，计算每个类别对应的fpr和tpr
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(num_class):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(_num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(_num_class):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

    # 绘制所有类别平均的roc曲线
    plt.figure()
    lw = 2
    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(_num_class), colors):
        plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc_dict[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('../Result/class10_roc.jpg')


def draw_loss_accuracy(epochs, _loss, _train_acc, _test_acc):
    x = np.arange(1, epochs + 1).astype(dtype=np.str)
    # 损失函数图像
    plt.figure()
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, 3)
    plt.plot(x, _loss, label='Loss')
    plt.legend()
    plt.savefig('../Result/Loss vs. Epochs.jpg')
    # 训练集准确率图像
    plt.figure()
    plt.title('Train Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Train Accuracy')
    plt.ylim(0, 1)
    plt.plot(x, _train_acc, label='Train Accuracy')
    plt.legend()
    plt.savefig('../Result/Train Accuracy vs. Epochs.jpg')
    # 测试集准确率图像
    plt.figure()
    plt.title('Test Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1)
    plt.plot(x, _test_acc, label='Test Auccracy')
    plt.legend()
    plt.savefig('../Result/Test Accuracy vs. Epochs.jpg')
    plt.show()


if __name__ == "__main__":
    # 设置随机种子，保证可复现性
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    # 创建解析器对象
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50)
    parser.add_argument('--draw', default=1, choices=['0', '1'])
    parser.add_argument('--train_batch_size', default=64)
    parser.add_argument('--test_batch_size', default=128)
    parser.add_argument('--num_class', default=10)
    args = parser.parse_args()
    # 设置超参数和其它参数
    num_epochs = int(args.epochs)
    draw = int(args.draw)
    num_class = int(args.num_class)
    train_batch_size = int(args.train_batch_size)
    test_batch_size = int(args.test_batch_size)
    # 载入数据库和网络
    train_set = CIFAR10('./data', train=True, transform=data_tf, download=True)
    train_data = data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    test_set = CIFAR10('./data', train=False, transform=data_tf, download=True)
    test_data = data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    net = DenseNet(3, 10)  # 输入为rgb三通道图片，cifar10数据集任务为将数据分为10类
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # 选择优化器和损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # 选择随机梯度下降作为优化方法
    criterion = nn.CrossEntropyLoss()  # 选择交叉熵损失函数
    # 显示模型参数量
    print('parameters_number = {}'.format(sum(p.numel() for p in list(net.parameters()) if p.requires_grad)))
    # 开始训练
    print("Started training:")
    loss, train_acc, test_acc = train(net, train_data, test_data, num_epochs, optimizer, criterion)
    # 绘图
    if draw == 1:
        draw_ROC(net, test_data, num_class)
        draw_loss_accuracy(num_epochs, loss, train_acc, test_acc)
