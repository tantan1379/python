#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: main.py
# author: twh
# time: 2021/3/11 19:37
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import warnings
import numpy as np
import argparse
from tqdm import tqdm
import time
from dataset import *
from Resnet import *

warnings.filterwarnings("ignore")


def evaluate(md, loader):
    correct_num = 0
    TP, TN, FN, FP = 0, 0, 0, 0
    total_num = len(loader.dataset)
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        output = md(data)
        _, prediction = torch.max(output, 1)
        correct_num += torch.eq(prediction, target).cpu().sum()
        TP += ((target == 1) & (prediction == 1)).cpu().sum()
        TN += ((target == 0) & (prediction == 0)).cpu().sum()
        FN += ((target == 0) & (prediction == 1)).cpu().sum()
        FP += ((target == 1) & (prediction == 0)).cpu().sum()

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    acc1 = (TP + TN) / (TP + TN + FP + FN)
    acc2 = 100 * correct_num / total_num

    return acc1, acc2


def train(md, epochs_num, tl, lr):
    md.train()
    print("\nStart training:\n")
    bs_epoch, bs_acc = 0, 0
    losses = []
    ls_list = []
    ac_list = []
    for epoch in range(epochs_num):
        time.sleep(0.1)
        txt = 'Epoch {}'.format(epoch + 1)
        for img, label in tqdm(tl, desc=txt, ncols=100, mininterval=0.01):
            # img.size [b,3,224,224]  label.size [b]
            img, label = img.to(device), label.to(device)
            logits = md(img)
            loss = loss_fn(logits, label)
            losses.append(loss.item())
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = np.mean(losses)
        ls_list.append(loss)
        print('Average loss in epoch {} is {}'.format(epoch + 1, loss))
        losses = []
        val_acc,val_acc1 = evaluate(md, val_loader)
        print('The accuracy of validation set is {:.6f}%\n'.format(val_acc))
        print('The accuracy of validation set is {:.6f}%\n'.format(val_acc1))
        ac_list.append(val_acc)

        # Save
        if val_acc > bs_acc:
            bs_epoch = epoch
            bs_acc = val_acc
            torch.save(md.state_dict(), r'AMDCL_ckp.pth')

    return bs_acc, bs_epoch, ls_list, ac_list


if __name__ == '__main__':
    # Set seeds
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Create the parser object
    parser = argparse.ArgumentParser()
    # Add parser argument
    parser.add_argument('--device', default='GPU', choices=['GPU', 'CPU'])
    parser.add_argument('--epochs', default='30')
    parser.add_argument('--batchsize', default='32')
    parser.add_argument('--draw', default='1', choices=['0', '1'])
    args = parser.parse_args()

    # Load superparameter
    epochs = int(args.epochs)
    batch_size = int(args.batchsize)
    draw = int(args.draw)
    learning_rate = 1e-4
    # lr = float(args.lr)
    if args.device == 'GPU':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load dataset and model
    train_db = AMD_CL(r"F:\Lab\AMD_CL\preprocessed", 224, 'train', True)  # dataset(0%~60%) as train_set
    val_db = AMD_CL(r"F:\Lab/AMD_CL\preprocessed", 224, 'val', False)  # dataset(60%~80%) as validation_set
    test_db = AMD_CL(r"F:\Lab\AMD_CL\preprocessed", 224, 'test', False)  # dataset(80%~100%) as test_set
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=18)
    val_loader = DataLoader(val_db, batch_size=batch_size, shuffle=False, num_workers=18)
    test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=False, num_workers=18)
    print("Classes & Labels are as follows:")
    model = resnet50(3).to(device)

    # Count the number of parameters
    print('parameters_number = {}'.format(sum(p.numel() for p in list(model.parameters()) if p.requires_grad)))

    # Select loss function
    loss_fn = nn.CrossEntropyLoss().to(device)

    # Start training
    best_acc, best_epoch, loss_list, ac_list = train(model, epochs, train_loader, learning_rate)
    print("Training Result:")
    print('lr={} : best_acc: {:.6f}% best_epoch: {}'.format(learning_rate, best_acc, best_epoch + 1))

    # Start test
    print('Start Test:\n')
    model.load_state_dict(torch.load(r'AMDCL_ckp.pth'))
    model.eval()
    test_acc = evaluate(model, test_loader)
    print('The accuracy of test set is {:.6f}%'.format(test_acc))

    # Draw loss function curve
    if draw == 1:
        x = np.arange(1, epochs + 1).astype(dtype=np.str)
        plt.figure()
        plt.title('Acc vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Acc')
        plt.plot(x, ac_list, color='black', label='Accuracy_lr_1e-4')

        plt.figure()
        plt.title('Loss vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(x, loss_list, color='green', label='Loss_lr_1e-4')
        plt.legend()
        plt.show()
