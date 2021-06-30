#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: main.py
# author: twh
# time: 2020/11/17 20:58
from torch.utils.data import DataLoader
from torch import optim
from ResNet import *
from test import *
import matplotlib.pyplot as plt
import warnings
import numpy as np
import argparse
from tqdm import tqdm
from Pokemon import *
import time

warnings.filterwarnings("ignore")


def choose_lr(speed):
    # Select optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr[speed])
    return optimizer


def evaluate(md, loader):
    correct_num = 0
    total_num = len(loader.dataset)
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        output = md(data)
        _, prediction = torch.max(output, 1)
        print(prediction.shape)
        print(target.shape)
        correct_num += torch.eq(prediction, target).cpu().sum()
    return 100 * correct_num / total_num


def train(md, epochs_num, tl, spd):
    model.train()
    print("\nStart training:\n")
    bs_epoch, bs_acc = 0, 0
    losses = []
    ls_list = []
    ac_list = []
    for epoch in range(epochs_num):
        time.sleep(0.1)
        txt = 'lr={}  Epoch {}'.format(lr[spd], epoch + 1)
        for img, label in tqdm(tl, desc=txt, ncols=100, mininterval=0.01):
            # img.size [b,3,224,224]  label.size [b]
            img, label = img.to(device), label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            losses.append(loss.item())
            optimizer = choose_lr(spd)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = np.mean(losses)
        ls_list.append(loss)
        print('Average loss in epoch {} is {}'.format(epoch + 1, loss))
        losses = []
        val_acc = evaluate(md, val_loader)
        print('The accuracy of validation set is {:.6f}%\n'.format(val_acc))
        ac_list.append(val_acc)

        # Save
        if val_acc > bs_acc:
            bs_epoch = epoch
            bs_acc = val_acc
            torch.save(model.state_dict(), r'pokemon_ckp.pth')

    return bs_acc, bs_epoch, ls_list, ac_list


if __name__ == '__main__':
    # Set seeds
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    lr = [1e-5, 1e-4, 1e-3, 1e-2]

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
    # lr = float(args.lr)
    if args.device == 'GPU':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load dataset and model
    train_db = Pokemon(r'F:\\Dataset\\pokemon\\', 224, 'train')  # dataset(0%~70%) as train_set
    val_db = Pokemon(r'F:\\Dataset\\pokemon\\', 224, 'val')  # dataset(70%~85%) as validation_set
    test_db = Pokemon(r'F:\\Dataset\\pokemon\\', 224, 'test')  # dataset(85%~100%) as test_set
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_db, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=False, num_workers=0)
    print("Classes & Labels are as follows:")
    print(train_db.get_label())
    model = resnet50(4).to(device)

    # Count the number of parameters
    print('parameters_number = {}'.format(sum(p.numel() for p in list(model.parameters()) if p.requires_grad)))

    # Select loss function
    loss_fn = nn.CrossEntropyLoss().to(device)

    # Start training
    best_acc_0, best_epoch_0, loss_list_0, ac_list_0 = train(model, epochs, train_loader, 1)
    print("Training Result:")
    print('lr={} : best_acc: {:.6f}% best_epoch: {}'.format(lr[1], best_acc_0, best_epoch_0 + 1))
    # best_acc_1, best_epoch_1, loss_list_1 = train(model, epochs, train_loader, 1)
    # print("Training Result:")
    # print('lr={} : best_acc: {:.6f}% best_epoch: {}'.format(lr[1], best_acc_1, best_epoch_1 + 1))
    # best_acc_2, best_epoch_2, loss_list_2 = train(model, epochs, train_loader, 2)
    # print("Training Result:")
    # print('lr={} : best_acc: {:.6f}% best_epoch: {}'.format(lr[2], best_acc_2, best_epoch_2 + 1))
    # best_acc_3, best_epoch_3, loss_list_3 = train(model, epochs, train_loader, 3)
    # print("Training Result:")
    # print('lr={} : best_acc: {:.6f}% best_epoch: {}'.format(lr[3], best_acc_3, best_epoch_3 + 1))
    # print('Training finished!\n')

    # Start test
    print('Start Test:\n')
    model.load_state_dict(torch.load(r'pokemon_ckp.pth'))
    model.eval()
    test_acc = evaluate(model, test_loader)
    print('The accuracy of test set is {:.6f}%'.format(test_acc))

    # Draw loss function curve
    if draw == 1:
        x = np.arange(1, epochs + 1).astype(dtype=np.str)
        plt.figure()
        plt.title('Loss vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(x, ac_list_0, color='black', label='Accuracy_lr_1e-5')
        plt.plot(x, loss_list_0, color='green', label='Loss_lr_1e-5')
        # plt.plot(x, loss_list_1, color='red', label='Loss_lr_1e-4')
        # plt.plot(x, loss_list_2, color='blue', label='Loss_lr_1e-3')
        # plt.plot(x, loss_list_3, color='yellow', label='Loss_lr_1e-2')
        plt.legend()
        plt.show()
