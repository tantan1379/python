#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: main.py
# author: twh
# time: 2020/11/9 16:59
import torch.utils.data
import argparse
import vgg

if __name__ == '__main__':
    epochs = 5
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default='GPU', type=str, choices=['GPU', 'CPU'], help='选择使用cpu/gpu')
    args = parser.parse_args()
    # CPU或GPU运行
    if args.cuda == 'GPU':
        device = torch.device('cuda')
    elif args.cuda == 'CPU':
        device = torch.device('cpu')
    vgg.run(epochs,device)
