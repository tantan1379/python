import argparse
import datetime
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

from utils.config import Model_choice


def SearchFreeGPU(interval=60, threshold=0.5):
    while True:
        qargs = ['index', 'memory.free', 'memory.total']
        cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
        results = os.popen(cmd).readlines()
        gpus = pd.DataFrame(np.zeros([len(results), 3]), columns=qargs)
        for i, line in enumerate(results):
            info = line.strip().split(',')
            gpus.loc[i, 'index'] = info[0]
            gpus.loc[i, 'memory.free'] = float(info[1].upper().strip().replace('MIB', ''))
            gpus.loc[i, 'memory.total'] = float(info[2].upper().strip().replace('MIB', ''))
            gpus.loc[i, 'Freerate'] = gpus.loc[i, 'memory.free'] / gpus.loc[i, 'memory.total']

        maxrate = gpus.loc[:, "Freerate"].max()
        index = gpus.loc[:, "Freerate"].idxmax()
        if maxrate > threshold:
            print('GPU index is: {}'.format(index))
            return str(index)
        else:
            print('Searching Free GPU...')
            time.sleep(interval)

# Tasks_list = ['Mini_Unet_V0']
# Tasks_list = ['Baseline_V1_Contrast']
# Tasks_list = ['Baseline_V1', 'Crop_Mini_Unet_V0', 'Mini_Unet_V0']
# Tasks_list = ['Mini_Unet_V0']
# Tasks_list = ['Mini_Unet_Dcm_V0', 'Mini_Unet_Dcm_V1']
# Tasks_list = ['Mini_Unet_Dcm_V0']
Tasks_list = ['Mini_Unet_Dcm_For_Roi_V0']


while True:
    freeGPU = SearchFreeGPU(interval=1, threshold=0.5)
    if freeGPU == '0':
        for Task in Tasks_list:
            # cmd = 'CUDA_VISIBLE_DEVICES=' + freeGPU + ' python train.py ' + Task + ' --fold=3'
            cmd = 'CUDA_VISIBLE_DEVICES=' + freeGPU + ' python train_roi.py ' + Task + ' --fold=3'
            # cmd = 'CUDA_VISIBLE_DEVICES=' + freeGPU + ' python train_3d.py ' + Task
            print(cmd)
            os.system(cmd)
            print('*'*100)
        break
    else:
        continue


# for Task in Tasks_list:
#     cmd = 'CUDA_VISIBLE_DEVICES=' + '1' + ' python train.py ' + Task + ' --fold=3'
#     # cmd = 'CUDA_VISIBLE_DEVICES=' + freeGPU + ' python train_3d.py ' + Task
#     print(cmd)
#     os.system(cmd)
#     print('*' * 100)


