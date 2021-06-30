'''
@File    :   progress_bar.py
@Time    :   2021/06/07 16:24:50
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.
'''

# here put the import lib
import sys
import re
from datetime import datetime

# 进度条类
class Train_ProgressBar(object):
    DEFAULT = "Progress: %(bar)s %(percent)3d%%"

    def __init__(self, mode='train', epoch=None, total_epoch=None, current_loss=None, current_lr=0, save_model_path=None, total=None, current=None, width=76, symbol=">", output=sys.stderr):
        assert len(symbol) == 1
        self.mode = mode    # 文字显示模式
        self.total = total  # batch数量
        self.symbol = symbol  # bar显示符号，默认为<
        self.output = output  # 文件输出方式
        self.width = width  # bar长度，默认为80
        self.current = current  # 记录当前的batch数
        self.epoch = epoch  # 记录当前的epoch
        self.total_epoch = total_epoch  # 输入的总epoch数
        self.current_loss = current_loss  # 输入记录的当前loss
        self.save_model_path = save_model_path  # 网络名
        self.current_lr = current_lr  # 记录当前的学习率

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        bar = "[" + self.symbol * size + " " * (self.width - size) + "]"

        args = {
            "mode": self.mode,
            "total": self.total,
            "bar": bar,
            "current": self.current,
            "percent": percent * 100,
            "current_loss": self.current_loss,
            "epoch": self.epoch + 1,
            "epochs": self.total_epoch,
            "current_lr": self.current_lr
        }
        message = "\033[1;32;40m%(mode)s  Epoch: %(epoch)d/%(epochs)d %(bar)s\033[0m  [ Loss %(current_loss)f lr: %(current_lr)f ]  %(current)d/%(total)d \033[1;32;40m[%(percent)3d%%]\033[0m" % args
        self.write_message = "Epoch: %(epoch)d/%(epochs)d [ Current: Loss %(current_loss)f lr: %(current_lr)f ]" % args
        print("\r" + message, file=self.output, end="")

    def done(self):
        self.current = self.total
        self()
        print("", file=self.output)
        # # 向logs输出当前的结果
        with open("./logs/%s.txt" % (self.save_model_path.split('/')[-1]), "a") as f:
            print(self.write_message, file=f)

class Test_ProgressBar(object):
    def __init__(self, mode='test', total=None,save_model_path=None,current=None, width=29, symbol=">", output=sys.stderr):
        assert len(symbol) == 1
        self.mode = mode  # 文字显示模式
        self.total = total  # batch数量
        self.symbol = symbol  # bar显示符号，默认为<
        self.output = output  # 文件输出方式
        self.width = width  # bar长度
        self.current = current  # 记录当前的batch数
        self.val = [0.0]*5
        self.save_model_path = save_model_path  # 网络名

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        bar = "[" + self.symbol * size + " " * (self.width - size) + "]"

        args = {
            "mode": self.mode,
            "total": self.total,
            "bar": bar,
            "current": self.current,
            "percent": percent * 100,
            "dice": self.val[0],
            "precision": self.val[1],
            "jaccard": self.val[2],
            "sensitivity": self.val[3],
            "specificity": self.val[4]
        }
        message = "\033[1;32;40m%(mode)s  %(bar)s\033[0m  [ Dice:%(dice)f Precision:%(precision)f Jaccard:%(jaccard)f Sensitivity:%(sensitivity)f Specificity:%(specificity)f ]  %(current)d/%(total)d \033[1;32;40m[%(percent)3d%%]\033[0m" % args
        self.write_message = "Dice:%(dice)f Precision:%(precision)f Jaccard:%(jaccard)f Sensitivity:%(sensitivity)f Specificity:%(specificity)f" % args
        print("\r" + message, file=self.output, end="")

    def done(self):
        self.current = self.total # 保证进度条进行完
        self()
        print("", file=self.output)
        print("")
        # 向logs输出当前的结果
        with open("./logs/%s_test_indicator.txt" % (self.save_model_path.split('/')[-1]), "a") as f:
            print("Test Result:",file=f)
            print(self.write_message, file=f)


class Val_ProgressBar(object):
    def __init__(self, mode='val', save_model_path=None, total=None, current=None, width=29, symbol=">", output=sys.stderr):
        assert len(symbol) == 1
        self.mode = mode  # 文字显示模式
        self.total = total  # batch数量
        self.symbol = symbol  # bar显示符号，默认为<
        self.output = output  # 文件输出方式
        self.width = width  # bar长度
        self.current = current  # 记录当前的batch数
        self.save_model_path = save_model_path  # 网络名
        self.val = [0.]*5


    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        bar = "[" + self.symbol * size + " " * (self.width - size) + "]"

        args = {
            "mode": self.mode,
            "total": self.total,
            "bar": bar,
            "current": self.current,
            "percent": percent * 100,
            "dice": self.val[0],
            "Accuracy": self.val[1],
            "jaccard": self.val[2],
            "sensitivity": self.val[3],
            "specificity": self.val[4]
        }
        message = "\033[1;32;40m%(mode)s  %(bar)s\033[0m  [ Dice:%(dice)f Accuracy:%(Accuracy)f Jaccard:%(jaccard)f Sensitivity:%(sensitivity)f Specificity:%(specificity)f ]  %(current)d/%(total)d \033[1;32;40m[%(percent)3d%%]\033[0m" % args
        self.write_message = "Metric: Dice:%(dice)f Accuracy:%(Accuracy)f Jaccard:%(jaccard)f Sensitivity:%(sensitivity)f Specificity:%(specificity)f" % args
        print("\r" + message, file=self.output, end="")

    def done(self):
        self.current = self.total
        self()
        print("", file=self.output)
        # # 向logs输出当前的结果
        with open("./logs/%s.txt" % (self.save_model_path.split('/')[-1]), "a") as f:
            print(self.write_message, file=f)

class Val_ProgressBar_2(object):
    def __init__(self, mode='val', save_model_path=None, total=None, current=None, width=29, symbol=">", output=sys.stderr):
        assert len(symbol) == 1
        self.mode = mode  # 文字显示模式
        self.total = total  # batch数量
        self.symbol = symbol  # bar显示符号，默认为<
        self.output = output  # 文件输出方式
        self.width = width  # bar长度
        self.current = current  # 记录当前的batch数
        self.save_model_path = save_model_path  # 网络名
        self.val = [(0.,0.)]*5


    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        bar = "[" + self.symbol * size + " " * (self.width - size) + "]"

        args = {
            "mode": self.mode,
            "total": self.total,
            "bar": bar,
            "current": self.current,
            "percent": percent * 100,
            "srf_dice": self.val[0][0],
            "srf_accuracy": self.val[1][0],
            "srf_jaccard": self.val[2][0],
            "srf_sensitivity": self.val[3][0],
            "srf_specificity": self.val[4][0],
            "cnv_dice": self.val[0][1],
            "cnv_accuracy": self.val[1][1],
            "cnv_jaccard": self.val[2][1],
            "cnv_sensitivity": self.val[3][1],
            "cnv_specificity": self.val[4][1],
            "avg_dice": (self.val[0][0]+self.val[0][1])/2,
            "avg_accuracy": (self.val[1][0]+self.val[1][1])/2,
            "avg_jaccard": (self.val[2][0]+self.val[2][1])/2,
            "avg_sensitivity": (self.val[3][0]+self.val[3][1])/2,
            "avg_specificity": (self.val[4][0]+self.val[4][1])/2,
        }
        message = "\033[1;32;40m%(mode)s  %(bar)s\033[0m Average Metirc: [ Dice:%(avg_dice)f Accuracy:%(avg_accuracy)f Sensitivity:%(avg_sensitivity)f Specificity:%(avg_specificity)f ] %(current)d/%(total)d \033[1;32;40m[%(percent)3d%%]\033[0m" % args
                     
        self.write_message = "Metric for CNV: [ Dice:%(cnv_dice)f Accuracy:%(cnv_accuracy)f Jaccard:%(cnv_jaccard)f Sensitivity:%(cnv_sensitivity)f Specificity:%(cnv_specificity)f ]\
              \nMetric for SRF: [ Dice:%(srf_dice)f Accuracy:%(srf_accuracy)f Jaccard:%(srf_jaccard)f Sensitivity:%(srf_sensitivity)f Specificity:%(srf_specificity)f ]\n \
              \nAverage Metirc: [ Dice:%(avg_dice)f Accuracy:%(avg_accuracy)f Jaccard:%(avg_jaccard)f Sensitivity:%(avg_sensitivity)f Specificity:%(avg_specificity)f ]" % args
        print("\r" + message, file=self.output, end="")

    def done(self):
        
        args = {
            "srf_dice": self.val[0][0],
            "srf_accuracy": self.val[1][0],
            "srf_jaccard": self.val[2][0],
            "srf_sensitivity": self.val[3][0],
            "srf_specificity": self.val[4][0],
            "cnv_dice": self.val[0][1],
            "cnv_accuracy": self.val[1][1],
            "cnv_jaccard": self.val[2][1],
            "cnv_sensitivity": self.val[3][1],
            "cnv_specificity": self.val[4][1],
            "avg_dice": (self.val[0][0]+self.val[0][1])/2,
            "avg_accuracy": (self.val[1][0]+self.val[1][1])/2,
            "avg_jaccard": (self.val[2][0]+self.val[2][1])/2,
            "avg_sensitivity": (self.val[3][0]+self.val[3][1])/2,
            "avg_specificity": (self.val[4][0]+self.val[4][1])/2,
        }
        self.current = self.total
        self()
        print("\nMetric for CNV: [ Dice:%(cnv_dice)f Accuracy:%(cnv_accuracy)f Jaccard:%(cnv_jaccard)f Sensitivity:%(cnv_sensitivity)f Specificity:%(cnv_specificity)f ]\
              \nMetric for SRF: [ Dice:%(srf_dice)f Accuracy:%(srf_accuracy)f Jaccard:%(srf_jaccard)f Sensitivity:%(srf_sensitivity)f Specificity:%(srf_specificity)f ]\n"% args)
        print("", file=self.output)
        # # 向logs输出当前的结果
        with open("./logs/%s.txt" % (self.save_model_path.split('/')[-1]), "a") as f:
            print(self.write_message, file=f)

class Val_ProgressBar_dice_only(object):
    def __init__(self, mode='val', save_model_path=None, total=None, current=None, width=29, symbol=">", output=sys.stderr):
        assert len(symbol) == 1
        self.mode = mode  # 文字显示模式
        self.total = total  # batch数量
        self.symbol = symbol  # bar显示符号，默认为<
        self.output = output  # 文件输出方式
        self.width = width  # bar长度
        self.current = current  # 记录当前的batch数
        self.save_model_path = save_model_path  # 网络名
        self.val = [(0.,0.)]


    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        bar = "[" + self.symbol * size + " " * (self.width - size) + "]"

        args = {
            "mode": self.mode,
            "total": self.total,
            "bar": bar,
            "current": self.current,
            "percent": percent * 100,
            "srf_dice": self.val[0],
            "cnv_dice": self.val[1],
            "avg_dice": (self.val[0]+self.val[1])/2,
        }
        message = "\033[1;32;40m%(mode)s  %(bar)s\033[0m Average Metirc: [ Dice:%(avg_dice)f ] %(current)d/%(total)d \033[1;32;40m[%(percent)3d%%]\033[0m" % args
                     
        self.write_message = "Metric for CNV: [ Dice:%(cnv_dice)f ] Metric for SRF: [ Dice:%(srf_dice)f ]\n \
              \nAverage Metirc: [ Dice:%(avg_dice)f ]" % args
        print("\r" + message, file=self.output, end="")

    def done(self):
        
        args = {
            "srf_dice": self.val[0],
            "cnv_dice": self.val[1],
            "avg_dice": (self.val[0]+self.val[1])/2,
        }
        self.current = self.total
        self()
        print("\nMetric for CNV: [ Dice:%(cnv_dice)f ] Metric for SRF: [ Dice:%(srf_dice)f]\n"% args)
        print("", file=self.output)
        # # 向logs输出当前的结果
        with open("./logs/%s.txt" % (self.save_model_path.split('/')[-1]), "a") as f:
            print(self.write_message, file=f)


class Test_ProgressBar_2(object):
    def __init__(self, mode='val', save_model_path=None, total=None, current=None, width=29, symbol=">", output=sys.stderr):
        assert len(symbol) == 1
        self.mode = mode  # 文字显示模式
        self.total = total  # batch数量
        self.symbol = symbol  # bar显示符号，默认为<
        self.output = output  # 文件输出方式
        self.width = width  # bar长度
        self.current = current  # 记录当前的batch数
        self.save_model_path = save_model_path  # 网络名
        self.val = [(0.,0.)]*5


    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        bar = "[" + self.symbol * size + " " * (self.width - size) + "]"

        args = {
            "mode": self.mode,
            "total": self.total,
            "bar": bar,
            "current": self.current,
            "percent": percent * 100,
            "srf_dice": self.val[0][0],
            "srf_accuracy": self.val[1][0],
            "srf_jaccard": self.val[2][0],
            "srf_sensitivity": self.val[3][0],
            "srf_specificity": self.val[4][0],
            "cnv_dice": self.val[0][1],
            "cnv_accuracy": self.val[1][1],
            "cnv_jaccard": self.val[2][1],
            "cnv_sensitivity": self.val[3][1],
            "cnv_specificity": self.val[4][1],
            "avg_dice": (self.val[0][0]+self.val[0][1])/2,
            "avg_accuracy": (self.val[1][0]+self.val[1][1])/2,
            "avg_jaccard": (self.val[2][0]+self.val[2][1])/2,
            "avg_sensitivity": (self.val[3][0]+self.val[3][1])/2,
            "avg_specificity": (self.val[4][0]+self.val[4][1])/2,
        }
        message = "\033[1;32;40m%(mode)s  %(bar)s\033[0m Average Metirc: [ Dice:%(avg_dice)f Accuracy:%(avg_accuracy)f Sensitivity:%(avg_sensitivity)f Specificity:%(avg_specificity)f ] %(current)d/%(total)d \033[1;32;40m[%(percent)3d%%]\033[0m" % args
                     
        self.write_message = "Metric for CNV: [ Dice:%(cnv_dice)f Accuracy:%(cnv_accuracy)f Jaccard:%(cnv_jaccard)f Sensitivity:%(cnv_sensitivity)f Specificity:%(cnv_specificity)f ]\
              \nMetric for SRF: [ Dice:%(srf_dice)f Accuracy:%(srf_accuracy)f Jaccard:%(srf_jaccard)f Sensitivity:%(srf_sensitivity)f Specificity:%(srf_specificity)f ]\
              \nAverage Metirc: [ Dice:%(avg_dice)f Accuracy:%(avg_accuracy)f Jaccard:%(avg_jaccard)f Sensitivity:%(avg_sensitivity)f Specificity:%(avg_specificity)f ]\n" % args
        print("\r" + message, file=self.output, end="")

    def done(self):
        
        args = {
            "srf_dice": self.val[0][0],
            "srf_accuracy": self.val[1][0],
            "srf_jaccard": self.val[2][0],
            "srf_sensitivity": self.val[3][0],
            "srf_specificity": self.val[4][0],
            "cnv_dice": self.val[0][1],
            "cnv_accuracy": self.val[1][1],
            "cnv_jaccard": self.val[2][1],
            "cnv_sensitivity": self.val[3][1],
            "cnv_specificity": self.val[4][1],
            "avg_dice": (self.val[0][0]+self.val[0][1])/2,
            "avg_accuracy": (self.val[1][0]+self.val[1][1])/2,
            "avg_jaccard": (self.val[2][0]+self.val[2][1])/2,
            "avg_sensitivity": (self.val[3][0]+self.val[3][1])/2,
            "avg_specificity": (self.val[4][0]+self.val[4][1])/2,
        }
        self.current = self.total
        self()
        print("\nMetric for CNV: [ Dice:%(cnv_dice)f Accuracy:%(cnv_accuracy)f Jaccard:%(cnv_jaccard)f Sensitivity:%(cnv_sensitivity)f Specificity:%(cnv_specificity)f ]\
              \nMetric for SRF: [ Dice:%(srf_dice)f Accuracy:%(srf_accuracy)f Jaccard:%(srf_jaccard)f Sensitivity:%(srf_sensitivity)f Specificity:%(srf_specificity)f ]\n"% args)
        print("", file=self.output)
        # # 向logs输出当前的结果
        with open("./logs/%s.txt" % (self.save_model_path.split('/')[-1]), "a") as f:
            print(self.write_message, file=f)

