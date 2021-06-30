import sys
import re
from datetime import datetime


class Train_ProgressBar(object):
    DEFAULT = "Progress: %(bar)s %(percent)3d%%"

    def __init__(self, mode='train', fold=None, epoch=None, top1=None,total_epoch=None, current_loss=None, current_lr=0, model_name=None, total=None, current=None, width=80, symbol=">", output=sys.stderr):
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
        self.top1 = top1
        self.model_name = model_name  # 网络名
        self.current_lr = current_lr  # 记录当前的学习率
        self.fold = fold
        # current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        # with open("./logs/train/%s_%s.txt" % (self.model_name, self.net_index), "a") as f:
        #     print(current_time, file=f)
        #     print("fold:"+fold,file=f)

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        bar = "[" + self.symbol * size + " " * (self.width - size) + "]"

        args = {
            "fold": self.fold,
            "mode": self.mode,
            "total": self.total,
            "bar": bar,
            "current": self.current,
            "percent": percent * 100,
            "current_loss": self.current_loss,
            "top1":self.top1,
            "epoch": self.epoch + 1,
            "epochs": self.total_epoch,
            "current_lr": self.current_lr
        }
        message = "\033[1;32;40mfold:%(fold)d %(mode)s  Epoch: %(epoch)d/%(epochs)d %(bar)s\033[0m  [ Loss: %(current_loss)f Top1: %(top1)f lr: %(current_lr)f ]  %(current)d/%(total)d \033[1;32;40m[%(percent)3d%%]\033[0m" % args
        # self.write_message = "fold:%(fold)d  Epoch: %(epoch)d/%(epochs)d [ Current: Loss %(current_loss)f lr: %(current_lr)f ]" % args
        print("\r" + message, file=self.output, end="")

    def done(self):
        self.current = self.total
        self()
        print("", file=self.output)
        # # 向logs输出当前的结果
        # with open("./logs/train/%s_%s.txt" % (self.model_name,self.net_index), "a") as f:
        #     print(self.write_message, file=f)


class Val_ProgressBar(object):
    def __init__(self, mode='val', fold=None, epoch=None, model_name=None,  total=None, current=None, width=29, symbol=">", output=sys.stderr):
        assert len(symbol) == 1
        self.mode = mode  # 文字显示模式
        self.fold = fold  # 当前折数
        self.total = total  # batch数量
        self.symbol = symbol  # bar显示符号，默认为<
        self.output = output  # 文件输出方式
        self.width = width  # bar长度
        self.current = current  # 记录当前的batch数
        self.model_name = model_name  # 网络名
        self.val = [0.0]*4
        self.epoch = epoch


    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        bar = "[" + self.symbol * size + " " * (self.width - size) + "]"

        args = {
            "fold": self.fold,
            "epoch": self.epoch,
            "mode": self.mode,
            "total": self.total,
            "bar": bar,
            "current": self.current,
            "percent": percent * 100,
            "acc": self.val[0],
            "loss":self.val[1]
            # "pre":self.val[1],
            # "rec":self.val[2],
            # "f1":self.val[3]
        }
        message = "\033[1;32;40mfold:%(fold)d %(mode)s  %(bar)s\033[0m  [ Accuracy:%(acc)f Loss:%(loss)f  %(current)d/%(total)d \033[1;32;40m[%(percent)3d%%]\033[0m" % args
        # message = "\033[1;32;40mfold:%(fold)d %(mode)s  %(bar)s\033[0m  [ Accuracy:%(acc)f Precision:%(pre)f Recall:%(rec)f F1:%(f1)f %(current)d/%(total)d \033[1;32;40m[%(percent)3d%%]\033[0m" % args
        self.write_message = "fold:%(fold)d epoch:%(epoch)d   Accuracy:%(acc)f Loss:%(loss)f" % args
        print("\r" + message, file=self.output, end="")

    def done(self):
        self.current = self.total
        self()
        print("", file=self.output)
        print("")
        # # 向logs输出当前的结果
        with open("./logs/%s.txt" % (self.model_name), "a") as f:
            print(self.write_message, file=f)
