import sys
import re


class ProgressBar(object):
    DEFAULT = "Progress: %(bar)s %(percent)3d%%"

    def __init__(self, mode, epoch=None, total_epoch=None, current_loss=None, current_top1=None,current_lr=0, model_name=None, total=None, current=None, width=50, symbol=">", output=sys.stderr):
        assert len(symbol) == 1

        self.mode = mode  # 文字显示模式
        self.total = total  # batch数量
        self.symbol = symbol  # bar显示符号，默认为<
        self.output = output  # 文件输出方式
        self.width = width  # bar长度，默认为50
        self.current = current  # 记录当前的batch数
        self.epoch = epoch  # 记录当前的epoch
        self.total_epoch = total_epoch  # 输入的总epoch数
        self.current_loss = current_loss  # 输入记录的当前loss
        self.current_top1 = current_top1  # 输入记录的当前top1 acc
        self.model_name = model_name  # 网络名
        self.current_lr = current_lr # 记录当前的学习率

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
            "current_top1": self.current_top1,
            "epoch": self.epoch + 1,
            "epochs": self.total_epoch,
            "current_lr":self.current_lr
        }
        message = "\033[1;32;40m%(mode)s Epoch:  %(epoch)d/%(epochs)d %(bar)s\033[0m  [Current: Loss %(current_loss)f Top1: %(current_top1)f lr: %(current_lr).2e ]  %(current)d/%(total)d \033[1;32;40m[ %(percent)3d%% ]\033[0m" % args
        self.write_message = "%(mode)s Epoch:  %(epoch)d/%(epochs)d [Current: Loss %(current_loss)f Top1: %(current_top1)f lr: %(current_lr).2e ]" % args
        print("\r" + message, file=self.output, end="")

    def done(self):
        self.current = self.total
        self()
        print("", file=self.output)
        # 向logs输出当前的结果
        with open("./logs/%s.txt" % self.model_name, "a") as f:
            print(self.write_message, file=f)


if __name__ == "__main__":

    from time import sleep
    progress = ProgressBar("Train", total_epoch=150, model_name="resnet159")
    for i in range(150):
        progress.total = 50
        progress.epoch = i
        progress.current_loss = 0.15
        progress.current_top1 = 0.45
        for x in range(50):
            progress.current = x
            progress()
            sleep(0.1)
        progress.done()
