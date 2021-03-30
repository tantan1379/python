from time import sleep


class ProgressBar(object):
    
    def __init__(self, mode, epoch=None, total_epoch=None, total=None, current=None, width=50, model_name="ResNet18",symbol=">"):
        self.mode = mode
        self.epoch = epoch
        self.total_epoch = total_epoch
        self.total = total
        self.current = current
        self.width = width
        self.symbol = symbol
        self.model_name = model_name

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(percent * self.width)
        bar = "[" + self.symbol * size + " " * (self.width - size) + "]"
        args = {
            "percent": percent * 100,
            "current": self.current,
            "total": self.total,
            "bar": bar,
            "mode": self.mode,
            "epoch": self.epoch + 1,
            "epochs": self.total_epoch,
            "model_name":self.model_name
        }
        message = "\033[1;32;40m%(mode)s Epoch: %(epoch)d/%(epochs)d %(bar)s\033[0m [Current: LOSS=0.1 Top1=0.9] %(current)d/%(total)d \033[1;32;40m[ %(percent)3d%% ]" % args
        self.write_message = "%(mode)s Epoch: %(epoch)d/%(epochs)d %(bar)s [Current: LOSS=0.1 Top1=0.9] %(current)d/%(total)d [ %(percent)3d%% ]" % args
        print("\r" + message, end="")

    def done(self):
        self.current = self.total
        self()
        print("")
        with open("%s.txt"%self.model_name,"a") as f:
            print(self.write_message,file=f)
        # with open("./logs/%s.txt" % self.model_name, "a") as f:
        #     print(self.write_message, file=f)


if __name__ == "__main__":
    
    progress = ProgressBar("Train", total_epoch=10,model_name="ResNet159")
    for i in range(10):
        progress.total = 50
        progress.epoch = i
        for x in range(50):
            progress.current = x
            progress()
            sleep(0.1)
        progress.done()
