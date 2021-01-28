# Deep Learning
## Pytorch Basic
#### Basic knowledge
1、tensor比ndarray的优势在于可以使用GPU进行加速计算   
2、Variable相当于包装tensor的盒子，里面包含着tensor及对他的操作。Tensor不能反向传播，而Variable可以反向传播。         
调用方法：`from torch.autograd import Variable`   
3、有多少个卷积核就有多少个feature map，一个feature map对应图像被提取的一种特征

#### Create tensor
1、`torch.Tensor(*sizes)`:随机创建指定形状的Tensor   
2、`torch.Tensor(data)`:同torch.FloatTensor()，将List转换为Tensor，生成单精度浮点类型的张量   
3、`torch.tensor(data)`:从data中的数据部分做拷贝，根据原始数据类型生成相应的torch.LongTensor，torch.FloatTensor，torch.DoubleTensor   
4、`torch.ones(*size)`:创建全1的Tensor   
5、`torch.zeros(*size)`:创建全0的Tensor   
6、`torch.eye(*size)`:创建对角Tensor   
7、`torch.arange(s, e, step)`:生成从s到e，步长为step的一维Tensor    
8、`torch.randn(*size)`：标准分布   
9、`torch.rand(*size)`：均匀分布   
10、`tensor.view()`：调整tensor的形状（常用）  
e.g.:`x=x.view(x.size(0),-1)`   
其中x.size(0)为batch_size，将四维(batch_size,Channel,Height,Width)的张量flatten为二维的张量，作为全连接层的输入      
11、`tensor.unsqueeze()`:为Tensor添加维度   
12、`tensor.squeeze()`:为Tensor减少维度

#### Convert
1、`data.cuda()`：cpu –> gpu   
`data.cpu()`：gpu –> cpu   
`data.numpy()`：Tensor –> Numpy.ndarray     
`torch.from_numpy(data)`：Numpy.ndarray –> Tensor   
`a.item()`：对只含一个元素的tensor使用，将tensor转换为python对象类型,   
`newtensor = tensor.int()`：将tensor投射指定类型：

#### Property
查看Tensor的大小：`tensor.size()`   
查看Tensor的大小：`tensor.shape`   
统计Tensor的元素个数：`Tensor.numel()`   

---

## Numpy
**（以下用np表示）**  
1、`data.numpy()`   
将tensor类型转换为numpy类型;   
2、`np.equal(x1, x2)`    
Return (x1 == x2) element-wise.   
比较两个数组的值(若相等,在对应位置上取True，不等取False)；   
3、`np.all()`Test whether all array elements along a given axis evaluate to True.
4、`if array`判断numpy数组是否为空，将列表作为布尔值，若不为空返回True，否则视为False;

---
## torchvision
1、在datasets模块中保存着各类数据集   
2、在models模块中保存搭建好的网络（可以不加载数据）   
3、在transforms模块中封装了一些处理数据的方法   
Note:    
1.torchvision的datasets的输出是[0,1]的PILImage，所以我们需要归一化为[-1,1]的Tensor   
2.数据增强虽然会使训练过程收敛变慢，但可以提高测试集准确度，防止过拟合，提高模型的泛化能力。  

---

## argparse 命令行解析工具
#### 创建句柄
`parser = argparse.ArgumentParser()`   创建一个命令解析器的句柄

#### 增加属性
`parser.add_argument("echo")`给实例增加一个属性   
e.g.   
default：设置参数的默认值   
type：把从命令行输入的结果转成设置的类型   
choice：允许的参数值（用列表限定）   
help: 显示帮助（命令中加-h)

**Notes:**   
1、直接用单引号括起来的表示定位参数而加有–为可选参数，并规定定位参数必选，可选参数可选。   
2、当'-'和'--'同时出现的时候，系统默认后者为参数名，但命令行输入可以不遵从这个规则

#### 实例化
`args = parser.parse_args()`把parser中设置的所有"add_argument"给返回到args子类实例当中

---

## Flag
1、`torch.backends.cudnn.benchmark = True`  
大部分情况下，设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。   
**Notes:**   
1、如果网络的输入数据维度或类型上变化不大，设置该Flag为True可以增加运行效率；   
2、如果网络的输入数据在每次iteration都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。

---

## Practical application   
1、计算网络的参数个数   
```
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

2、动态修改学习率   
```
for param_group in optimizer.param_groups:
    param_group["lr"] = lr
```
