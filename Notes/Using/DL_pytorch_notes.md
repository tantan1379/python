# Deep Learning（pytorch)

### 1、Basic knowledge

1、tensor比ndarray的优势在于可以使用GPU进行加速计算

2、Variable相当于包装tensor的盒子，里面包含着tensor及对他的操作。Tensor不能反向传播，而Variable可以反向传播。
调用方法：`from torch.autograd import Variable` 

3、有多少个卷积核就有多少个feature map，一个feature map对应图像被提取的一种特征

4、output = ( input - K + 2 * P ) / S + 1 ，改变S可以改变输入的维度

5、经过trainloader加载后，图片的维度为(batchsize,channel,height,width)

6、DoubleTensor比FloatTensor有更高的精度，适合增强学习

------

### 2、Tensor operations：

#### Convert

1、cpu –> gpu: `data.cuda()`

2、gpu –> cpu：`data.cpu()`

3、Numpy.ndarray –> Tensor **（导入）**： `torch.from_numpy(data)`

4、Tensor –> Numpy.ndarray ：`data.numpy()`

5、Tensor -> DoubleTensor: `torch.set_default_tensor_tepe(torch.DoubleTensor)`

6、将List转换为Tensor，生成单精度浮点类型的张量：`torch.Tensor(data)` 同torch.FloatTensor()

7、根据原始数据类型生成相应的张量：`torch.tensor(data)`

8、将tensor转换为python对象类型：`a.item()`：对只含一个元素的tensor使用，,



#### Create tensor

1、随机创建指定形状的Tensor：`torch.Tensor(*sizes)`

2、生成从s到e(不包含e)，步长为step的一维Tensor ：`torch.arange(s, e, step)`
生成从s到e(包含e)，元素个数为steps的一维Tensor：`torch.linspace(s,e,steps)`

3、生成随机分布的Tensor:
标准分布(0,1正态分布) ：`torch.randn(*size)`
均匀分布：`torch.rand(*size)`

4、创建特殊的Tensor：`torch.ones(*size)` `torch.zeros(*size)` `torch.eyes(*size)`

5、创建具有相同值的Tensor:`torch.full(*size,val)` 如果size写[]，生成标量



#### Index&slice

1、选取指定维度进行切片：`a.index_select(dim,torch.tensor)`  2、冒号    3、省略号 



#### Dimension

1、调整Tensor的形状（常用）：`tensor.view()` e.g.:`x=x.view(x.size(0),-1)`
**notes:** 在神经网络中图像的维度为(batchsize,channel,height,width),一定要以这个顺序和逻辑进行view；view前后的size要相同

2、修改维度
（增维）：`tensor.unsqueeze(pos)` 在posi前的一个位置加一维
（减维）：`tensor.squeeze()`自动挤压所有值为1的维度 `tensor.squeeze(pos)`挤压（减去）pos位置的维度
（维度扩展）：`tensor.expand(*size)` [=] 
**note:** 1、自动复制broadcast tensor，变换前后维数不变；  2、需要扩张的维值必须为1；  3、如果某一位置填-1则表示该维保持不变

3、交换维度：
（二维）：`tensor.t()`
（多维）：`tensor.transpose(dim1,dim2)`
（通用）：`tensor.permute(dim1,dim2,dim3,...)`

4、Broadcast：
先在某一维度**之前**插入维度(unsqueeze)，再在大小为1的维度上进行扩张(expand)

5、topk:

`torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)`



#### Merge&Split

1、cat:`torch.cat([a,b],dim) `两个拼接的tensor必须在dim维之外的维度均相等



#### Property

查看Tensor的大小：`tensor.size()` 
查看Tensor的大小：`tensor.shape` 
统计Tensor的元素个数：`Tensor.numel()`   



#### Comparision

(1) tensor和Tensor(FloatTensor)的区别在于，tensor只能接受现有的数据，Tensor可以接受数据的维度()或数据([])
为避免混淆，使用时建议，要维度用大写Tensor，要具体数据用小写tensor

------

### 3、Numpy 
**（以下用np表示）**
1、`data.numpy()` 
将tensor类型转换为numpy类型; 
2、`np.equal(x1, x2)` 
Return (x1 == x2) element-wise. 
比较两个数组的值(若相等,在对应位置上取True，不等取False)；
3、`np.all()`Test whether all array elements along a given axis evaluate to True.
4、`if array`判断numpy数组是否为空，将列表作为布尔值，若不为空返回True，否则视为False;

---
### 4、torchvision
1、在datasets模块中保存着各类数据集
2、在models模块中保存搭建好的网络（可以不加载数据）
3、在transforms模块中封装了一些处理数据的方法
**Note:** 
1.torchvision的datasets的输出是[0,1]的PILImage，所以我们需要归一化为[-1,1]的Tensor
2.数据增强虽然会使训练过程收敛变慢，但可以提高测试集准确度，防止过拟合，提高模型的泛化能力。 

---

### 5、argparse 命令行解析工具
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

### 6、Flag
1、`torch.backends.cudnn.benchmark = True`
大部分情况下，设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
**Notes:**   
1、如果网络的输入数据维度或类型上变化不大，设置该Flag为True可以增加运行效率；
2、如果网络的输入数据在每次iteration都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。

---

### 7、Practical application   

1、计算网络的参数个数   
```
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

2、动态修改学习率   

```
for param_group in optimizer.param_groups:
    param_group["lr"] = lr
```

---

### 8、Others

在dataloader加载后的input和output张量维度为（batchsize,