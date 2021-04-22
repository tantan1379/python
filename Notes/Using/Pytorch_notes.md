# Pytorch

## Basic knowledge

* tensor比ndarray的优势在于可以使用GPU进行加速计算
* 有多少个卷积核就有多少个feature map，一个feature map对应图像被提取的一种特征
* output = ( input - K + 2 * P ) / S + 1 ，改变S可以改变输入的维度
* DoubleTensor比FloatTensor有更高的精度，适合增强学习
* 需要将model先移动到cuda后，再创建optimizer
* 应该在optimizer更新后，再对scheduler进行更新
* tensor和Tensor(FloatTensor)的区别在于，tensor只能接受现有的数据，Tensor可以接受数据的维度()或数据([])
* 通常 a.方法和torch.方法(a)可以替换

* 一些shape问题：
  （1）经过trainloader加载后，loader的维度为(batchsize,channel,height,width)
  （2）dim=?的意思就是删除某一维度
  （3）对于tensor，.shape和.size()是相同的都输出张量的形状，.dim()则表示有多少个维度，.shape[0]和.size(0)也是一样的
  （4）PIL读入的图片channel在最后，torch的channel在高宽的前面
  （5）pytorch中，参数矩阵w一般将输出后的通道写前面，即y=x@w.t() 注意：.t()方法只适合于2d的tensor
  （6）dataloader的迭代中，每次循环（每个batch）输入到网络的img的shape为(batchsize,channel,height,width)，label(target)的shape为(batchsize,)，经过model输出后output的shape为(batchsize,num_class)[实际上output是用数值大小描述每种类预测的概率，需要通过max等函数得到真实的预测值]

------

## Tensor

**Convert**

* cpu-gpu：`data.cuda()`
  gpu-cpu：`data.cpu()`

* 数组-tensor： `torch.from_numpy(data)`  

* Tensor-数组 ：`data.numpy()`

* 将tensor转换为python对象类型：`a.item()`：对只含一个元素的tensor使用

**Create tensor**

* 随机创建指定形状的Tensor：`torch.Tensor(sizes)` 直接写维度

* 创建指定值的Tensor: `torch.tensor(data)` 或 `torch.Tensor(data)`

* 创建指定的Tensor: `torch.zeros(*sizes, requires_grad=False`
  `torch.ones(*sizes, requires_grad=False)`
  `torch.eye(n, requires_grad=False)`
  `torch.empty(*sizes, requires_grad=False)`
  `torch.full(*size, fill_value, requires_grad=False)`

**Index&slice**

1、选取指定维度进行切片：`a.index_select(dim,torch.tensor)`  2、冒号    3、省略号 

**Dimension**

* 调整Tensor的形状（常用）：`tensor.view()` e.g.:`x=x.view(x.size(0),-1)`

  注意点：在神经网络中图像的维度为(batchsize,channel,height,width),一定要以这个顺序和逻辑进行view；view前后的size要相同

* 修改维度
  （增维）：`tensor.unsqueeze(pos)` 在posi前的一个位置加一维
  （减维）：`tensor.squeeze()`自动挤压所有值为1的维度 `tensor.squeeze(pos)`挤压（减去）pos位置的维度
  （维度扩展）：`tensor.expand(*size)` [=] 
  注意点：1、自动复制broadcast tensor，变换前后维数不变；  2、需要扩张的维值必须为1；  3、如果某一位置填-1则表示该维保持不变

* 交换维度：
  （二维）：`tensor.t()`
  （多维）：`tensor.transpose(dim1,dim2)`  常用于图片和torch之间的维度交换
  （通用）：`tensor.permute(dim1,dim2,dim3,...)`

**Merge&Split**

* cat:`torch.cat([a,b],dim) `两个拼接的tensor必须在dim维之外的维度均相等


**Math**

* 元素乘：`a*b`
* 矩阵乘：`torch.matmul(a,b) `或 `a@b`   当维度高于2时，只取最后两维进行运算
* 元素幂：`a.pow(2)` 或 `a**2`
* 近似值：向下取整：`a.floor()` 向上取整：`a.ceil()` 取整：`a.trunc()` 取小数：`a.frac()` 四舍五入：`a.round()`
* 裁剪：`a.clamp(min,max)` 只有一个参数表示只限定最小值
* 求范数：`a.norm(level,dim)`
* 统计学：
  `a.min()`  `a.max() ` `a.mean()`  `a.prod()`(累乘)  `a.sum()` 如果没有dim参数，则会自动展平再计算
* 最大值：
  `torch.max(input,dim)` 返回有两个tensor的元组：1、第dim维最大的values   2、该最大values所在该dim的索引，两者shape一致
  `torch.argmax(input,dim)`与上类似，只返回最大值所在的索引
  `torch.topk(input,k,dim)`沿给定dim维度返回输入张量input中 k 个最大值(含两个tensor的元组)与torch.max类似
* 比较：torch.eq(a,b)

**Property**

* 查看Tensor的大小：`tensor.size()` 或 `tensor.shape` 
* 统计Tensor的元素个数：`Tensor.numel()` 

**Advanced**

* `torch.where(condition,x,y)` 判断condition中是否为True，成立取x中元素，不成立取y中元素
* `torch.gather(input,dim,index)` 收集输入的特定维度指定位置的数值

---



## Numpy 
**（以下用np表示）** 

* `np.equal(x1, x2)` 
  Return (x1 == x2) element-wise. 
  比较两个数组的值(若相等,在对应位置上取True，不等取False)；

* `np.all()`Test whether all array elements along a given axis evaluate to True.

* `if array`判断numpy数组是否为空，将列表作为布尔值，若不为空返回True，否则视为False;

* `np.newaxis`用于给数组增加维度：e.g.`aa=a[:,np.newaxis]`在第一维后加一维(1)
* `np.hstack(*list，dim)`用于拼接矩阵

---
## Torchvision
1、在datasets模块中保存着各类数据集
2、在models模块中保存搭建好的网络（可以不加载数据）
3、在transforms模块中封装了一些处理数据的方法
**Note:** 
1.torchvision的datasets的输出是[0,1]的PILImage，所以我们需要归一化为[-1,1]的Tensor
2.数据增强虽然会使训练过程收敛变慢，但可以提高测试集准确度，防止过拟合，提高模型的泛化能力。 

---

## DataLoader

参数表：

```python
class torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=<function default_collate>,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None)
```

shuffle：设置为True的时候，每个epoch都会打乱数据集 
collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能 
drop_last：告诉如何处理数据集长度除于batch_size余下的数据。True就抛弃，否则保留

---

## Pandas

Pandas的基础结构可以分为两种：**数据框和序列**。

**数据框（DataFrame）**是拥有轴标签的二维链表，换言之数据框是拥有标签的行和列组成的矩阵 - 列标签位列名，行标签为索引。Pandas中的行和列是Pandas序列 - 拥有轴标签的一维链表。

1、`iterrows()` 是在数据框中的行进行迭代的一个生成器，它返回每行的索引及一个包含行本身的对象。

2、dataframe的两种**加载**方式
（1）以字典形式载入：`df = pd.DataFrame({"a":arr1,"b":arr2})`
（2）将整个数组传入：`df = pd.DataFrame(data,columns=['a','b'])`注意，data的列数必须与columns匹配

3、dataframe的切片
（1）整数索引切片（前闭后开，不能单条）：`df[0:1]`
（2）标签索引切片（前闭后闭）：`df[:'a']`
（3）布尔数组索引：`df[[True,False]]` 
（选取age值大于30的行）：`df[df['age']>30]`
（选取出所有age大于30，且isMarried为no的行）：`df[(df['age']>30) & (df['isMarried']=='no')]` 注意&为按位与
（4）列选取：`df[['name','age']]`
（选取指定列）：`df[lambda df: df.columns[0]]`

4、dataframe的**区域选取**
（1）整数索引选取：`df.iloc[[1,3,5], :]` 
（2）标签索引选取：`df.loc[['a','b','c'], :]`

5、dataframe的**单元格选取**
（1）标签索引选取：df.at['b','name']
（2）整数索引选取：df.iat[1,0]

6、其他注意事项
（1）如果返回值包括单行多列或多行单列时，返回值为Series对象；如果返回值包括多行多列时，返回值为DataFrame对象；如果返回值仅为一个单元格（单行单列）时，返回值为基本数据类型
（2）df[]用于选取行和列数据，iloc和loc用于选取区域

---

## Optimizer & Scheduler & Loss_fn

**优化器设置：**

```python
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optimizer = torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
```

**学习率衰减设置：**

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
for epoch in range(epochs):
	train()
    validate()
    scheduler.step()
    
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_StepLR, step_size=step_size, gamma=0.65)
for epoch in range(epochs):
	train()
    validate()
    scheduler.step()
```

**损失函数设置：**

```python
criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.BCELoss()
```



## Others

1、`torch.backends.cudnn.benchmark = True`
大部分情况下，设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题

2、`sum(p.numel() for p in model.parameters() if p.requires_grad)` 计算网络的参数量



## Copy

##### Confusion Matrix

```python
class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    model = get_net().cuda()
    test_files = get_files(config.test_data)
    test_dataloader = DataLoader(ChaojieDataset(
        test_files), batch_size=1, shuffle=False, pin_memory=False)
    best_model = torch.load("checkpoints/best_model/%s/0/model_best.pth.tar" % config.model_name)
    model.load_state_dict(best_model["state_dict"])
    labels = [1, 2, 3]
    confusion = ConfusionMatrix(num_classes=len(labels), labels=labels)
    model.eval()
    with torch.no_grad():
        for val_data in test_dataloader:
            val_images, val_labels = val_data
            outputs = model(val_images.cuda())
            outputs = torch.softmax(outputs, dim=1) # outputs[batch,channel,height,width]
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(),
                             val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()
```

