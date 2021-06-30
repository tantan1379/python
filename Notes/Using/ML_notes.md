# Machine Learning

## Basic

### Step

**深度学习的步骤：**第一步是设定目标和评估指标，第二步是不断优化系统，瞄准和设计目标



### metric

准确率 accuracy(ACC) = (TP+TN)/(TP+TN+FP+FN)

召回率（查全率、灵敏度）recall(TPR) = TP/(TP+FN)		   所有“猫”样本，被分类正确的概率

精确率（查准率）precision(PPV) = TP/(TP+FP)					分类为“猫”的样本，真正是“猫”的概率

特异度 Specificity(TNR) = TN/(TN+FP)

F1 score 是对查准率和查全率的调和平均 F1 = 2(precision*recall)/(precision+recall)



### Overfitting

现象：在训练数据上的误差非常小，而在测试数据上误差反而增大。

解决过拟合的方法：
1、增加数据量
2、选用shallow的网络模型降低模型复杂度
3、正则化(regularization)：对模型参数添加先验，使得模型复杂度较小，对于噪声以及outliers的输入扰动相对较小
4、加dropout
5、对数据做数据增强(data augumentation)
6、early stopping (不推荐)



### Learning rate

一般来说，越大的`batch-size`使用越大的学习率，越大的`batch-size`意味着我们学习的时候，收敛方向的`confidence`越大，我们前进的方向更加坚定，而小的`batch-size`则显得比较杂乱，毫无规律性，因为相比批次大的时候，批次小的情况下无法照顾到更多的情况，所以需要小的学习率来保证不至于出错。

**动量(momentum)：**将上一次的梯度变化考虑在内，提高收敛的速度



### Weight init

权重初始化相比于其他的trick来说在平常使用并不是很频繁。因为大部分人使用的模型都是预训练模型，使用的权重都是在大型数据集上训练好的模型，当然不需要自己去初始化权重了。只有没有预训练模型的领域会自己初始化权重，或者在模型中去初始化神经网络最后那几个全连接层的权重。常用的权重初始化算法是 **「kaiming_normal」** 或者 **「xavier_normal」**。

不初始化可能会减慢收敛速度，影响收敛效果。



### Model ensemble

Ensemble是论文刷结果的终极核武器,深度学习中一般有以下几种方式

- 同样的参数,不同的初始化方式
- 不同的参数,通过cross-validation,选取最好的几组
- 同样的参数,模型训练的不同阶段，即不同迭代次数的模型。
- 不同的模型,进行线性融合. 例如RNN和传统模型.

提高模型性能和鲁棒性大法：probs融合 和 投票法。





## Trick

### divide dataset

传统方法采用交叉验证或622划分。如果数据量大于10000，则可以考虑改变622的占比，一般我们使得训练集尽可能的大。



### Softmax and Sigmoid

两者最大的区别在于softmax的计算的是一个比重，而sigmoid只是对每一个输出值进行非线性化

但是当输出层为一个神经元时，此时会使用sigmoid代替softmax，因为此时还按照softmax公式的话计算值为1。

softmax一般用于多分类的结果，一般和one-hot的真实标签值配合使用，大多数用于网络的最后一层；而sigmoid是原本一种隐层之间的激活函数，但是因为效果比其他激活函数差，目前一般也只会出现在二分类的输出层中，与0 1真实标签配合使用。
