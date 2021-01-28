# 基础

## 填坑:

1、可变类型：列表list、字典dict、集合set
不可变类型：元组tuple、字符串str、数据类型int&float (创建后只能查询和访问）

2、对象中存放的三个数据：id(对象地址）、type、value

3、python允许不加括号的赋值以及输出，则此时默认为元组

5、在为变量重新赋值时，会改变变量的id；   
但如果通过(可变)变量修改对象的内容，不会改变id：见 {/Basic/variable/variable_object}

6、列表的存储性能好，但查找数据的性能差；字典的查找效率高

8、字典的键是唯一的，且只能为不变的值，如字符串、数字、元组

9、函数的参数传递有两种，一种是顺序传递（默认），另一种为关键词传递（b=1,a=2)

10、在函数内用global来声明变量，则在调用该函数后，全局变量可以实现在函数内修改

11、在形参前加 * 表示可以接受多个实参值存进数组,对于在形参前加 ** 表示表示接受参数转化为字典类型    

12、python面向对象的三大特性为：
（1）封装：对数据属性进行严格控制，隔离复杂度；（2）继承：解决代码的复用性问题；（3）多态：增加程序的灵活性与可扩展性

13、静态方法允许在一个类中定义一个与类无关的方法（可以没有任何参数，其余方法都至少需要一个参数self作为对象自身传入）

14、若a和b是列表/字典/集合，a=b，则完成的是引用，改变a或b都将改变彼此;如果要实现赋值，可以如下三种方法：
（1）a=b[:]  （2）a=list(b)  NOTE:b不一定是列表，可以是任意类型  （3）a=b*1

15、len(a)实际上得到的是a第一个维度的长度

16、sort对现有列表进行排序，sorted可以对list和iterable进行排序

17、os.sep可以提高复用性，根据系统生成分隔符，在windows里会生成\，在linux中生成/

18、三目运算符：`a if a>b else b`如果满足a>b则返回a，否则返回b   

---
---

#  numpy包(以下简称np)

## ndarray的属性
`ndarray.ndim` 用于返回数组的维数，等于秩

`ndarray.shape` 表示数组的维度，返回一个元组（也可以用于调整数组大小，不常用）

`ndarray.flags` 返回 ndarray 对象的内存信息

`ndarray.itemsize` 以字节的形式返回数组中每一个元素的大小

## ndarray与matrix异同点
表现形式与用法上几乎一样。
但对matrix而言*是矩阵乘，multiply是元素乘，而对ndarray而言，dot才是矩阵乘，*是元素乘

## 具体操作
#### 创建一个指定维数的全0数组,全1数,单位矩阵
`my_mat20=np.zeros(dim，dtype=<class 'float'>)` # 注意dim为一个表示维度的tuple

`my_mat21=np.ones(dim, dtype=<class 'float'>)` # 注意dim为一个表示维度的tuple

`my_mat23=np.eye(N,M=None,k=0,dtype=<class 'float'>,order='C)` # N为行数，M为列数

#### 创建空数组与初始数组
`np.empty(dim,dtype)` # dim应有一维输入为0(?)

`np.full(dim, val, dtype)` # zeros和ones的通用式

#### 创建一个随机分布且指定维数的矩阵
`my_mat41=np.random.rand(d0,d1,…dn)` # 0-1均匀分布

`my_mat41=np.random.randn(d0,d1,…dn)`# 正态分布
`my_mat42=np.random.randint(low,high=None,size=None,dtype)` # 整数

#### 自动生成narray（arrage为间距控制，linspace为元素数控制）
`np.arange(start,end,step,dtype)` # 不包含end

`np.linspace(start, end, element_nums, endpoint=True,dtype)` #包含end,endpoint决定是否包含end，dtype决定数组元素类型

`np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)`

* example:
1. `a = np.arange(1, 5, 1)` #[1 2 3 4]

2. `b = np.linspace(1, 5, 5, endpoint=True, dtype=np.int32)` #[1 2 3 4 5]

#### 返回一份展开的数组拷贝，对拷贝所做的修改不会影响原始数组
`arr7=arr.flatten(order='C')` # 默认按行展开

#### 从给定数组的形状中删除一维的条目
`arr8=np.squeeze(arr, axis)` # axis：用于选择形状中一维条目的子集

#### 添加数据元素
`arr92=arr.append(values,axis=None)`

#### 返回列表的对应值
 `ndarray.sum(axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True)`

#### 以二维数组为例，axis不写则返回整个数组的最大值，axis=0取每列的最大值，axis=1取每行的最大值
`ndarray.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)`

#### 分割与结合
`numpy.split(ary, indices_or_sections, axis)` # numpy.split 函数沿特定的轴将数组分割为子数组

---
---
# 文件操作
#### 将URL表示的网络对象复制到本地文件
`urllib.request.urlretrieve(url, filename=None, reporthook=None, data=None)`

url：外部或者本地url
filename：指定了保存到本地的路径（如果未指定该参数，urllib会生成一个临时文件来保存数据）；
reporthook：是一个回调函数，当连接上服务器、以及相应的数据块传输完毕的时候会触发该回调。我们可以利用这个回调函数来显示当前的下载进度。
data：指post到服务器的数据。该方法返回一个包含两个元素的元组(filename, headers)，filename表示保存到本地的路径，header表示服务器的响应头。

## 读写

#### 编码方式：
1. 利用方法自带的输入属性，encoding='utf-8'
2. 'xxx'.encode('utf-8')

#### 文件模式：
1. w、wb为覆盖写，文件指针位于文件首部；a、ab用于追加数据，文件指针位于文件结尾
2. r为普通场景的读入；rb的模式适合图片、视频、音频这样的文件读取
3. w+和r+为读写，但w+文件可以不存在，r+文件必须存在

#### 读:
1. read或readline或readlines，文件指针位置是继承的
2. `f.readline()` 读一行（括号内选择一行读多少字符），返回一个字符串
3. `f.readlines()` 按行读取（括号内选择最大读多少行），一次性读取（剩余）所有内容，并返回一个每个元素均为行内容的列表

#### 其他：
1. 使用`with open as x` 可以省略 close
2. `f.truncate()`截取字符（括号内指定截取大小），模式需要+
3. `f.tell()`得到文件指针当前位置
4. `f.seek(offset,from)` 用于定位指针位置（offset为偏移量，from可选012，0为文件头，1为当前位置，2为文件结尾

## os模块：

#### 重命名文件
 `os.rename(src,dst)` src为源文件（原名），dst为目标文件（新名）

#### 删除文件
`os.remove(path)`

#### 创建/删除文件夹
`os.mkdir(path)` # 只能创建一级目录

`os.mkdirs(path)` # 可以创建多级目录

`os.rmdir(path)` # 文件必须存在，只能删除空目录

`shutil.retree(path)` # （非空）删除整个目录（先导入）

#### 获取路径内容
`os.listdir(path)` # 返回指定路径下的文件和文件夹列表

`os.scandir(path)` # 新版本用法，返回一个迭代器对象

`os.getcwd()` # 获取当前目录

`os.path.join(path1,path2)` # 获取拼接的目录

`os.path.split(path)` # 将目录分割为目录和文件名以二元组返回



---
---

# 深入

#### 比较直接赋值，浅拷贝和深拷贝：
直接赋值：其实就是对象的引用（别名）。
浅拷贝(copy)：拷贝父对象，不会拷贝对象的内部的子对象。
深拷贝(deepcopy)： copy 模块的 deepcopy 方法，完全拷贝了父对象及其子对象。

#### 类的继承
如果只在class的()中加父类的名字，没有super只能调用父类无二次操作(self.xx)的方法
super().__init__()可以执行父类的构造函数，使得我们能够调用父类的属性。

#### axis=0或1的理解
使用0值表示沿着每一列或行标签索引值向下执行方法（drop时沿着列的方向选一行）
使用1值表示沿着每一行或者列标签模向执行对应的方法

#### 抽象类只能被继承，不能实例化

#### python中的对象传递：
1、python对所有的参数都采用对象传递。对象传递是值传递和引用传递的一种综合。如果函数收到的是一个**可变对象**（字典、列表）的引用，就能修改对象的原始值，相当于**引用传递**来传递对象。如果函数收到的是一个**不可变对象**（数字、字符或元组）的引用，就不能直接修改原始对象--相当于通过**值传递**来传递对象。   
```
# 经典错误：选择可变对象为参数的默认值
# 此时对象传递会按照引用传递对a_list进行操作，每次append都是对同一对象a_list进行操作
def bad_append(new_item, a_list=[]):
    a_list.append(new_item)
    return a_list

def good_append(new_item,a_list=None):
    if a_list is None:
        a_list = []
    a_list.append(new_item)
    return a_list

print(bad_append(1))  # output:[1]
print(bad_append(1))  # output:[1,1]
print(good_append(1))  # output:[1]
print(good_append(1))  # output:[1]
```
2、调用`id()`方法得到的其实是**变量指向对象所在的内存空间的地址**。变量同时表示着对象在内存中的存储和地址。   
例1：
```
# 每当出现一个对象（等式右边）时，就为它创建空间
a="xyz"  # 创建a变量和字符串"xyz"，将a指向字符串"xyz"这一对象（将"xyz"的内存地址赋值给a)，此后a中存放了"xyz"的地址
b=a  # 将b也指向a指向的内存空间->字符串"xyz"，这样b中也存放着"xyz"的地址
print(id(a))  #output:1629550825392
print(id(b))  #output:1629550825392
a="qwe"  # 创建字符串"qwe"，将a重新指向字符串"qwe"这一对象，改而存放"qwe"的地址。
print(id(a))  #output:1629551598000
print(id(b))  #output:1629550825392
```
例2：
```
# arg在执行append之前，实参变量b把自己引用对象的地址也给了arg
# 所以对arg进行操作的时候对引用对象b也进行了修改
def bar(args):
    print(id(args))  # output:4324106952(即[]的地址)
    args.append(1)

b = []
print(b)  #output:[]
print(id(b))  #output:4324106952
bar(b)
print(b)  #output:[1]
print(id(b))  #output:4324106952
```
---
---
