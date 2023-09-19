# 1、TensorFlow准备工作

## 1.1、TensorFlow安装

请在带有numpy等库的虚拟环境中安装

### 1.1.1、CPU版本

```python
pip install tensorflow==1.8 -i https://mirrors.aliyun.com/pypi/simple
```

### 1.1.2、GPU版本

# 2、TensorFlow框架介绍

TensorFlow 程序通常被组织成一个构建图阶段和一个执行图阶段。

在**构建阶段**，数据与操作的执行步骤被描述成一个图。

在**执行阶段**，使用会话执行构建好的图中的操作。

- 图和会话：
  - 图：这是TensorFlow将计算表示为指令之间的依赖关系的一种表示法
  - 会话：TensorFlow跨一个或多个本地或远程设备运行数据流图的机制(只针对1.x版本，由于2.x版本为动态图，因此不需要会话)
- 张量：TensorFlow中的基本数据对象
- 节点：提供图当中执行的操作

## 2.1、TF数据流图

### 2.1.1、案例：TensorFlow实现一个加法运算

```python
import tensorflow as tf
import os

# 修改日志等级，去掉tf的警告日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 2.x版本没有session
def tensorflow_demo():
    # TensorFlow的基本结构
    # 原生python加法运算
    a = 2
    b = 3
    c = a + b
    print("普通加法运算的结果: {}\n".format(c))

    # TensorFlow实现加法运算
    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print("TensorFlow加法运算的结果：{}\n".format(c_t))

    # 用于TensorFlow 1.x版本迁移到2.x版本
    tf.compat.v1.disable_eager_execution()

    # 开启会话
    with tf.compat.v1.Session() as sess:
        c_t_value = sess.run(c_t)
        print("c_t_value is: {}\n".format(c_t_value))

    return None


if __name__ == "__main__":
    # 代码1：TensorFlow的基本结构
    tensorflow_demo()

```

结果展示：

![image-20230917142857489](https://raw.githubusercontent.com/1793925850/user-image/master/imgs/202309171428525.png)

### 2.1.2、数据流图介绍

![image-20230917140635591](https://raw.githubusercontent.com/1793925850/user-image/master/imgs/202309171406685.png)

## 2.2、图与TensorBoard

### 2.2.1、图结构

图包含了一组tf.Operation代表的计算单元对象和tf.Tensor代表的计算单元之间流动的数据。

即：数据(Tensor) + 操作(Operation)

### 2.2.2、图相关操作

#### 2.2.2.1、默认图

通常TensorFlow会默认帮我们创建一张图

查看方法：

1. 调用方法

   用tf.get_default_graph()

2. 查看属性

   op、sess都含有graph属性

   .graph

```python
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def graph_demo():
    tf.compat.v1.disable_eager_execution()
    # 图的演示
    # TensorFlow实现加法运算
    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print("TensorFlow加法运算的结果：{}".format(c_t))

    # 查看默认图
    # 方法1
    default_g = tf.compat.v1.get_default_graph()
    print("default_g is: {}".format(default_g))

    # 方法2
    tf.compat.v1.disable_eager_execution()
    print("a_t 的图属性: {}".format(a_t.graph))
    print("b_t 的图属性: {}".format(b_t.graph))
    print("c_t 的图属性: {}".format(c_t.graph))

    return None


if __name__ == "__main__":
    # 代码2：图的演示
    graph_demo()
```

结果展示：

![image-20230917142831876](https://raw.githubusercontent.com/1793925850/user-image/master/imgs/202309171428921.png)

#### 2.2.2.2、创建图

- 可以通过tf.Graph()自定义创建图
- 如果要在这张图中创建OP，典型用法是使用tf.Graph.as_default()上下文管理器

```python
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def tensorflow_demo():
    # 自定义图
    new_g = tf.Graph()
    # 在自己的图中定义数据和操作
    with new_g.as_default():
        a_new = tf.constant(20)
        b_new = tf.constant(30)
        c_new = a_new + b_new
        print("c_new is: {}".format(c_new))
        with tf.compat.v1.Session() as sess:
            c_new_value = sess.run(c_new)
            print("c_new_value is: {}".format(c_new_value))

    return None


if __name__ == "__main__":
    # 代码1：TensorFlow的基本结构
    tensorflow_demo()

```

结果展示：

![image-20230917143916385](https://raw.githubusercontent.com/1793925850/user-image/master/imgs/202309171439423.png)



### 2.2.3、TensorBoard：可视化学习

TensorFlow提供了TensorBoard可视化工具

**实现程序可视化过程：**

1. 数据序列化-events文件
2. 启动TensorBoard

#### 2.2.3.1、数据序列化-events文件

TensorBoard通过读取TensorFlow的事件文件来运行，需要将数据生成一个序列化的Summary protobuf对象。

```
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def tensorflow_demo():
    # 原生python加法运算
    a = 2
    b = 3
    c = a + b
    print("普通加法运算的结果: {}".format(c))

    # TensorFlow实现加法运算
    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print("TensorFlow加法运算的结果：{}".format(c_t))

    # 自定义图
    new_g = tf.Graph()
    # 在自己的图中定义数据和操作
    with new_g.as_default():
        a_new = tf.constant(20)
        b_new = tf.constant(30)
        c_new = a_new + b_new
        print("c_new is: {}".format(c_new))
        with tf.compat.v1.Session() as sess:
            c_new_value = sess.run(c_new)
            print("c_new_value is: {}".format(c_new_value))

            # 可视化
            # 1、将图写入本地生成events文件
            tf.compat.v1.summary.FileWriter("./tmp/summary", graph=new_g)

    return None


if __name__ == "__main__":
    # 代码1：TensorFlow的基本结构
    tensorflow_demo()

```

结果展示：会生成相关文件：![image-20230917175603145](https://raw.githubusercontent.com/1793925850/user-image/master/imgs/202309171756186.png)

#### 2.2.3.2、启动TensorBoard

输入命令：

```shell
tensorboard --logdir="./tmp/summary" 
```

![image-20230917175747977](https://raw.githubusercontent.com/1793925850/user-image/master/imgs/202309171757009.png)

最后输入网址即可

### 2.2.4、OP

数据：Tensor对象

操作：Operation对象——OP

#### 2.2.4.1、常见OP

|      类型      |                         实例                         |
| :------------: | :--------------------------------------------------: |
|    标量运算    |  add, sub, mul, div, exp, log, greater, less, equal  |
|    向量运算    | concat, slice, splot, constant, rank, shape, shuffle |
|    矩阵运算    |       matmul, matrixinverse, matrixdateminant        |
|  带状态的运算  |             variable, assign, assignadd              |
|  神经网络组件  |    softmax, sigmoid, relu, convolution, max_pool     |
|   存储、恢复   |                    save, restore                     |
| 队列及同步运算 |     enqueue, dequeue, mutexacquire, mutexrelease     |
|     控制流     |      merge, switch, enter, leave, nextiteration      |

#### 2.2.4.2、指令名称

每个OP指令都对应一个唯一的名称

一张图对应一个命名空间

**修改指令名称：**

```python
a = tf.constant(3.0, name="a")
b = tf.constant(4.0, name="b")
```

## 2.3、会话(2.x版本没有)

两种开启方式：

- tf.Session：用于完整的程序当中
- tf.InteractiveSession：用于交互式上下文中的TensorFlow，例如shell

## 2.4、张量

### 2.4.1、张量的属性

- type：数据类型

  ![image-20230919212619189](https://raw.githubusercontent.com/1793925850/user-image/master/imgs/202309192126240.png)

- shape：形状(阶)

### 2.4.2、创建张量的指令

- 固定值张量

  ```python
  # 创建所有元素为零的张量
  tf.zeros(shape, dtype=tf.float32, name=None)
  # 创建所有元素为1的张量
  tf.ones(shape, dtype=tf.float32, name=None)
  # 创建一个常数张量
  tf.constant(value, dtype=None, shape=None, name='Const')
  ```

  

- 随机值张量

  ```python
  # 从正态分布中输出随机值，由随机正态分布的数字组成的矩阵
  tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
  ```

















