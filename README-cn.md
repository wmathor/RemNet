<script type="text/javascript" src="[http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default">](http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default%22%3E%3C/script%3E) 

<p align = "center">
  <a href = "https://github.com/wmathor/RemNet">
    <img height="60%" width = "70%" src = "https://s2.ax1x.com/2020/02/16/3prR1J.png">
  </a>
</p>
<p align = "center">
  <a href = "https://github.com/wmathor/RemNet">
    <img src="https://img.shields.io/badge/language-C++-brightgreen.svg">
  </a>
  <a href = "https://github.com/wmathor/RemNet">
    <img src = "https://img.shields.io/badge/Compiler-Visual Studio 2019-blue.svg">
  </a>
  <a href = "https://wmathor.com/" target = "_blank">
    <img src = "https://img.shields.io/badge/Blog-wmathor-orange.svg">
  </a>
</p>

[中文](https://github.com/wmathor/RemNet/blob/master/README-cn.md) | [English](https://github.com/wmathor/RemNet)

RemNet是基于C++编写的开源深度学习框架。它非常容易上手使用，只需要您定义好网络结构并设定相关参数即可开始训练

- [下载](https://github.com/wmathor/RemNet/blob/master/README-cn.md#%E4%B8%8B%E8%BD%BD)
- [有关RemNet的更多信息](https://github.com/wmathor/RemNet/blob/master/README-cn.md#%E6%9C%89%E5%85%B3remnet%E7%9A%84%E6%9B%B4%E5%A4%9A%E4%BF%A1%E6%81%AF)
  - [设计重点](https://github.com/wmathor/RemNet/blob/master/README-cn.md#%E8%AE%BE%E8%AE%A1%E9%87%8D%E7%82%B9)
  - [FAQ](https://github.com/wmathor/RemNet/blob/master/README-cn.md#faq)
- [TODO](https://github.com/wmathor/RemNet/blob/master/README-cn.md#art-todo)
- [贡献者](https://github.com/wmathor/RemNet/blob/master/README-cn.md#%E8%B4%A1%E7%8C%AE%E8%80%85)
- [许可](https://github.com/wmathor/RemNet/blob/master/README-cn.md#%E8%AE%B8%E5%8F%AF)

## 下载

```bash
$ git clone https://github.com/wmathor/RemNet.git
```

clone到本地之后，请使用Visual Studio导入该项目，然后运行即可，非常简单。您也可以尝试修改`myModel.json`中的信息，例如添加其它一些层，然后再重新运行。

对了，一开始您不需要准备什么数据，我已经为您提供好了，就在`./RemNet/mnist_data`中



## 有关RemNet的更多信息

- [x] 支持目前最常用的层类型：Conv、Pool、FC、ReLU、Tanh、Dropout、BN、Scale
- [x] 支持两种最常用的损失层：CrossEntropy、SVM
- [x] 支持多种优化器：SGD、Momentum、RMSProp
- [x] 支持两种权重初始化：Gaussian、MSRA
- [x] 支持Fine-tune操作

RemNet的整体编写思路类似于Caffee，它的基本数据类型包括Cube和Blob。在RemNet中，它们之间的关系如下图

![](https://s2.ax1x.com/2020/02/16/3pfBng.png)

下面是一个简单的网络结构示意图，帮助您理解RemNet中的Net、Layer、Blob的关系

![](https://s2.ax1x.com/2020/02/16/39V9dU.png)

### 设计重点

以MNIST数据集为例，下图展示了它的Images和Labels在RemNet中的存储方式

![](https://s2.ax1x.com/2020/02/19/3EOVv6.png)

很明显，对于标签我进行了one-hot Encoding，这方便后面的Loss计算，而且都统一成Blob的格式相比于普通的数据类型更有助于理解，因为绝大多数深度学习问题都是这么做的。

即便是MNIST数据集也有60000个样本，因此无法一次性输入，所以RemNet也支持开发者自己设置batch size。假设我定义了一个超小型的卷积神经网络（Conv->ReLU->Pool->FC->Softmax/SVM），前向传播和反向传播的过程如下图。你会注意到，有些层里面只有$x$，没有$w$和$b$，但为了编程方便，我都声明了，只不过没有使用而已。同样，反向传播的时候也是一样，有的层根本没有$w$和$b$的梯度信息

![](https://s2.ax1x.com/2020/02/19/3EXyYd.png)

传统的CNN操作使用Blob这种数据表示没有什么问题，但如果到了FC Layer呢？我在使用PyToch的时候，几乎在每次在FC Layer之前都会将输入数据进行一次Flatten。RemNet在解决这个问题的时候并没有采用Flatten，因为这会浪费更多时间。RemNet采用的方法类似于"卷积"，具体看下图

![](https://s2.ax1x.com/2020/02/19/3ExdO0.png)

先看左边，如果您对卷积和全连接不太熟悉，请先回想一下。左边是RemNet的解决方法，对应Channel中每个位置上的值相乘在做和，这是卷积的操作，对于RemNet来说同样也是FC Layer的前向传播过程。如果把每个Cube都Flatten，就变成了右边的形式

### FAQ

1. 与Tensorflow和PyTorch最大的区别是什么？

   Tensorflow我几乎没用过，因为太难学了。PyTorch是我一直使用的深度学习框架。我就说一下RemNet和PyTorch的区别，PyTorch需要自定义优化器，然后传入参数，在迭代过程中还需要梯度清零以及反向传播。但是RemNet不需要这么麻烦，你只需要修改`myModel.json`的参数，使用什么优化器，优化器的权重衰减是多少，剩下的全部交给程序就行了

2. 为什么起名为RemNet？

   因为我喜欢的一个女生叫レム ，她名字的英文翻译就是Rem:heart:

3. 我可以做什么？

   你可以`fork`这个项目，然后为它添加新的功能。如果有人帮忙维护这个项目，我会添加一个"贡献者"名单，期待您成为第一位

4. 是否可以商用？

   如果您是"贡献者"的一员，我可以考虑。否则不可以，具体的内容请看"许可"

## :art: TODO

- [ ] 实现L2正则化
- [ ] 实现通用的图片数据接口
- [ ] 优化代码，封装为可执行文件
- [ ] 支持循环神经网络
- [ ] 支持GPU训练（有生之年系列）
- [ ] 设计图形界面（有生之年系列×2）

## 贡献者

|                            :tada:                            |
| :----------------------------------------------------------: |
| <img height='48' width='48' src='https://avatars1.githubusercontent.com/u/32392878?s=460&v=4'> |
|            [@wmathor](https://github.com/wmathor)            |

## 许可

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br/>本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。