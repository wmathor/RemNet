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

RemNet is an open source deep learning framework written in C++. It is very easy to use, you just need to define the network structure and set the relevant parameters to start training!

- [Download](https://github.com/wmathor/RemNet#download)
- [More about RemNet](https://github.com/wmathor/RemNet#more-about-remnet)
  - [Design focus](https://github.com/wmathor/RemNet#design-focus)
  - [FAQ](https://github.com/wmathor/RemNet#faq)
- [TODO](https://github.com/wmathor/RemNet#-todo)
- [LICENSE](https://github.com/wmathor/RemNet#license)

## Download

```shell
$ git clone https://github.com/wmathor/RemNet.git
```

Once the clone is local, please use Visual Studio to import the project and run it, it's very easy. You can also try changing the information in myModel.json, such as adding some other layers, and then run it again.

By the way, you don't need to prepare any data at the beginning, I've already provided it for you, it's in `./RemNet/mnist_data`.

## More about RemNet

- [x] Supports the most commonly used layer types by far: Conv、Pool、FC、ReLU、Tanh、Dropout、BN、Scale
- [x] Support for the two most commonly used loss layers: CrossEntropy、SVM
- [x] Supports multiple optimizers: SGD、Momentum、RMSProp
- [x] Two kinds of weight initialization are supported: Gaussian、MSRA
- [x] Support fine-tune operation

RemNet is written in a similar way to Caffee, and its basic data types include Cube and Blob, which are related to each other in RemNet as follows

![](https://s2.ax1x.com/2020/02/16/3pfBng.png)

Here is a simple network diagram to help you understand the Net, Layer, and Blob relationships in the source code

![](https://s2.ax1x.com/2020/02/16/39V9dU.png)



### Design focus

Taking the MNIST dataset as an example, the following figure shows how its Images and Labels are stored in RemNet

![](https://s2.ax1x.com/2020/02/19/3VPNpF.png)

Obviously, I've done one-hot Encoding for the tags, which makes it easier to calculate Loss later, and it's all in Blob format, which is more useful for understanding than a normal data type, because that's how most deep learning problems are done.

Even the MNIST dataset has 60,000 samples, so it is not possible to enter them all at once, so RemNet also supports developers to set their own batch size. suppose I define an ultra-small convolutional neural network (Conv->ReLU->Pool->FC->Softmax/SVM), with forward and backward propagation as follows . You'll notice that some layers only have x in them, not w and b, but I've declared them all for programming convenience, I just don't use them. The same goes for backpropagation, some layers have no w and b gradient information at all!

![](https://s2.ax1x.com/2020/02/19/3EXyYd.png)

Traditional CNN operations have no problem using a data representation like a blob, but what about when it comes to the FC Layer? When I use PyToch, I almost always flatten the input data before the FC Layer, but RemNet solves this problem by not flattening it because it wastes more time.

![](https://s2.ax1x.com/2020/02/19/3ExdO0.png)

Let's look at the left side first, if you are not familiar with convolution and full connection, please think back a bit. On the left side is the solution of RemNet, it corresponds to multiplying the values in each position of the channel and summing them. If you flatten each Cube, it becomes the right side.

### FAQ

1. What is the biggest difference between RemNet and Tensorflow and PyTorch?

   Tensorflow I almost never use, because it's too hard to learn, PyTorch is the deep learning framework I always use, I will talk about the difference between RemNet and PyTorch, PyTorch need custom optimizer, then pass parameters, in the iterative process also need gradient zeroing and backpropagation, but RemNet doesn't need so much trouble, you just need to modify myModel.json parameters, for example, what optimizer to use, how much the weight decay is. You just need to modify the parameters of `myModel.json`, such as what optimizer to use and how much the optimizer's weight decay is, and leave the rest to the program.


2. Why is this project called RemNet?

   Because I like a girl called レム, her name on the English translation is Rem:heart:

3. What can I do?

   You can `fork` the project and add new features to it. If someone helps maintain the project, I'll add a list of "contributors" and expect you to be the first

4. Can I take it to a competition or business?

   If you are a contributor, I can consider it. Otherwise, you can't. For details, please refer to "license".

## 🎨 TODO

- [ ] Implement support for L2 regularization
- [ ] Implement a common image data interface
- [ ] Optimized code, encapsulated as executable
- [ ] Support for RNN
- [ ] Support GPU training (lifetime series)
- [ ] Design graphical interface（lifetime series × 2）

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
