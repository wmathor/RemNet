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

RemNet is an open source deep learning framework based on C++. It is very easy to use, you only need to define the network structure and set the relevant parameters can start training

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

After cloned locally, import the project using visual studio and run it. Very simple. You can also try to modify the information in `myModel.json`, such as adding additional layers, and then rerun.

By the way, you don't need to prepare any data at first, I've provided it for you, right here in`./remnet/mnist_data`

## More about RemNet

- [x] Supports the most commonly used layer types by far: Conv、Pool、FC、ReLU、Tanh、Dropout、BN、Scale
- [x] Support for the two most commonly used loss layers: CrossEntropy、SVM
- [x] Supports multiple optimizers: SGD、Momentum、RMSProp
- [x] Two kinds of weight initialization are supported: Gaussian、MSRA
- [x] Support fine-tune operation

RemNet is written in a similar way to Caffee in that its basic data types include Cube and Blob. In RemNet, the relationship between them is shown below

![](https://s2.ax1x.com/2020/02/16/3pfBng.png)

Here is a simple network diagram to help you understand the Net, Layer, and Blob relationships in the source code

![](https://s2.ax1x.com/2020/02/16/39V9dU.png)



### Design focus

Taking the MNIST dataset as an example, the following figure shows how its Images and Labels are stored in RemNet

![](https://s2.ax1x.com/2020/02/19/3VPNpF.png)

Obviously, I did one-hot Encoding for the tags, which is convenient for later Loss calculations, and they're all unified into blobs, which is better for understanding than normal data types, because that's what most deep learning problems do.

Even the MNIST dataset has 60,000 samples, so it can't be entered all at once, so RemNet also allows developers to set the batch size themselves. Suppose I define an ultra-small convolutional neural network (conv-> relu-> pool-> fc-> Softmax/SVM), the process of forward propagation and back propagation is as follows. You'll notice that there are some layers that only have x in them and no w or b in them, but for programming convenience, I've declared them all, just not using them. Again, the same is true for back propagation, some layers have no w and b gradient information at all

![](https://s2.ax1x.com/2020/02/19/3EXyYd.png)

Traditional CNN operations are fine with Blob data, but what about FC Layer? When I use PyToch, I Flatten the input data almost every time before the FC Layer. RemNet did not use Flatten to solve this problem because it would have wasted more time. RemNet takes a similar approach to convolution, as shown below

![](https://s2.ax1x.com/2020/02/19/3ExdO0.png)

Let's look at the left-hand side first, and if you're not familiar with convolution and full concatenation, think back. On the left is the solution of RemNet. The multiplication of values at each position in the Channel is the sum, which is the operation of convolution. For RemNet, it is also the forward propagation process of FC Layer. If each Cube is Flatten, it becomes the right-hand form.

### FAQ

1. What is the biggest difference between RemNet and Tensorflow and PyTorch?

   I've barely used Tensorflow because it's so hard to learn. PyTorch is the deep learning framework that I've been using, so let me tell you the difference between PyTorch and RemNet. PyTorch needs to customize the optimizer, then pass in parameters, gradient zeroing and back propagation during iteration. RemNet doesn't have to go through all the trouble. You just need to change the parameters of `myModel.json`, such as what optimizer to use, what weight attenuation to use, and leave the rest to the program

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