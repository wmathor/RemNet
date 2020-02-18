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

## Download

```shell
$ git clone https://github.com/wmathor/RemNet.git
```

After cloned locally, import the project using visual studio and run it. Very simple. You can also try to modify the information in `myModel.json`, such as adding additional layers, and then rerun.

By the way, you don't need to prepare any data at first, I've provided it for you, right here in`./remnet/mnist_data`

## More about RemNet

RemNet is written in a similar way to Caffee in that its basic data types include Cube and Blob. In RemNet, the relationship between them is shown below

![](https://s2.ax1x.com/2020/02/16/3pfBng.png)

Here is a simple network diagram to help you understand the Net, Layer, and Blob relationships in the source code

![](https://s2.ax1x.com/2020/02/16/39V9dU.png)



## FAQ

1. Why is this project called RemNet?

   Because I like a girl called レ ム, her name on the English translation is Rem:heart:

2. What can I do?

   You can `fork` the project and add new features to it. If someone helps maintain the project, I'll add a list of "contributors" and expect you to be the first

3. Can I take it to a competition or business?

   If you are a contributor, I can consider it. Otherwise, you can't. For details, please refer to "license".

## :tada:TODO

- [x] Achieve the fine-tune function
- [ ] Realize the Dropout layer
- [x] Implement more optimizers
- [ ] Implement support for L2 regularization
- [ ] Implement a common image data interface

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.