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

[中文](https://github.com/mathors/2019-nCoV/blob/master/README-cn.md) | [English](https://github.com/mathors/2019-nCoV)

RemNet是基于C++编写的开源深度学习框架。它非常容易上手使用，只需要您定义好网络结构并设定相关参数即可开始训练。

RemNet的整体编写思路类似于Caffee，它的基本数据类型包括Cube和Blob。在RemNet中，它们之间的关系如下图

![](https://s2.ax1x.com/2020/02/16/3pfBng.png)

下面是一个简单的网络结构示意图，帮助您理解源代码中的Net、Layer、Blob的关系

![](https://s2.ax1x.com/2020/02/16/39V9dU.png)