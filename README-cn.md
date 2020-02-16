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

## FAQ

1. 为什么起名为RemNet？

   因为我喜欢的一个女生叫レム ，她名字的英文翻译就是Rem:heart:

2. 我可以做什么？

   你可以`fork`这个项目，然后为它添加新的功能。如果有人帮忙维护这个项目，我会添加一个"贡献者"名单，期待您成为第一位

3. 是否可以拿去参加比赛或是商用？

   如果您是"贡献者"的一员，我可以考虑。否则不可以，具体的内容请看"许可证"

## :tada:TODO

- [ ] 实现Fine-tune功能
- [ ] 实现Dropout层
- [ ] 实现更多的优化器
- [ ] 实现对L2正则化的支持
- [ ] 实现通用的图片数据接口

## 许可

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br/>本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。