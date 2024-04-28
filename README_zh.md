<a href="https://github.com/WenjieDu/PyPOTS">
    <img src="https://pypots.com/figs/pypots_logos/PyPOTS/logo_FFBG.svg" width="200" align="right">
</a>

<h3 align="center">欢迎来到 PyPOTS</h3>

<p align="center"><i>一个用于部分观测时间序列的 Python 机器学习工具箱</i></p>

<p align="center">
    <a href="https://docs.pypots.com/en/latest/install.html#reasons-of-version-limitations-on-dependencies">
       <img alt="Python version" src="https://img.shields.io/badge/Python-v3.7+-E97040?logo=python&logoColor=white">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS">
        <img alt="powered by Pytorch" src="https://img.shields.io/badge/PyTorch-❤️-F8C6B5?logo=pytorch&logoColor=white">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/releases">
        <img alt="the latest release version" src="https://img.shields.io/github/v/release/wenjiedu/pypots?color=EE781F&include_prereleases&label=Release&logo=github&logoColor=white">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/blob/main/LICENSE">
        <img alt="BSD-3 license" src="https://img.shields.io/badge/License-BSD--3-E9BB41?logo=opensourceinitiative&logoColor=white">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS#-community">
        <img alt="Community" src="https://img.shields.io/badge/join_us-community!-C8A062">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/graphs/contributors">
        <img alt="GitHub contributors" src="https://img.shields.io/github/contributors/wenjiedu/pypots?color=D8E699&label=Contributors&logo=GitHub">
    </a>
    <a href="https://star-history.com/#wenjiedu/pypots">
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/wenjiedu/pypots?logo=None&color=6BB392&label=%E2%98%85%20Stars">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/network/members">
        <img alt="GitHub Repo forks" src="https://img.shields.io/github/forks/wenjiedu/pypots?logo=forgejo&logoColor=black&label=Forks">
    </a>
    <a href="https://codeclimate.com/github/WenjieDu/PyPOTS">
        <img alt="Code Climate maintainability" src="https://img.shields.io/codeclimate/maintainability-percentage/WenjieDu/PyPOTS?color=3C7699&label=Maintainability&logo=codeclimate">
    </a>
    <a href="https://coveralls.io/github/WenjieDu/PyPOTS">
        <img alt="Coveralls coverage" src="https://img.shields.io/coverallsCoverage/github/WenjieDu/PyPOTS?branch=main&logo=coveralls&color=75C1C4&label=Coverage">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/actions/workflows/testing_ci.yml">
        <img alt="GitHub Testing" src="https://img.shields.io/github/actions/workflow/status/wenjiedu/pypots/testing_ci.yml?logo=circleci&color=C8D8E1&label=CI">
    </a>
    <a href="https://docs.pypots.com">
        <img alt="Docs building" src="https://img.shields.io/readthedocs/pypots?logo=readthedocs&label=Docs&logoColor=white&color=395260">
    </a>
    <a href="https://anaconda.org/conda-forge/pypots">
        <img alt="Conda downloads" src="https://img.shields.io/endpoint?url=https://pypots.com/figs/downloads_badges/conda_pypots_downloads.json">
    </a>
    <a href="https://pepy.tech/project/pypots">
        <img alt="PyPI downloads" src="https://img.shields.io/endpoint?url=https://pypots.com/figs/downloads_badges/pypi_pypots_downloads.json">
    </a>
    <a href="https://arxiv.org/abs/2305.18811">
        <img alt="arXiv DOI" src="https://img.shields.io/badge/DOI-10.48550/arXiv.2305.18811-F8F7F0">
    </a>
</p>

⦿ `动机`：由于各种原因，如收集传感器故障、通信错误和意外故障，实际环境中的时间序列常见缺失值。这使得部分观测时间序列（POTS）在开放世界建模中成为普遍问题，阻碍了高级数据分析。尽管这个问题很重要，但在POTS上的机器学习领域仍然缺乏专门的工具箱。PyPOTS旨在填补这一空白。

⦿ `使命`：PyPOTS（发音为 "Pie Pots"）旨在成为一个方便的工具箱，使机器学习在POTS上变得容易而不是繁琐，帮助工程师和研究人员更多地关注他们手头的核心问题，而不是如何处理数据中的缺失部分。PyPOTS将继续整合经典和最新的机器学习算法，用于部分观测的多变量时间序列。当然，除了各种算法，PyPOTS还将具有统一的API、详细的文档和跨算法的交互式示例作为教程。

🤗 **请** 星标此仓库以帮助其他人注意到 PyPOTS，如果您认为它是一个有用的工具箱。
**请** 在您的出版物中适当[引用 PyPOTS](https://github.com/WenjieDu/PyPOTS#-citing-pypots)，如果它有助于您的研究。这对我们的开源研究意义重大！谢谢！

接下来的内容组织如下：
[**❖ 可用算法**](#-available-algorithms),
[**❖ PyPOTS 生态系统**](#-pypots-ecosystem),
[**❖ 安装**](#-installation),
[**❖ 使用**](#-usage),
[**❖ 引用 PyPOTS**](#-citing-pypots),
[**❖ 贡献**](#-contribution),
[**❖ 社区**](#-community).
## ❖ 可用算法
PyPOTS 支持在具有缺失值的多变量部分观测时间序列上进行填补、分类、聚类、预测和异常检测任务。下表显示了 PyPOTS 中每种算法的可用性。
✅ 符号表示相应任务的算法可用（请注意，模型将持续更新，以处理当前不支持的任务。敬请期待❗️）。
任务类型缩写如下：**`IMPU`**: 填补; **`FORE`**: 预测;
**`CLAS`**: 分类; **`CLUS`**: 聚类; **`ANOD`**: 异常检测。
论文参考文献都列在本 readme 文件的底部。

🌟 从 **v0.2** 开始，PyPOTS 中的所有神经网络模型都支持超参数优化。
此功能是通过 [Microsoft NNI](https://github.com/microsoft/nni) 框架实现的。您可能希望参考我们的时间序列
填补调查仓库 [Awesome_Imputation](https://github.com/WenjieDu/Awesome_Imputation) 来了解如何配置和
调整超参数。
🔥 注意，Transformer, Crossformer, PatchTST, DLinear, ETSformer, FEDformer, Informer, Autoformer 在其原始论文中不是作为填补方法提出的，
并且不能接受 POTS 作为输入。**为了使它们适用于 POTS 数据，我们应用了嵌入策略和训练方法（ORT+MIT）
与 [SAITS 论文](https://arxiv.org/pdf/2202.08516) 中相同。**

| **类型**      | **算法**         | **IMPU** | **FORE** | **CLAS** | **CLUS** | **ANOD** | **年份** |
|:--------------|:-----------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| 神经网络    | SAITS[^1]        |    ✅     |          |          |          |          |   2023   |
| 神经网络    | Crossformer[^16] |    ✅     |          |          |          |          |   2023   |
| 神经网络    | TimesNet[^14]    |    ✅     |          |          |          |          |   2023   |
| 神经网络    | PatchTST[^18]    |    ✅     |          |          |          |          |   2023   |
| 神经网络    | DLinear[^17]     |    ✅     |          |          |          |          |   2023   |
| 神经网络    | ETSformer[^19]   |    ✅     |          |          |          |          |   2023   |
| 神经网络    | FEDformer[^20]   |    ✅     |          |          |          |          |   2022   |
| 神经网络    | Raindrop[^5]     |          |          |    ✅     |          |          |   2022   |
| 神经网络    | Informer[^21]    |    ✅     |          |          |          |          |   2021   |
| 神经网络    | Autoformer[^15]  |    ✅     |          |          |          |          |   2021   |
| 神经网络    | CSDI[^12]        |    ✅     |    ✅     |          |          |          |   2021   |
| 神经网络    | US-GAN[^10]      |    ✅     |          |          |          |          |   2021   |
| 神经网络    | CRLI[^6]         |          |          |          |    ✅     |          |   2021   |
| 概率模型 | BTTF[^8]         |          |    ✅     |          |          |          |   2021   |
| 神经网络    | GP-VAE[^16]      |    ✅     |          |          |          |          |   2020   |
| 神经网络    | VaDER[^7]        |          |          |          |    ✅     |          |   2019   |
| 神经网络    | M-RNN[^9]        |    ✅     |          |          |          |          |   2019   |
| 神经网络    | BRITS[^3]        |    ✅     |          |    ✅     |          |          |   2018   |
| 神经网络    | GRU-D[^4]        |    ✅     |          |    ✅     |          |          |   2018   |
| 神经网络    | Transformer[^2]  |    ✅     |          |          |          |          |   2017   |
| 朴素方法    | LOCF/NOCB        |    ✅     |          |          |          |          |          |
| 朴素方法    | Mean             |    ✅     |          |          |          |          |          |
| 朴素方法    | Median           |    ✅     |          |          |          |          |          |


## ❖ PyPOTS 生态系统
在 PyPOTS 中，一切都与我们熟悉的咖啡相关。是的，这是一个咖啡宇宙！
如您所见，PyPOTS 徽标中有一个咖啡壶。
还有什么呢？请继续阅读 ;-)

<a href="https://github.com/WenjieDu/TSDB">
    <img src="https://pypots.com/figs/pypots_logos/TSDB/logo_FFBG.svg" align="left" width="140" alt="TSDB logo"/>
</a>

👈 在 PyPOTS 中，时间序列数据集被视为咖啡豆，POTS 数据集是带有意义的不完整咖啡豆。
为了使各种公共时间序列数据集方便用户使用，
<i>时间序列数据豆 (TSDB)</i> 被创建，使加载时间序列数据集变得超级简单！
立即访问 [TSDB](https://github.com/WenjieDu/TSDB) 了解更多关于这个便利工具的信息 🛠，它现在支持总共 168 个开源数据集！

<a href="https://github.com/WenjieDu/PyGrinder">
    <img src="https://pypots.com/figs/pypots_logos/PyGrinder/logo_FFBG.svg" align="right" width="140" alt="PyGrinder logo"/>
</a>

👉 为了模拟带有缺失性的现实世界数据豆，生态系统库 [PyGrinder](https://github.com/WenjieDu/PyGrinder)，
一个帮助将您的咖啡豆研磨成不完整豆的工具包，已被创建。缺失模式根据罗宾的理论[^13]分为三类：
MCAR（完全随机缺失），MAR（随机缺失）和 MNAR（非随机缺失）。
PyGrinder 支持所有这些，并且还支持与缺失相关的附加功能。
使用 PyGrinder，您可以通过一行代码将合成缺失值引入您的数据集。

<a href="https://github.com/WenjieDu/BrewPOTS">
    <img src="https://pypots.com/figs/pypots_logos/BrewPOTS/logo_FFBG.svg" align="left" width="140" alt="BrewPOTS logo"/>
</a>

👈 现在我们有了豆子、研磨机和壶，怎样才能泡一杯咖啡呢？教程是必不可少的！
考虑到未来的工作量，PyPOTS 教程在一个单独的仓库中发布，
您可以在 [BrewPOTS](https://github.com/WenjieDu/BrewPOTS) 中找到它们。
现在就去看看吧，学
