<a href="https://github.com/WenjieDu/PyPOTS">
    <img src="https://pypots.com/figs/pypots_logos/PyPOTS/logo_FFBG.svg" width="200" align="right">
</a>

<h3 align="center">欢迎来到PyPOTS</h3>

<p align="center"><i>一个使用机器学习建模部分观测时间序列(POTS)的Python算法工具库</i></p>

<p align="center">
    <a href="https://docs.pypots.com/en/latest/install.html#reasons-of-version-limitations-on-dependencies">
       <img alt="Python version" src="https://img.shields.io/badge/Python-v3.8+-E97040?logo=python&logoColor=white">
    </a>
    <a href="https://www.google.com/search?q=%22PyPOTS%22+site%3Apytorch.org">
        <img alt="powered by Pytorch" src="https://img.shields.io/badge/PyTorch-%E2%9D%A4%EF%B8%8F-F8C6B5?logo=pytorch&logoColor=white">
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
    <a href="https://coveralls.io/github/WenjieDu/PyPOTS?branch=full_test">
        <img alt="Coveralls coverage" src="https://img.shields.io/coverallsCoverage/github/WenjieDu/PyPOTS?branch=full_test&logo=coveralls&color=75C1C4&label=Coverage">
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
    <a href="https://github.com/WenjieDu/PyPOTS/blob/main/README.md">
        <img alt="README in English" src="https://pypots.com/figs/pypots_logos/readme/US.svg">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/blob/main/README_zh.md">
        <img alt="README in Chinese" src="https://pypots.com/figs/pypots_logos/readme/CN.svg">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS">
        <img alt="PyPOTS Hits" src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FPyPOTS%2FPyPOTS&count_bg=%23009A0A&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Hits&edge_flat=false">
    </a>
</p>

⦿ `开发背景`: 由于传感器故障、通信异常以及不可预见的未知原因, 在现实环境中收集的时间序列数据普遍存在缺失值,
这使得部分观测时间序列(partially-observed time series, 简称为POTS)成为现实世界数据的建模中普遍存在的问题.
数据缺失会严重阻碍数据的高级分析、建模、与后续应用, 所以如何直接面向POTS建模成为一个亟需解决的问题.
尽管关于在POTS上进行不同任务的机器学习算法已经有了不少的研究, 但当前没有专门针对POTS建模开发的工具箱.
因此, 旨在填补该领域空白的"PyPOTS"应运而生.

⦿ `应用意义`: PyPOTS(发音为"Pie Pots")是一个易上手的工具箱, 工程师和研究人员可以通过PyPOTS轻松地处理POTS数据建模问题,
进而将注意力更多地聚焦在要解决的核心问题上. PyPOTS会持续不断的更新关于部分观测多变量时间序列的经典算法和先进算法.
除此之外, PyPOTS还提供了统一的应用程序接口,详细的算法学习指南和应用示例.

🤗 如果你认为PyPOTS有用, 请星标🌟该项目来帮助更多人注意到PyPOTS的存在.
如果PyPOTS对你的研究有帮助, 请在你的研究中[引用PyPOTS](#-引用pypots).
这是对我们开源研究工作的最大支持, 谢谢！

该说明文档的后续内容如下:
[**❖ 支持的算法**](#-支持的算法),
[**❖ PyPOTS生态系统**](#-pypots生态系统),
[**❖ 安装教程**](#-安装教程),
[**❖ 使用案例**](#-使用案例),
[**❖ 引用PyPOTS**](#-引用pypots),
[**❖ 贡献声明**](#-贡献声明),
[**❖ 社区组织**](#-社区组织).

## ❖ 支持的算法

PyPOTS当前支持多变量POTS数据的插补, 预测, 分类, 聚类以及异常检测五类任务. 下表描述了当前PyPOTS中所集成的算法以及对应不同任务的可用性.
符号`✅`表示该算法当前可用于相应的任务(注意, 目前模型尚不支持的任务在未来版本中可能会逐步添加, 敬请关注！).
算法的参考文献以及论文链接在该文档底部可以找到.

🌟 自**v0.2**版本开始, PyPOTS中所有神经网络模型都支持超参数调优. 该功能基于[微软的NNI](https://github.com/microsoft/nni)
框架实现.
你可以通过参考我们的时间序列插补综述和基准评估项目的代码[Awesome_Imputation](https://github.com/WenjieDu/Awesome_Imputation)
来了解如何使用PyPOTS调优模型的超参.

🔥 请注意: 表格中名称带有`🧑‍🔧`的模型(例如Transformer, iTransformer, Informer等)在它们的原始论文中并非作为可以处理POTS数据的算法提出,
所以这些模型的输入中不能带有缺失值, 无法接受POTS数据作为输入, 更加不是插补算法.
**为了使上述模型能够适用于POTS数据, 我们采用了与[SAITS论文](https://arxiv.org/pdf/2202.08516)[^1]
中相同的embedding策略和训练方法(ORT+MIT)对它们进行改进**.

| **类型**        | **算法**                                                                                                                           | **插补** | **预测** | **分类** | **聚类** | **异常检测** | **年份 - 刊物**                                        |
|:--------------|:---------------------------------------------------------------------------------------------------------------------------------|:------:|:------:|:------:|:------:|:--------:|:---------------------------------------------------|
| LLM&TSFM      | <a href="https://time-series.ai"><img src="https://time-series.ai/static/figs/robot.svg" width="26px"> Time-Series.AI</a>  [^36] |    ✅     |    ✅     |    ✅     |    ✅     |    ✅     | <a href="https://time-series.ai">Join waitlist</a> |
| LLM           | Time-LLM🧑‍🔧[^45]                                                                                                               |    ✅     |    ✅     |          |          |          | `2024 - ICLR`                                      |
| TSFM          | MOMENT[^47]                                                                                                                      |    ✅     |    ✅     |          |          |          | `2024 - ICML`                                      |
| Neural Net    | TEFN🧑‍🔧[^39]                                                                                                                   |    ✅     |    ✅     |          |          |          | `2024 - arXiv`                                     |
| Neural Net    | FITS🧑‍🔧[^41]                                                                                                                   |    ✅     |    ✅     |          |          |          | `2024 - ICLR`                                      |
| Neural Net    | TimeMixer[^37]                                                                                                                   |    ✅     |    ✅     |          |          |          | `2024 - ICLR`                                      |
| Neural Net    | iTransformer🧑‍🔧[^24]                                                                                                           |    ✅     |          |          |          |          | `2024 - ICLR`                                      |
| Neural Net    | ModernTCN[^38]                                                                                                                   |    ✅     |          |          |          |          | `2024 - ICLR`                                      |
| Neural Net    | ImputeFormer🧑‍🔧[^34]                                                                                                           |    ✅     |          |          |          |          | `2024 - KDD`                                       |
| Neural Net    | SAITS[^1]                                                                                                                        |    ✅     |          |          |          |          | `2023 - ESWA`                                      |
| LLM           | GPT4TS[^46]                                                                                                                      |    ✅     |    ✅     |          |          |          | `2023 - NeurIPS`                                   |
| Neural Net    | FreTS🧑‍🔧[^23]                                                                                                                  |    ✅     |          |          |          |          | `2023 - NeurIPS`                                   |
| Neural Net    | Koopa🧑‍🔧[^29]                                                                                                                  |    ✅     |          |          |          |          | `2023 - NeurIPS`                                   |
| Neural Net    | Crossformer🧑‍🔧[^16]                                                                                                            |    ✅     |          |          |          |          | `2023 - ICLR`                                      |
| Neural Net    | TimesNet[^14]                                                                                                                    |    ✅     |          |          |          |          | `2023 - ICLR`                                      |
| Neural Net    | PatchTST🧑‍🔧[^18]                                                                                                               |    ✅     |          |          |          |          | `2023 - ICLR`                                      |
| Neural Net    | ETSformer🧑‍🔧[^19]                                                                                                              |    ✅     |          |          |          |          | `2023 - ICLR`                                      |
| Neural Net    | MICN🧑‍🔧[^27]                                                                                                                   |    ✅     |          |          |          |          | `2023 - ICLR`                                      |
| Neural Net    | DLinear🧑‍🔧[^17]                                                                                                                |    ✅     |          |          |          |          | `2023 - AAAI`                                      |
| Neural Net    | TiDE🧑‍🔧[^28]                                                                                                                   |    ✅     |          |          |          |          | `2023 - TMLR`                                      |
| Neural Net    | CSAI[^42]                                                                                                                        |    ✅     |          |    ✅     |          |          | `2023 - arXiv`                                     |
| Neural Net    | SegRNN🧑‍🔧[^43]                                                                                                                 |    ✅     |          |          |          |          | `2023 - arXiv`                                     |
| Neural Net    | SCINet🧑‍🔧[^30]                                                                                                                 |    ✅     |          |          |          |          | `2022 - NeurIPS`                                   |
| Neural Net    | Nonstationary Tr.🧑‍🔧[^25]                                                                                                      |    ✅     |          |          |          |          | `2022 - NeurIPS`                                   |
| Neural Net    | FiLM🧑‍🔧[^22]                                                                                                                   |    ✅     |          |          |          |          | `2022 - NeurIPS`                                   |
| Neural Net    | RevIN_SCINet🧑‍🔧[^31]                                                                                                           |    ✅     |          |          |          |          | `2022 - ICLR`                                      |
| Neural Net    | Pyraformer🧑‍🔧[^26]                                                                                                             |    ✅     |          |          |          |          | `2022 - ICLR`                                      |
| Neural Net    | Raindrop[^5]                                                                                                                     |          |          |    ✅     |          |          | `2022 - ICLR`                                      |
| Neural Net    | FEDformer🧑‍🔧[^20]                                                                                                              |    ✅     |          |          |          |          | `2022 - ICML`                                      |
| Neural Net    | Autoformer🧑‍🔧[^15]                                                                                                             |    ✅     |          |          |          |          | `2021 - NeurIPS`                                   |
| Neural Net    | CSDI[^12]                                                                                                                        |    ✅     |    ✅     |          |          |          | `2021 - NeurIPS`                                   |
| Neural Net    | Informer🧑‍🔧[^21]                                                                                                               |    ✅     |          |          |          |          | `2021 - AAAI`                                      |
| Neural Net    | US-GAN[^10]                                                                                                                      |    ✅     |          |          |          |          | `2021 - AAAI`                                      |
| Neural Net    | CRLI[^6]                                                                                                                         |          |          |          |    ✅     |          | `2021 - AAAI`                                      |
| Probabilistic | BTTF[^8]                                                                                                                         |          |    ✅     |          |          |          | `2021 - TPAMI`                                     |
| Neural Net    | StemGNN🧑‍🔧[^33]                                                                                                                |    ✅     |          |          |          |          | `2020 - NeurIPS`                                   |
| Neural Net    | Reformer🧑‍🔧[^32]                                                                                                               |    ✅     |          |          |          |          | `2020 - ICLR`                                      |
| Neural Net    | GP-VAE[^11]                                                                                                                      |    ✅     |          |          |          |          | `2020 - AISTATS`                                   |
| Neural Net    | VaDER[^7]                                                                                                                        |          |          |          |    ✅     |          | `2019 - GigaSci.`                                  |
| Neural Net    | M-RNN[^9]                                                                                                                        |    ✅     |          |          |          |          | `2019 - TBME`                                      |
| Neural Net    | BRITS[^3]                                                                                                                        |    ✅     |          |    ✅     |          |          | `2018 - NeurIPS`                                   |
| Neural Net    | GRU-D[^4]                                                                                                                        |    ✅     |          |    ✅     |          |          | `2018 - Sci. Rep.`                                 |
| Neural Net    | TCN🧑‍🔧[^35]                                                                                                                    |    ✅     |          |          |          |          | `2018 - arXiv`                                     |
| Neural Net    | Transformer🧑‍🔧[^2]                                                                                                             |    ✅     |    ✅     |          |          |          | `2017 - NeurIPS`                                   |
| MF            | TRMF[^44]                                                                                                                        |    ✅     |          |          |          |          | `2016 - NeurIPS`                                   |
| Naive         | Lerp[^40]                                                                                                                        |    ✅     |          |          |          |          |                                                    |
| Naive         | LOCF/NOCB                                                                                                                        |    ✅     |          |          |          |          |                                                    |
| Naive         | Mean                                                                                                                             |    ✅     |          |          |          |          |                                                    |
| Naive         | Median                                                                                                                           |    ✅     |          |          |          |          |                                                    |

🙋 上表中`LLM (Large Language Model)`, `TSFM (Time-Series Foundation Model)`之间的区别:
`LLM`是指在大规模文本数据上进行预训练的模型, 可以针对特定任务进行微调.
`TSFM`是指在大规模时间序列数据上进行预训练的模型，其灵感来自CV和NLP领域中基座模型的最新成就.

💯 现在贡献你的模型来增加你的研究影响力！PyPOTS的下载量正在迅速增长
(**[目前PyPI上总共超过60万次且每日超1000的下载](https://www.pepy.tech/projects/pypots)**),
你的工作将被社区广泛使用和引用. 请参阅[贡献指南](#-%E8%B4%A1%E7%8C%AE%E5%A3%B0%E6%98%8E)
, 了解如何将模型包含在PyPOTS中.

## ❖ PyPOTS生态系统

在PyPOTS生态系统中, 一切都与我们熟悉的咖啡息息相关, 甚至可以将其视为一杯咖啡的诞生过程！
如你所见, PyPOTS的标志中有一个咖啡壶. 除此之外还需要什么呢？请接着看下去、

<a href="https://github.com/WenjieDu/TSDB">
    <img src="https://pypots.com/figs/pypots_logos/TSDB/logo_FFBG.svg" align="left" width="140" alt="TSDB logo"/>
</a>

👈 在PyPOTS中, 数据可以被看作是咖啡豆, 而写的携带缺失值的POTS数据则是不完整的咖啡豆.
为了让用户能够轻松使用各种开源的时间序列数据集, 我们创建了开源时间序列数据集的仓库 Time Series Data Beans (TSDB)
(可以将其视为咖啡豆仓库),
TSDB让加载开源时序数据集变得超级简单！访问 [TSDB](https://github.com/WenjieDu/TSDB), 了解更多关于TSDB的信息,
目前总共支持172个开源数据集！

<a href="https://github.com/WenjieDu/PyGrinder">
    <img src="https://pypots.com/figs/pypots_logos/PyGrinder/logo_FFBG.svg" align="right" width="140" alt="PyGrinder logo"/>
</a>

👉
为了在真实数据中模拟缺失进而获得不完整的咖啡豆,
我们创建了生态系统中的另一个仓库[PyGrinder](https://github.com/WenjieDu/PyGrinder)
(可以将其视为磨豆机),
帮助你在数据集中模拟缺失数据, 用于评估机器学习算法. 根据Robin的理论[^13], 缺失模式分为三类：
完全随机缺失(missing completely at random, 简称为MCAR)、随机缺失(missing at random, 简称为MAR)和非随机缺失(missing not at
random, 简称为MNAR ).
PyGrinder支持以上所有模式并提供与缺失相关的其他功能函数. 通过PyGrinder, 你可以仅仅通过一行代码就将模拟缺失引入你的数据集中.

<a href="https://github.com/WenjieDu/BenchPOTS">
    <img src="https://pypots.com/figs/pypots_logos/BenchPOTS/logo_FFBG.svg" align="left" width="140" alt="BenchPOTS logo"/>
</a>

👈
为了评估机器学习算法在POTS数据上的性能,
我们创建了生态系统中的另一个仓库[BenchPOTS](https://github.com/WenjieDu/BenchPOTS),
其提供了标准且统一的数据预处理管道来帮助你在多种任务上衡量不同POTS算法的性能.

<a href="https://github.com/WenjieDu/BrewPOTS">
    <img src="https://pypots.com/figs/pypots_logos/BrewPOTS/logo_FFBG.svg" align="right" width="140" alt="BrewPOTS logo"/>
</a>

👉 现在我们有了咖啡豆(beans)、磨豆机(grinder)和咖啡壶(pot), 让我们坐在长凳(bench)上想想如何萃取一杯咖啡呢？
教程必不可少！考虑到未来的工作量, PyPOTS的相关教程将发布在一个独立的仓库[BrewPOTS](https://github.com/WenjieDu/BrewPOTS)
中. 点击访问查看教程, 学习如何萃取你的POTS数据！

<p align="center">
<a href="https://pypots.com/ecosystem/">
    <img src="https://pypots.com/figs/pypots_logos/Ecosystem/PyPOTS_Ecosystem_Pipeline.png" width="95%"/>
</a>
<br>
<b>☕️ 欢迎来到 PyPOTS 生态系统 !</b>
</p>

## ❖ 安装教程

你可以参考PyPOTS文档中的 [安装说明](https://docs.pypots.com/en/latest/install.html) 以获取更详细的指南.
PyPOTS可以在 [PyPI](https://pypi.python.org/pypi/pypots) 和 [Anaconda](https://anaconda.org/conda-forge/pypots) 上安装.
你可以按照以下方式安装PyPOTS(同样适用于
[TSDB](https://github.com/WenjieDu/TSDB), [PyGrinder](https://github.com/WenjieDu/PyGrinder),
[BenchPOTS](https://github.com/WenjieDu/BenchPOTS), 和[AI4TS](https://github.com/WenjieDu/AI4TS):)：

```bash
# 通过pip安装
pip install pypots            # 首次安装
pip install pypots --upgrade  # 更新为最新版本
# 利用最新源代码安装最新版本, 可能带有尚未正式发布的最新功能
pip install https://github.com/WenjieDu/PyPOTS/archive/main.zip

# 通过conda安装
conda install conda-forge::pypots  # 首次安装
conda update  conda-forge::pypots  # 更新为最新版本
```

## ❖ 使用案例

除了[BrewPOTS](https://github.com/WenjieDu/BrewPOTS)之外, 你还可以在Google Colab
<a href="https://colab.research.google.com/drive/1HEFjylEy05-r47jRy0H9jiS_WhD0UWmQ">
<img src="https://img.shields.io/badge/GoogleColab-PyPOTS教程-F9AB00?logo=googlecolab&logoColor=white" alt="Colab tutorials" align="center"/>
</a>上找到一个简单且快速的入门教程. 如果你有其他问题, 请参考[PyPOTS文档](https://docs.pypots.com).
你也可以在我们的[社区](#-community)中提问, 或直接[发起issue](https://github.com/WenjieDu/PyPOTS/issues).

下面, 我们为你演示使用PyPOTS进行POTS数据插补的示例：

<details open>
<summary><b>点击此处查看 SAITS 模型应用于 PhysioNet2012 数据集插补任务的简单案例:</b></summary>

``` python
import numpy as np
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar, calc_missing_rate
from benchpots.datasets import preprocess_physionet2012
data = preprocess_physionet2012(subset='set-a', rate=0.1)  # 我们的工具库会自动下载并解压数据集
train_X, val_X, test_X = data["train_X"], data["val_X"], data["test_X"]
print(train_X.shape)  # (n_samples, n_steps, n_features)
print(val_X.shape)  # 验证集的样本数与训练集不同（n_samples不同），但样本长度（n_steps）和特征维度（n_features）一致
print(f"训练集 train_X 中缺失值的比例为 {calc_missing_rate(train_X):.1%}")
train_set = {"X": train_X}  # 训练集只需包含不完整时间序列
val_set = {
    "X": val_X,
    "X_ori": data["val_X_ori"],  # 验证集中我们需要真实值用于评估和选择模型
}
test_set = {"X": test_X}  # 测试集仅提供待填补的不完整时间序列
test_X_ori = data["test_X_ori"]  # test_X_ori 包含用于最终评估的真实值
indicating_mask = np.isnan(test_X) ^ np.isnan(test_X_ori)  # 生成指示掩码：标记出测试集中人为添加的缺失位置（X中存在缺失但X_ori中不缺失的位置）

from pypots.imputation import SAITS  # 导入你想要使用的模型
from pypots.nn.functional import calc_mae
saits = SAITS(n_steps=train_X.shape[1], n_features=train_X.shape[2], n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64, d_ffn=128, dropout=0.1, epochs=5)
saits.fit(train_set, val_set)  # 在数据集上训练模型
imputation = saits.impute(test_set)  # 对测试集中原始缺失和人为缺失的值进行填补
mae = calc_mae(imputation, np.nan_to_num(test_X_ori), indicating_mask)  # 在人为添加的缺失位置上计算 MAE（对比填补结果与真实值）
saits.save("save_it_here/saits_physionet2012.pypots")  # 保存模型供后续使用
saits.load("save_it_here/saits_physionet2012.pypots")  # 重新加载模型用于后续填补或继续训练
```

</details>

## ❖ 引用PyPOTS

> [!TIP]
> **[2024年6月更新]** 😎
> 第一个全面的时间序列插补基准论文[TSI-Bench: Benchmarking Time Series Imputation](https://arxiv.org/abs/2406.12747)
> 现在来了.
> 所有代码开源在[Awesome_Imputation](https://github.com/WenjieDu/Awesome_Imputation)
> 仓库中. 通过近35,000个实验, 我们对28种imputation方法, 3种缺失模式(点, 序列, 块), 各种缺失率, 和8个真实数据集进行了全面的基准研究.
>
> **[2024年2月更新]** 🎉
> 我们的综述论文[Deep Learning for Multivariate Time Series Imputation: A Survey](https://arxiv.org/abs/2402.04059)
> 已在 arXiv 上发布. 我们全面调研总结了最新基于深度学习的时间序列插补方法文献并对现有的方法进行分类, 此外,
> 还讨论了该领域当前的挑战和未来发展方向.

PyPOTS的论文可以[在arXiv上获取](https://arxiv.org/abs/2305.18811), 其5页的短版论文已被第9届SIGKDD international workshop
on Mining and Learning from Time Series ([MiLeTS'23](https://kdd-milets.github.io/milets2023/))收录, 与此同时,
PyPOTS也已被纳入[PyTorch Ecosystem](https://pytorch.org/ecosystem/). 我们正在努力将其发表在更具影响力的学术刊物上,
如JMLR (track for [Machine Learning Open Source Software](https://www.jmlr.org/mloss/)).
如果你在工作中使用了PyPOTS, 请按照以下格式引用我们的论文并为将项目设为星标🌟, 以便让更多人关注到它, 对此我们深表感谢🤗.

据不完全统计, 该[列表](https://scholar.google.com/scholar?as_ylo=2022&q=%E2%80%9CPyPOTS%E2%80%9D&hl=en>)
为当前使用PyPOTS并在其论文中引用PyPOTS的科学研究项目

```bibtex
@article{du2023pypots,
    title = {{PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series}},
    author = {Wenjie Du},
    journal = {arXiv preprint arXiv:2305.18811},
    year = {2023},
}
```

或者
> Wenjie Du. (2023).
> PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series.
> arXiv, abs/2305.18811. https://arxiv.org/abs/2305.18811

## ❖ 贡献声明

非常欢迎你为这个激动人心的项目做出贡献！

通过提交你的代码, 你将：

1. 把你开发完善的模型直接提供给PyPOTS的所有用户使用, 让你的工作更加广为人知.
   请查看我们的[收录标准](https://docs.pypots.com/en/latest/faq.html#inclusion-criteria).
   你也可以利用项目文件中的模板`template`(如：
   [pypots/imputation/template](https://github.com/WenjieDu/PyPOTS/tree/main/pypots/imputation/template))快速启动你的开发;
2. 成为[PyPOTS贡献者](https://github.com/WenjieDu/PyPOTS/graphs/contributors)之一,
   并在[PyPOTS网站](https://pypots.com/about/#volunteer-developers)上被列为志愿开发者；
3. 在PyPOTS发布新版本的[更新日志](https://github.com/WenjieDu/PyPOTS/releases)中被提及；

你也可以通过为该项目设置星标🌟, 帮助更多人关注它. 你的星标🌟既是对PyPOTS的认可, 也是对PyPOTS发展所做出的重要贡献！

<details open>
<summary>
    <b><i>
    👏 点击这里可以查看PyPOTS当前的星标者和分支者<br>
   我们为拥有越来越多的出色用户以及更多的星标✨而感到自豪：
    </i></b>
</summary>
<a href="https://github.com/WenjieDu/PyPOTS/stargazers">
    <img alt="PyPOTS stargazers" src="http://reporoster.com/stars/dark/WenjieDu/PyPOTS">
</a>
<br>
<a href="https://github.com/WenjieDu/PyPOTS/network/members">
    <img alt="PyPOTS forkers" src="http://reporoster.com/forks/dark/WenjieDu/PyPOTS">
</a>
</details>

👀请在[PyPOTS网站](https://pypots.com/users/)上查看我们用户所属机构的完整列表！

## ❖ 社区组织

我们非常关心用户的反馈, 因此我们正在建立PyPOTS社区:

- [Slack](https://join.slack.com/t/pypots-org/shared_invite/zt-1gq6ufwsi-p0OZdW~e9UW_IA4_f1OfxA):
  你可以在这里进行日常讨论、问答以及与我们的开发团队交流；
- [领英](https://www.linkedin.com/company/pypots)：你可以在这里获取官方公告和新闻；
- [微信公众号](https://mp.weixin.qq.com/s/X3ukIgL1QpNH8ZEXq1YifA)：你可以关注官方公众号并加入微信群聊参与讨论以及获取最新动态；

如果你有任何建议、想法、或打算分享与时间序列相关的论文, 欢迎加入我们！
PyPOTS社区是一个开放、透明、友好的社区, 让我们共同努力建设并改进PyPOTS！


[//]: # (Use APA reference style below)
[^1]: Du, W., Cote, D., & Liu, Y. (2023).
[SAITS: Self-Attention-based Imputation for Time Series](https://doi.org/10.1016/j.eswa.2023.119619).
*Expert systems with applications*.
[^2]: Vaswani, A., Shazeer, N.M., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., & Polosukhin, I. (
2017).
[Attention is All you Need](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html).
*NeurIPS 2017*.
[^3]: Cao, W., Wang, D., Li, J., Zhou, H., Li, L., & Li, Y. (2018).
[BRITS: Bidirectional Recurrent Imputation for Time Series](https://papers.nips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html).
*NeurIPS 2018*.
[^4]: Che, Z., Purushotham, S., Cho, K., Sontag, D.A., & Liu, Y. (2018).
[Recurrent Neural Networks for Multivariate Time Series with Missing Values](https://www.nature.com/articles/s41598-018-24271-9).
*Scientific Reports*.
[^5]: Zhang, X., Zeman, M., Tsiligkaridis, T., & Zitnik, M. (2022).
[Graph-Guided Network for Irregularly Sampled Multivariate Time Series](https://arxiv.org/abs/2110.05357). *ICLR 2022*.
[^6]: Ma, Q., Chen, C., Li, S., & Cottrell, G. W. (2021).
[Learning Representations for Incomplete Time Series Clustering](https://ojs.aaai.org/index.php/AAAI/article/view/17070).
*AAAI 2021*.
[^7]: Jong, J.D., Emon, M.A., Wu, P., Karki, R., Sood, M., Godard, P., Ahmad, A., Vrooman, H.A., Hofmann-Apitius, M., &
Fröhlich, H. (2019).
[Deep learning for clustering of multivariate clinical patient trajectories with missing values](https://academic.oup.com/gigascience/article/8/11/giz134/5626377).
*GigaScience*.
[^8]: Chen, X., & Sun, L. (2021).
[Bayesian Temporal Factorization for Multidimensional Time Series Prediction](https://arxiv.org/abs/1910.06366).
*IEEE transactions on pattern analysis and machine intelligence*.
[^9]: Yoon, J., Zame, W. R., & van der Schaar, M. (2019).
[Estimating Missing Data in Temporal Data Streams Using Multi-Directional Recurrent Neural Networks](https://ieeexplore.ieee.org/document/8485748).
*IEEE Transactions on Biomedical Engineering*.
[^10]: Miao, X., Wu, Y., Wang, J., Gao, Y., Mao, X., & Yin, J. (2021).
[Generative Semi-supervised Learning for Multivariate Time Series Imputation](https://ojs.aaai.org/index.php/AAAI/article/view/17086).
*AAAI 2021*.
[^11]: Fortuin, V., Baranchuk, D., Raetsch, G. & Mandt, S. (2020).
[GP-VAE: Deep Probabilistic Time Series Imputation](https://proceedings.mlr.press/v108/fortuin20a.html).
*AISTATS 2020*.
[^12]: Tashiro, Y., Song, J., Song, Y., & Ermon, S. (2021).
[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html).
*NeurIPS 2021*.
[^13]: Rubin, D. B. (1976).
[Inference and missing data](https://academic.oup.com/biomet/article-abstract/63/3/581/270932).
*Biometrika*.
[^14]: Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., & Long, M. (2023).
[TimesNet: Temporal 2d-variation modeling for general time series analysis](https://openreview.net/forum?id=ju_Uqw384Oq).
*ICLR 2023*
[^15]: Wu, H., Xu, J., Wang, J., & Long, M. (2021).
[Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting](https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html).
*NeurIPS 2021*.
[^16]: Zhang, Y., & Yan, J. (2023).
[Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting](https://openreview.net/forum?id=vSVLM2j9eie).
*ICLR 2023*.
[^17]: Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023).
[Are transformers effective for time series forecasting?](https://ojs.aaai.org/index.php/AAAI/article/view/26317).
*AAAI 2023*
[^18]: Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023).
[A time series is worth 64 words: Long-term forecasting with transformers](https://openreview.net/forum?id=Jbdc0vTOcol).
*ICLR 2023*
[^19]: Woo, G., Liu, C., Sahoo, D., Kumar, A., & Hoi, S. (2023).
[ETSformer: Exponential Smoothing Transformers for Time-series Forecasting](https://openreview.net/forum?id=5m_3whfo483).
*ICLR 2023*
[^20]: Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., & Jin, R. (2022).
[FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting](https://proceedings.mlr.press/v162/zhou22g.html).
*ICML 2022*.
[^21]: Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021).
[Informer: Beyond efficient transformer for long sequence time-series forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/17325).
*AAAI 2021*.
[^22]: Zhou, T., Ma, Z., Wen, Q., Sun, L., Yao, T., Yin, W., & Jin, R. (2022).
[FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting](https://proceedings.neurips.cc/paper_files/paper/2022/hash/524ef58c2bd075775861234266e5e020-Abstract-Conference.html).
*NeurIPS 2022*.
[^23]: Yi, K., Zhang, Q., Fan, W., Wang, S., Wang, P., He, H., An, N., Lian, D., Cao, L., & Niu, Z. (2023).
[Frequency-domain MLPs are More Effective Learners in Time Series Forecasting](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f1d16af76939f476b5f040fd1398c0a3-Abstract-Conference.html).
*NeurIPS 2023*.
[^24]: Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024).
[iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://openreview.net/forum?id=JePfAI8fah).
*ICLR 2024*.
[^25]: Liu, Y., Wu, H., Wang, J., & Long, M. (2022).
[Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html).
*NeurIPS 2022*.
[^26]: Liu, S., Yu, H., Liao, C., Li, J., Lin, W., Liu, A. X., & Dustdar, S. (2022).
[Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting](https://openreview.net/forum?id=0EXmFzUn5I).
*ICLR 2022*.
[^27]: Wang, H., Peng, J., Huang, F., Wang, J., Chen, J., & Xiao, Y. (2023).
[MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting](https://openreview.net/forum?id=zt53IDUR1U).
*ICLR 2023*.
[^28]: Das, A., Kong, W., Leach, A., Mathur, S., Sen, R., & Yu, R. (2023).
[Long-term Forecasting with TiDE: Time-series Dense Encoder](https://openreview.net/forum?id=pCbC3aQB5W).
*TMLR 2023*.
[^29]: Liu, Y., Li, C., Wang, J., & Long, M. (2023).
[Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors](https://proceedings.neurips.cc/paper_files/paper/2023/hash/28b3dc0970fa4624a63278a4268de997-Abstract-Conference.html).
*NeurIPS 2023*.
[^30]: Liu, M., Zeng, A., Chen, M., Xu, Z., Lai, Q., Ma, L., & Xu, Q. (2022).
[SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction](https://proceedings.neurips.cc/paper_files/paper/2022/hash/266983d0949aed78a16fa4782237dea7-Abstract-Conference.html).
*NeurIPS 2022*.
[^31]: Kim, T., Kim, J., Tae, Y., Park, C., Choi, J. H., & Choo, J. (2022).
[Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift](https://openreview.net/forum?id=cGDAkQo1C0p).
*ICLR 2022*.
[^32]: Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020).
[Reformer: The Efficient Transformer](https://openreview.net/forum?id=rkgNKkHtvB).
*ICLR 2020*.
[^33]: Cao, D., Wang, Y., Duan, J., Zhang, C., Zhu, X., Huang, C., Tong, Y., Xu, B., Bai, J., Tong, J., & Zhang, Q. (
2020).
[Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting](https://proceedings.neurips.cc/paper/2020/hash/cdf6581cb7aca4b7e19ef136c6e601a5-Abstract.html).
*NeurIPS 2020*.
[^34]: Nie, T., Qin, G., Mei, Y., & Sun, J. (2024).
[ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation](https://arxiv.org/abs/2312.01728).
*KDD 2024*.
[^35]: Bai, S., Kolter, J. Z., & Koltun, V. (2018).
[An empirical evaluation of generic convolutional and recurrent networks for sequence modeling](https://arxiv.org/abs/1803.01271).
*arXiv 2018*.
[^36]: Project Gungnir, the world 1st LLM for time-series multitask modeling, will meet you soon. 🚀 Missing values and
variable lengths in your datasets?
Hard to perform multitask learning with your time series? Not problems no longer. We'll open application for public beta
test recently ;-) Follow us, and stay tuned!
<a href="https://time-series.ai"><img src="https://time-series.ai/static/figs/robot.svg" width="20px" align="center">
Time-Series.AI</a>
[^37]: Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., ... & ZHOU, J. (2024).
[TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting](https://openreview.net/forum?id=7oLshfEIC2).
*ICLR 2024*.
[^38]: Luo, D., & Wang X. (2024).
[ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis](https://openreview.net/forum?id=vpJMJerXHU).
*ICLR 2024*.
[^39]: Zhan, T., He, Y., Deng, Y., Li, Z., Du, W., & Wen, Q. (2024).
[Time Evidence Fusion Network: Multi-source View in Long-Term Time Series Forecasting](https://arxiv.org/abs/2405.06419).
*arXiv 2024*.
[^40]: [Wikipedia: Linear interpolation](https://en.wikipedia.org/wiki/Linear_interpolation)
[^41]: Xu, Z., Zeng, A., & Xu, Q. (2024).
[FITS: Modeling Time Series with 10k parameters](https://openreview.net/forum?id=bWcnvZ3qMb).
*ICLR 2024*.
[^42]: Qian, L., Ibrahim, Z., Ellis, H. L., Zhang, A., Zhang, Y., Wang, T., & Dobson, R. (2023). 
[Knowledge Enhanced Conditional Imputation for Healthcare Time-series](https://arxiv.org/abs/2312.16713).
*arXiv 2023*.
[^43]: Lin, S., Lin, W., Wu, W., Zhao, F., Mo, R., & Zhang, H. (2023). 
[SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting](https://arxiv.org/abs/2308.11200).
*arXiv 2023*.
[^44]: Yu, H. F., Rao, N., & Dhillon, I. S. (2016). 
[Temporal regularized matrix factorization for high-dimensional time series prediction](https://papers.nips.cc/paper_files/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html).
*NeurIPS 2016*.
[^45]: Jin, M., Wang, S., Ma, L., Chu, Z., Zhang, J. Y., Shi, X., ... & Wen, Q. (2024). 
[Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://openreview.net/forum?id=Unb5CVPtae).
*ICLR 2024*.
[^46]: Zhou, T., Niu, P., Sun, L., & Jin, R. (2023). 
[One Fits All: Power General Time Series Analysis by Pretrained LM](https://openreview.net/forum?id=gMS6FVZvmF).
*NeurIPS 2023*.
[^47]: Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., & Dubrawski, A. (2024).
[MOMENT: A Family of Open Time-series Foundation Models](https://proceedings.mlr.press/v235/goswami24a.html).
*ICML 2024*.
