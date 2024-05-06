<a href="https://github.com/WenjieDu/PyPOTS">
    <img src="https://pypots.com/figs/pypots_logos/PyPOTS/logo_FFBG.svg" width="200" align="right">
</a>

<h3 align="center">欢迎来到PyPOTS</h3>

<p align="center"><i>一个致力于部分观测时间序列(POTS)机器学习任务的Python工具箱</i></p>

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
⦿ `开发背景`: 由于传感器故障、通信异常以及不可预见的未知原因，在现实环境中收集的时间序列数据普遍存在缺失值，这使得部分观测时间序列（partially-observed time series，简称为POTS）成为开放世界建模中普遍存在的问题。
数据缺失将会严重影响数据使用者后续的深入分析与应用，那么如何面向POTS建模成为一个亟需解决的问题。尽管已存在大量的研究，但当前没有专门针对POTS建模开发的工具箱。基于此，旨在填补该领域空白的“PyPOTS”工具箱应运而生。

⦿ `应用意义`: PyPOTS（发音为"Pie Pots"）是一个易上手的工具箱，工程师和研究人员可以通过PyPOTS轻松处理其数据集中的缺失部分，进而将注意力更多地聚焦在要解决的核心问题上。PyPOTS会持续不断的更新关于部分观测多变量时间序列（partially-observed multivariate time series）的经典算法和先进算法。除此之外，PyPOTS还提供了统一的应用程序接口（Application programming interfaces，简称为APIs）,详细的算法学习指南和应用示例。

🤗 如果您认为PyPOTS是一个有用的工具箱，请将该项目设为**星标**🌟，以帮助它被更多人所了解。<br>
🤗 如果PyPOTS对您的研究有帮助，请在您的成果中 [引用 PyPOTS](https://github.com/WenjieDu/PyPOTS#-citing-pypots)。这是对我们开源研究工作的最大支持，感谢您的支持！

该说明文档的后续内容如下:
[**❖ 支持的算法**](#-available-algorithms),
[**❖ PyPOTS 生态系统**](#-pypots-ecosystem),
[**❖ 安装教程**](#-installation),
[**❖ 使用案例**](#-usage),
[**❖ 引用 PyPOTS**](#-citing-pypots),
[**❖ 贡献声明**](#-contribution),
[**❖ 项目社区**](#-community)。


## ❖ 支持的算法 
PyPOTS当前支持多变量POTS数据的插补，预测，分类，聚类以及异常检测五类任务。下表描述了当前PyPOTS中所集成的算法以及对应不同任务的可用性。
符号 ✅ 表示该算法当前可用于相应的任务**<u>（请注意，我们将在未来持续发布更新以支持当前暂不支持的任务，敬请关注哦！）</u>**，相关参考文献见本文档末尾。

🌟 在 **v0.2**版本更新后, PyPOTS中所有神经网络模型都支持超参数优化。此功能基于 [Microsoft NNI](https://github.com/microsoft/nni) 框架实现。
您可以通过参考我们的时间序列插补综述项目 [Awesome_Imputation](https://github.com/WenjieDu/Awesome_Imputation)，了解如何配置和调整模型超参数。

🔥 请注意：Transformer, Crossformer, PatchTST, DLinear, ETSformer, FEDformer, Informer, Autoformer 模型在其原始论文中<u>**并未提及用作插补方法**</u>，并且这些模型也不接受POTS数据作为输入。
**为了使上述模型能够适用于POTS数据，我们采用了与 [SAITS paper](https://arxiv.org/pdf/2202.08516) 中相同的嵌入策略和训练方法（ORT+MIT）进行改进**。

| **类型**      | **算法**         | **插补** | **预测** | **分类** | **聚类** | **异常检测** | **年份** |
|:--------------|:-----------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| Neural Net    | SAITS[^1]        |    ✅     |          |          |          |          |   2023   |
| Neural Net    | Crossformer[^16] |    ✅     |          |          |          |          |   2023   |
| Neural Net    | TimesNet[^14]    |    ✅     |          |          |          |          |   2023   |
| Neural Net    | PatchTST[^18]    |    ✅     |          |          |          |          |   2023   |
| Neural Net    | DLinear[^17]     |    ✅     |          |          |          |          |   2023   |
| Neural Net    | ETSformer[^19]   |    ✅     |          |          |          |          |   2023   |
| Neural Net    | FEDformer[^20]   |    ✅     |          |          |          |          |   2022   |
| Neural Net    | Raindrop[^5]     |          |          |    ✅     |          |          |   2022   |
| Neural Net    | Informer[^21]    |    ✅     |          |          |          |          |   2021   |
| Neural Net    | Autoformer[^15]  |    ✅     |          |          |          |          |   2021   |
| Neural Net    | CSDI[^12]        |    ✅     |    ✅     |          |          |          |   2021   |
| Neural Net    | US-GAN[^10]      |    ✅     |          |          |          |          |   2021   |
| Neural Net    | CRLI[^6]         |          |          |          |    ✅     |          |   2021   |
| Probabilistic | BTTF[^8]         |          |    ✅     |          |          |          |   2021   |
| Neural Net    | GP-VAE[^16]      |    ✅     |          |          |          |          |   2020   |
| Neural Net    | VaDER[^7]        |          |          |          |    ✅     |          |   2019   |
| Neural Net    | M-RNN[^9]        |    ✅     |          |          |          |          |   2019   |
| Neural Net    | BRITS[^3]        |    ✅     |          |    ✅     |          |          |   2018   |
| Neural Net    | GRU-D[^4]        |    ✅     |          |    ✅     |          |          |   2018   |
| Neural Net    | Transformer[^2]  |    ✅     |          |          |          |          |   2017   |
| Naive         | LOCF/NOCB        |    ✅     |          |          |          |          |          |
| Naive         | Mean             |    ✅     |          |          |          |          |          |
| Naive         | Median           |    ✅     |          |          |          |          |          |


## ❖ PyPOTS 生态系统
在PyPOTS生态系统中，一切都与我们熟悉的咖啡息息相关，甚至可以将其视为一杯咖啡的诞生过程！
如您所见，PyPOTS的标志中有一个咖啡壶。除此之外还需要什么呢？请听我徐徐道来。

<a href="https://github.com/WenjieDu/TSDB">
    <img src="https://pypots.com/figs/pypots_logos/TSDB/logo_FFBG.svg" align="left" width="140" alt="TSDB logo"/>
</a>

👈 在PyPOTS中，时间序列数据可以被看作一连串的咖啡豆，而POTS数据则是带缺失部分的不完整咖啡豆。
为了让用户能够轻松使用各种开源的时间序列数据集，我们创建了时间序列数据库：<i>Time Series Data Beans (TSDB)</i>（可以将其视为咖啡豆仓库），使加载时间序列数据集变得超级简单！
访问 [TSDB](https://github.com/WenjieDu/TSDB)，了解更多关于该工具🛠的信息，它目前总共支持168个开源数据集！

<a href="https://github.com/WenjieDu/PyGrinder">
    <img src="https://pypots.com/figs/pypots_logos/PyGrinder/logo_FFBG.svg" align="right" width="140" alt="PyGrinder logo"/>
</a>

👉 为了在真实数据中模拟缺失进而获得不完整的咖啡豆，我们创建了生态系统中的另一个库：[PyGrinder](https://github.com/WenjieDu/PyGrinder)（可以将其视为磨豆机），
帮助您在输入数据集中模拟缺失。根据Robin的理论[^13]，缺失模式分为三类：完全随机缺失（missing completely at random，简称为MCAR）、随机缺失（missing at random，简称为MAR）和非随机缺失（missing not at random，简称为MNAR ）。
PyGrinder支持以上所有模式并提供与缺失相关的其他功能。通过PyGrinder，您可以仅仅通过一行代码将模拟缺失引入您的数据集中。


<a href="https://github.com/WenjieDu/BrewPOTS">
    <img src="https://pypots.com/figs/pypots_logos/BrewPOTS/logo_FFBG.svg" align="left" width="140" alt="BrewPOTS logo"/>
</a>

👈 现在我们有了咖啡豆、磨豆机和咖啡壶，那么如何给自己冲一杯咖啡呢？冲泡教程是必不可少的！考虑到未来的工作量，
PyPOTS的相关教程将发布在一个独立的仓库：[BrewPOTS](https://github.com/WenjieDu/BrewPOTS)中。点击查看案例，学习如何冲泡您的POTS数据！


<p align="center">
<a href="https://pypots.com/ecosystem/">
    <img src="https://pypots.com/figs/pypots_logos/Ecosystem/PyPOTS_Ecosystem_Pipeline.png" width="95%"/>
</a>
<br>
<b> ☕️ 欢迎来到 PyPOTS 生态系统 !</b>
</p>


## ❖ 安装教程
您可以参考PyPOTS文档中的 [安装说明](https://docs.pypots.com/en/latest/install.html) 以获取更详细的指南。
PyPOTS可以在 [PyPI](https://pypi.python.org/pypi/pypots) 和 [Anaconda](https://anaconda.org/conda-forge/pypots) 上安装。
您可以按照以下方式安装PyPOTS，TSDB以及PyGrinder：


``` bash
# 通过 pip
pip install pypots            # 首次安装
pip install pypots --upgrade  # 更新为最新版本
# 利用最新源代码安装最新版本，可能带有尚未正式发布的最新功能
pip install https://github.com/WenjieDu/PyPOTS/archive/main.zip

# 通过 conda
conda install -c conda-forge pypots  # 首次安装
conda update  -c conda-forge pypots  # 更新为最新版本
```


## ❖ 使用案例
除了 [BrewPOTS](https://github.com/WenjieDu/BrewPOTS) 之外, 您还可以在 Google Colab
<a href="https://colab.research.google.com/drive/1HEFjylEy05-r47jRy0H9jiS_WhD0UWmQ">
<img src="https://img.shields.io/badge/GoogleColab-PyPOTS_Tutorials-F9AB00?logo=googlecolab&logoColor=white" alt="Colab tutorials" align="center"/>
</a>上找到一个简单且快速的入门教程。如果您有其他问题，请参考 [PyPOTS文档](https://docs.pypots.com)。您也可以在我们的 [社区](#-community) 中 [提出问题](https://github.com/WenjieDu/PyPOTS/issues)。
<br>下面，我们为您演示使用PyPOTS进行POTS数据插补的示例，您可以点击以下链接查看：

<details open>
<summary><b>点击此处查看 SAITS 模型应用于 PhysioNet2012 数据集插补任务的简单案例:</b></summary>

``` python
# 数据预处理，使用PyPOTS生态帮助完成繁琐的数据预处理。
import numpy as np
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar
from pypots.data import load_specific_dataset
data = load_specific_dataset('physionet_2012')  # PyPOTS 将自动下载和解压数据
X = data['X']
num_samples = len(X['RecordID'].unique())
X = X.drop(['RecordID', 'Time'], axis = 1)
X = StandardScaler().fit_transform(X.to_numpy())
X = X.reshape(num_samples, 48, -1)
X_ori = X  # keep X_ori for validation
X = mcar(X, 0.1)  # 随机掩盖观测值的10%，作为基准数据
dataset = {"X": X}  # X 用于模型输入
print(X.shape)  # (11988, 48, 37), 11988 samples and each sample has 48 time steps, 37 features

# 模型训练，开始应用PyPOTS.
from pypots.imputation import SAITS
from pypots.utils.metrics import calc_mae
saits = SAITS(n_steps=48, n_features=37, n_layers=2, d_model=256, d_ffn=128, n_heads=4, d_k=64, d_v=64, dropout=0.1, epochs=10)
# 因为基准数据对模型不可知，将整个数据集作为训练集, 也可以把数据集分为训练/验证/测试集
saits.fit(dataset)  # 基于数据集训练模型
imputation = saits.impute(dataset)  # 插补数据集中原始缺失部分和人为掩盖缺失部分
indicating_mask = np.isnan(X) ^ np.isnan(X_ori)  # 用于计算插补误差的掩码矩阵
mae = calc_mae(imputation, np.nan_to_num(X_ori), indicating_mask)  #计算人为掩盖部分数据的平均绝对误差
saits.save("save_it_here/saits_physionet2012.pypots")  # 保存模型
saits.load("save_it_here/saits_physionet2012.pypots")  # 重新加载序列化的模型文件以进行后续的插补或训练
```
</details>


## ❖ 引用PyPOTS
> :bulb: **注意**
> **[2024年2月更新]** 😎 我们的综述论文 [Deep Learning for Multivariate Time Series Imputation: A Survey](https://arxiv.org/abs/2402.04059) 已在 arXiv 上发布，
> 代码已在GitHub 项目（[Awesome_Imputation](https://github.com/WenjieDu/Awesome_Imputation) ）上开源。
> 我们全面综述了最新基于深度学习的时间序列插补方法文献并对现有的方法进行分类，除此之外，还讨论了该领域当前的挑战和未来发展方向。
>
> **[2023年6月更新]** 🎉 PyPOTS论文的精简版已被第9届SIGKDD 时间序列挖掘和学习 国际研讨会([MiLeTS'23](https://kdd-milets.github.io/milets2023/))接收。
> 此外，PyPOTS已被纳入[PyTorch 生态系统](https://pytorch.org/ecosystem/)。

介绍PyPOTS的论文可以通过该 [地址](https://arxiv.org/abs/2305.18811) 在arXiv上获取，我们正在努力将其发表在更具影响力的学术期刊上，
例如持续跟进 [机器学习开源软件](https://www.jmlr.org/mloss/) 的JMLR期刊。
如果您在工作中使用了PyPOTS，请按照以下格式引用并为将项目设为星标🌟，以便让更多人关注到它，对此我们深表感谢🤗。

据不完全统计，该 [列表](https://scholar.google.com/scholar?as_ylo=2022&q=%E2%80%9CPyPOTS%E2%80%9D&hl=en>) 为当前使用PyPOTS并在其论文中引用PyPOTS的科学研究项目


``` bibtex
@article{du2023pypots,
title={{PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series}},
author={Wenjie Du},
year={2023},
eprint={2305.18811},
archivePrefix={arXiv},
primaryClass={cs.LG},
url={https://arxiv.org/abs/2305.18811},
doi={10.48550/arXiv.2305.18811},
}
```
或者
> Wenjie Du. (2023).
> PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series.
> arXiv, abs/2305.18811.https://arxiv.org/abs/2305.18811


## ❖ 贡献声明
非常欢迎您为这个有趣的项目做出贡献！
<br>&emsp;&emsp;通过提交您的代码，您将能收获：

1. 将您<u>**开发完善的模型**</u>直接供PyPOTS用户使用，让您的工作更加广为人知。
   请查看我们的 [纳入标准](https://docs.pypots.com/en/latest/faq.html#inclusion-criteria)。
   您也可以利用项目文件中的 `模板` (如：
   [pypots/imputation/template](https://github.com/WenjieDu/PyPOTS/tree/main/pypots/imputation/template)) 进行快速开发;
2. 成为 [PyPOTS 贡献者](https://github.com/WenjieDu/PyPOTS/graphs/contributors) 之一，
并在 [PyPOTS 网站](https://pypots.com/about/#volunteer-developers) 上被列为志愿开发者；
3. 您的贡献将在我们发布的 [说明](https://github.com/WenjieDu/PyPOTS/releases) 中被提及；

您也可以通过为该项目设置星标🌟 ，帮助更多人关注它。
您的星标🌟 既是对PyPOTS的认可，也是对PyPOTS发展所做出的重要贡献！

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

👀 在 [PyPOTS 网站](https://pypots.com/users/) 上可以查看我们用户所属机构的完整列表！


## ❖ 项目社区
我们非常关心用户的反馈，因此我们正在建立PyPOTS社区:

- [Slack](https://join.slack.com/t/pypots-org/shared_invite/zt-1gq6ufwsi-p0OZdW~e9UW_IA4_f1OfxA)：您可以在这里进行日常讨论、问答以及与我们的开发团队交流；
- [领英](https://www.linkedin.com/company/pypots)： 您可以在这里获取官方公告和新闻；
- [微信公众号](https://mp.weixin.qq.com/s/sNgZmgAyxDn2sZxXoWJYMA)： 您可以关注官方公众号并加入微信群聊参与讨论以及获取最新动态；

如果您有任何建议、想法、或打算分享与时间序列相关的论文，请加入我们并告诉我们。
PyPOTS社区一个开放、透明、友好的社区，让我们共同努力建设和改进PyPOTS！


[//]: # (Use APA reference style below)
[^1]: Du, W., Cote, D., & Liu, Y. (2023). [SAITS: Self-Attention-based Imputation for Time Series](https://doi.org/10.1016/j.eswa.2023.119619). *Expert systems with applications*.
[^2]: Vaswani, A., Shazeer, N.M., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., & Polosukhin, I. (2017). [Attention is All you Need](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html). *NeurIPS 2017*.
[^3]: Cao, W., Wang, D., Li, J., Zhou, H., Li, L., & Li, Y. (2018). [BRITS: Bidirectional Recurrent Imputation for Time Series](https://papers.nips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html). *NeurIPS 2018*.
[^4]: Che, Z., Purushotham, S., Cho, K., Sontag, D.A., & Liu, Y. (2018). [Recurrent Neural Networks for Multivariate Time Series with Missing Values](https://www.nature.com/articles/s41598-018-24271-9). *Scientific Reports*.
[^5]: Zhang, X., Zeman, M., Tsiligkaridis, T., & Zitnik, M. (2022). [Graph-Guided Network for Irregularly Sampled Multivariate Time Series](https://arxiv.org/abs/2110.05357). *ICLR 2022*.
[^6]: Ma, Q., Chen, C., Li, S., & Cottrell, G. W. (2021). [Learning Representations for Incomplete Time Series Clustering](https://ojs.aaai.org/index.php/AAAI/article/view/17070). *AAAI 2021*.
[^7]: Jong, J.D., Emon, M.A., Wu, P., Karki, R., Sood, M., Godard, P., Ahmad, A., Vrooman, H.A., Hofmann-Apitius, M., & Fröhlich, H. (2019). [Deep learning for clustering of multivariate clinical patient trajectories with missing values](https://academic.oup.com/gigascience/article/8/11/giz134/5626377). *GigaScience*.
[^8]: Chen, X., & Sun, L. (2021). [Bayesian Temporal Factorization for Multidimensional Time Series Prediction](https://arxiv.org/abs/1910.06366). *IEEE transactions on pattern analysis and machine intelligence*.
[^9]: Yoon, J., Zame, W. R., & van der Schaar, M. (2019). [Estimating Missing Data in Temporal Data Streams Using Multi-Directional Recurrent Neural Networks](https://ieeexplore.ieee.org/document/8485748). *IEEE Transactions on Biomedical Engineering*.
[^10]: Miao, X., Wu, Y., Wang, J., Gao, Y., Mao, X., & Yin, J. (2021). [Generative Semi-supervised Learning for Multivariate Time Series Imputation](https://ojs.aaai.org/index.php/AAAI/article/view/17086). *AAAI 2021*.
[^11]: Fortuin, V., Baranchuk, D., Raetsch, G. & Mandt, S. (2020). [GP-VAE: Deep Probabilistic Time Series Imputation](https://proceedings.mlr.press/v108/fortuin20a.html). *AISTATS 2020*.
[^12]: Tashiro, Y., Song, J., Song, Y., & Ermon, S. (2021). [CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html). *NeurIPS 2021*.
[^13]: Rubin, D. B. (1976). [Inference and missing data](https://academic.oup.com/biomet/article-abstract/63/3/581/270932). *Biometrika*.
[^14]: Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., & Long, M. (2023). [TimesNet: Temporal 2d-variation modeling for general time series analysis](https://openreview.net/forum?id=ju_Uqw384Oq). *ICLR 2023*
[^15]: Wu, H., Xu, J., Wang, J., & Long, M. (2021). [Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting](https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html). *NeurIPS 2021*.
[^16]: Zhang, Y., & Yan, J. (2023). [Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting](https://openreview.net/forum?id=vSVLM2j9eie). *ICLR 2023*.
[^17]: Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). [Are transformers effective for time series forecasting?](https://ojs.aaai.org/index.php/AAAI/article/view/26317). *AAAI 2023*
[^18]: Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). [A time series is worth 64 words: Long-term forecasting with transformers](https://openreview.net/forum?id=Jbdc0vTOcol). *ICLR 2023*
[^19]: Woo, G., Liu, C., Sahoo, D., Kumar, A., & Hoi, S. (2023). [ETSformer: Exponential Smoothing Transformers for Time-series Forecasting](https://openreview.net/forum?id=5m_3whfo483).  *ICLR 2023*
[^20]: Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., & Jin, R. (2022). [FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting](https://proceedings.mlr.press/v162/zhou22g.html). *ICML 2022*.
[^21]: Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). [Informer: Beyond efficient transformer for long sequence time-series forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/17325). *AAAI 2021*.



<details>
<summary>🏠 Visits</summary>
<a href="https://github.com/WenjieDu/PyPOTS">
    <img alt="PyPOTS visits" align="left" src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FPyPOTS%2FPyPOTS&count_bg=%23009A0A&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visits%20since%20May%202022&edge_flat=false">
</a>
</details>
<br>
