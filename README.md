<a href="https://github.com/WenjieDu/PyPOTS">
    <img src="https://pypots.com/figs/pypots_logos/PyPOTS_logo_FFBG.svg?sanitize=true" width="200" align="right">
</a>

<h2 align="center">Welcome to PyPOTS</h2>

<p align="center"><i>a Python toolbox for machine learning on Partially-Observed Time Series</i></p>

<p align="center">
    <a href="https://docs.pypots.com/en/latest/install.html#reasons-of-version-limitations-on-dependencies">
       <img alt="Python version" src="https://img.shields.io/badge/Python-v3.7--3.10-E97040?logo=python&logoColor=white">
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
    <a href="https://github.com/WenjieDu/PyPOTS/blob/main/README.md#-community">
        <img alt="Community" src="https://img.shields.io/badge/join_us-community!-C8A062">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/graphs/contributors">
        <img alt="GitHub contributors" src="https://img.shields.io/github/contributors/wenjiedu/pypots?color=D8E699&label=Contributors&logo=GitHub">
    </a>
    <a href="https://star-history.com/#wenjiedu/pypots">
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/wenjiedu/pypots?logo=Github&color=6BB392&label=Stars">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/network/members">
        <img alt="GitHub Repo forks" src="https://img.shields.io/github/forks/wenjiedu/pypots?logo=Github&color=91B821&label=Forks">
    </a>
    <a href="https://codeclimate.com/github/WenjieDu/PyPOTS">
        <img alt="Code Climate maintainability" src="https://img.shields.io/codeclimate/maintainability-percentage/WenjieDu/PyPOTS?color=3C7699&label=Maintainability&logo=codeclimate">
    </a>
    <a href="https://coveralls.io/github/WenjieDu/PyPOTS">
        <img alt="Coveralls coverage" src="https://img.shields.io/coverallsCoverage/github/WenjieDu/PyPOTS?branch=main&logo=coveralls&color=75C1C4&label=Coverage">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/actions/workflows/testing_ci.yml">
        <img alt="GitHub Testing" src="https://img.shields.io/github/actions/workflow/status/wenjiedu/pypots/testing_ci.yml?logo=github&color=C8D8E1&label=CI">
    </a>
    <a href="https://arxiv.org/abs/2305.18811">
        <img alt="arXiv DOI" src="https://img.shields.io/badge/DOI-10.48550/arXiv.2305.18811-F8F7F0">
    </a>
    <a href="https://anaconda.org/conda-forge/pypots">
        <img alt="Conda downloads" src="https://img.shields.io/endpoint?url=https://pypots.com/figs/downloads_badges/conda_pypots_downloads.json">
    </a>
    <a href="https://pepy.tech/project/pypots">
        <img alt="PyPI downloads" src="https://img.shields.io/endpoint?url=https://pypots.com/figs/downloads_badges/pypi_pypots_downloads.json">
    </a>

</p>

⦿ `Motivation`: Due to all kinds of reasons like failure of collection sensors, communication error,
and unexpected malfunction, missing values are common to see in time series from the real-world environment.
This makes partially-observed time series (POTS) a pervasive problem in open-world modeling and prevents advanced
data analysis. Although this problem is important, the area of machine learning on POTS still lacks a dedicated toolkit.
PyPOTS is created to fill in this blank.

⦿ `Mission`: PyPOTS (pronounced "Pie Pots") is born to become a handy toolbox that is going to make machine learning on POTS easy rather than
tedious, to help engineers and researchers focus more on the core problems in their hands rather than on how to deal
with the missing parts in their data. PyPOTS will keep integrating classical and the latest state-of-the-art machine learning
algorithms for partially-observed multivariate time series. For sure, besides various algorithms, PyPOTS is going to
have unified APIs together with detailed documentation and interactive examples across algorithms as tutorials.

🤗 **Please** star this repo to help others notice PyPOTS if you think it is a useful toolkit.
**Please** properly [cite PyPOTS](https://github.com/WenjieDu/PyPOTS#-citing-pypots) in your publications
if it helps with your research. This really means a lot to our open-source research. Thank you!

The rest of this readme file is organized as follows:
[**❖ PyPOTS Ecosystem**](#-pypots-ecosystem),
[**❖ Installation**](#-installation),
[**❖ Usage**](#-usage),
[**❖ Available Algorithms**](#-available-algorithms),
[**❖ Citing PyPOTS**](#-citing-pypots),
[**❖ Contribution**](#-contribution),
[**❖ Community**](#-community).


## ❖ PyPOTS Ecosystem
At PyPOTS, things are related to coffee, which we're familiar with. Yes, this is a coffee universe!
As you can see, there is a coffee pot in the PyPOTS logo.
And what else? Please read on ;-)

<a href="https://github.com/WenjieDu/TSDB">
    <img src="https://pypots.com/figs/pypots_logos/TSDB_logo_FFBG.svg" align="left" width="140" alt="TSDB logo"/>
</a>

👈 Time series datasets are taken as coffee beans at PyPOTS, and POTS datasets are incomplete coffee beans with missing parts that have their own meanings.
To make various public time-series datasets readily available to users,
<i>Time Series Data Beans (TSDB)</i> is created to make loading time-series datasets super easy!
Visit [TSDB](https://github.com/WenjieDu/TSDB) right now to know more about this handy tool 🛠, and it now supports a total of 168 open-source datasets!

<a href="https://github.com/WenjieDu/PyGrinder">
    <img src="https://pypots.com/figs/pypots_logos/PyGrinder_logo_FFBG.svg" align="right" width="140" alt="PyGrinder logo"/>
</a>

👉 To simulate the real-world data beans with missingness, the ecosystem library [PyGrinder](https://github.com/WenjieDu/PyGrinder),
a toolkit helping grind your coffee beans into incomplete ones, is created. Missing patterns fall into three categories according to Robin's theory[^13]:
MCAR (missing completely at random), MAR (missing at random), and MNAR (missing not at random).
PyGrinder supports all of them and additional functionalities related to missingness.
With PyGrinder, you can introduce synthetic missing values into your datasets with a single line of code.

<a href="https://github.com/WenjieDu/BrewPOTS">
    <img src="https://pypots.com/figs/pypots_logos/BrewPOTS_logo_FFBG.svg" align="left" width="140" alt="BrewPOTS logo"/>
</a>

👈 Now we have the beans, the grinder, and the pot, how to brew us a cup of coffee? Tutorials are necessary!
Considering the future workload, PyPOTS tutorials are released in a single repo,
and you can find them in [BrewPOTS](https://github.com/WenjieDu/BrewPOTS).
Take a look at it now, and learn how to brew your POTS datasets!

☕️ Welcome to the universe of PyPOTS. Enjoy it and have fun!


## ❖ Installation
You can refer to [the installation instruction](https://docs.pypots.com/en/latest/install.html) in PyPOTS documentation for a guideline with more details.

PyPOTS is available on both [PyPI](https://pypi.python.org/pypi/pypots) and [Anaconda](https://anaconda.org/conda-forge/pypots).
You can install PyPOTS as shown below:

``` bash
# by pip
pip install pypots            # the first time installation
pip install pypots --upgrade  # update pypots to the latest version

# by conda
conda install -c conda-forge pypots  # the first time installation
conda update  -c conda-forge pypots  # update pypots to the latest version
````

Alternatively, you can install from the latest source code with the latest features but may be not officially released yet:
> pip install https://github.com/WenjieDu/PyPOTS/archive/main.zip


## ❖ Usage
Besides [BrewPOTS](https://github.com/WenjieDu/BrewPOTS), you can also find a simple and quick-start tutorial notebook
on Google Colab with [this link](https://colab.research.google.com/drive/1HEFjylEy05-r47jRy0H9jiS_WhD0UWmQ?usp=sharing).
If you have further questions, please refer to PyPOTS documentation [docs.pypots.com](https://docs.pypots.com).
You can also [raise an issue](https://github.com/WenjieDu/PyPOTS/issues) or [ask in our community](#-community).

We present you a usage example of imputing missing values in time series with PyPOTS below, you can click it to view.

<details open>
<summary><b>Click here to see an example applying SAITS on PhysioNet2012 for imputation:</b></summary>

``` python
import numpy as np
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar, masked_fill
from pypots.data import load_specific_dataset
from pypots.imputation import SAITS
from pypots.utils.metrics import cal_mae
# Data preprocessing. Tedious, but PyPOTS can help.
data = load_specific_dataset('physionet_2012')  # PyPOTS will automatically download and extract it.
X = data['X']
num_samples = len(X['RecordID'].unique())
X = X.drop(['RecordID', 'Time'], axis = 1)
X = StandardScaler().fit_transform(X.to_numpy())
X = X.reshape(num_samples, 48, -1)
X_intact, X, missing_mask, indicating_mask = mcar(X, 0.1) # hold out 10% observed values as ground truth
X = masked_fill(X, 1 - missing_mask, np.nan)
dataset = {"X": X}
print(dataset["X"].shape)  # (11988, 48, 37), 11988 samples, 48 time steps, 37 features
# Model training. This is PyPOTS showtime.
saits = SAITS(n_steps=48, n_features=37, n_layers=2, d_model=256, d_inner=128, n_heads=4, d_k=64, d_v=64, dropout=0.1, epochs=10)
# Here I use the whole dataset as the training set because ground truth is not visible to the model, you can also split it into train/val/test sets
saits.fit(dataset)
imputation = saits.impute(dataset)  # impute the originally-missing values and artificially-missing values
mae = cal_mae(imputation, X_intact, indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
```
</details>


## ❖ Available Algorithms
PyPOTS supports imputation, classification, clustering, and forecasting tasks on multivariate time series with missing values. The currently available algorithms of four tasks are cataloged in the following table with four partitions. The paper references are all listed at the bottom of this readme file. Please refer to them if you want more details.

|   ***`Imputation`***   |     🚥      |                                                                                        🚥                                                                                         |    🚥    |
|:----------------------:|:-----------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------:|
|        **Type**        |  **Abbr.**  |                                                                       **Full name of the algorithm/model**                                                                        | **Year** |
|       Neural Net       |    SAITS    |                                                               Self-Attention-based Imputation for Time Series [^1]                                                                |   2023   |
|       Neural Net       | Transformer | Attention is All you Need [^2];<br>Self-Attention-based Imputation for Time Series [^1];<br><sub>Note: proposed in [^2], and re-implemented as an imputation model in [^1].</sub> |   2017   |
|       Neural Net       |    CSDI     |                                              Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation [^12]                                              |   2021   |
|       Neural Net       |   US-GAN    |                                                          Unsupervised GAN for Multivariate Time Series Imputation [^10]                                                           |   2021   |
|       Neural Net       |   GP-VAE    |                                                                   Gaussian Process Variational Autoencoder [^11]                                                                  |   2020   |
|       Neural Net       |    BRITS    |                                                              Bidirectional Recurrent Imputation for Time Series [^3]                                                              |   2018   |
|       Neural Net       |    M-RNN    |                                                                  Multi-directional Recurrent Neural Network [^9]                                                                  |   2019   |
|         Naive          |    LOCF     |                                                                         Last Observation Carried Forward                                                                          |    -     |
| ***`Classification`*** |     🚥      |                                                                                        🚥                                                                                         |    🚥    |
|        **Type**        |  **Abbr.**  |                                                                    **Full name of the algorithm/model/paper**                                                                     | **Year** |
|       Neural Net       |    BRITS    |                                                              Bidirectional Recurrent Imputation for Time Series [^3]                                                              |   2018   |
|       Neural Net       |    GRU-D    |                                                  Recurrent Neural Networks for Multivariate Time Series with Missing Values [^4]                                                  |   2018   |
|       Neural Net       |  Raindrop   |                                                    Graph-Guided Network for Irregularly Sampled Multivariate Time Series [^5]                                                     |   2022   |
|   ***`Clustering`***   |     🚥      |                                                                                        🚥                                                                                         |    🚥    |
|        **Type**        |  **Abbr.**  |                                                                    **Full name of the algorithm/model/paper**                                                                     | **Year** |
|       Neural Net       |    CRLI     |                                                      Clustering Representation Learning on Incomplete time-series data [^6]                                                       |   2021   |
|       Neural Net       |    VaDER    |                                                                  Variational Deep Embedding with Recurrence [^7]                                                                  |   2019   |
|  ***`Forecasting`***   |     🚥      |                                                                                        🚥                                                                                         |    🚥    |
|        **Type**        |  **Abbr.**  |                                                                    **Full name of the algorithm/model/paper**                                                                     | **Year** |
|     Probabilistic      |    BTTF     |                                                                    Bayesian Temporal Tensor Factorization [^8]                                                                    |   2021   |


## ❖ Citing PyPOTS
**[Updates in Jun 2023]** 🎉A short version of the PyPOTS paper is accepted by the 9th SIGKDD international workshop on
Mining and Learning from Time Series ([MiLeTS'23](https://kdd-milets.github.io/milets2023/))).
Besides, PyPOTS has been included as a [PyTorch Ecosystem](https://pytorch.org/ecosystem/) project.

The paper introducing PyPOTS is available on arXiv at [this URL](https://arxiv.org/abs/2305.18811),
and we are pursuing to publish it in prestigious academic venues, e.g. JMLR (track for
[Machine Learning Open Source Software](https://www.jmlr.org/mloss/)). If you use PyPOTS in your work,
please cite it as below and 🌟star this repository to make others notice this library. 🤗

There are scientific research projects using PyPOTS and referencing in their papers.
Here is [an incomplete list of them](https://scholar.google.com/scholar?as_ylo=2022&q=%E2%80%9CPyPOTS%E2%80%9D&hl=en>).

``` bibtex
@article{du2023PyPOTS,
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

> Wenjie Du. (2023).
> PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series.
> arXiv, abs/2305.18811.https://arxiv.org/abs/2305.18811

or

``` bibtex
@inproceedings{du2023PyPOTS,
title={{PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series}},
booktitle={9th SIGKDD workshop on Mining and Learning from Time Series (MiLeTS'23)},
author={Wenjie Du},
year={2023},
url={https://arxiv.org/abs/2305.18811},
}
```

> Wenjie Du. (2023).
> PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series.
> In *9th SIGKDD workshop on Mining and Learning from Time Series (MiLeTS'23)*. https://arxiv.org/abs/2305.18811


## ❖ Contribution
You're very welcome to contribute to this exciting project!

By committing your code, you'll

1. make your well-established model out-of-the-box for PyPOTS users to run,
   and help your work obtain more exposure and impact.
   Take a look at our [inclusion criteria](https://docs.pypots.com/en/latest/faq.html#inclusion-criteria).
   You can utilize the `template` folder in each task package (e.g.
   [pypots/imputation/template](https://github.com/WenjieDu/PyPOTS/tree/main/pypots/imputation/template)) to quickly start;
2. be listed as one of [PyPOTS contributors](https://pypots.com/about/#all-contributors);
3. get mentioned in our [release notes](https://github.com/WenjieDu/PyPOTS/releases);

You can also contribute to PyPOTS by simply staring🌟 this repo to help more people notice it.
Your star is your recognition to PyPOTS, and it matters!

<details open>
<summary>
    <b><i>
    👏 Click here to view PyPOTS stargazers and forkers.<br>
    We're so proud to have more and more awesome users, as well as more bright ✨stars:
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

👀 Check out a full list of our users' affiliations [on PyPOTS website here](https://pypots.com/users/)!


## ❖ Community
We care about the feedback from our users, so we're building PyPOTS community on

- [Slack](https://join.slack.com/t/pypots-org/shared_invite/zt-1gq6ufwsi-p0OZdW~e9UW_IA4_f1OfxA). General discussion, Q&A, and our development team are here;
- [LinkedIn](https://www.linkedin.com/company/pypots). Official announcements and news are here;
- [WeChat (微信公众号)](https://mp.weixin.qq.com/s/sNgZmgAyxDn2sZxXoWJYMA). We also run a group chat on WeChat,
  and you can get the QR code from the official account after following it;

If you have any suggestions or want to contribute ideas or share time-series related papers, join us and tell.
PyPOTS community is open, transparent, and surely friendly. Let's work together to build and improve PyPOTS!


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


<details>
<summary>🏠 Visits</summary>
<a href="https://github.com/WenjieDu/PyPOTS">
    <img alt="PyPOTS visits" align="left" src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FPyPOTS%2FPyPOTS&count_bg=%23009A0A&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visits%20since%20May%202022&edge_flat=false">
</a>
</details>
<br>
