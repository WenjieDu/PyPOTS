<a href="https://github.com/WenjieDu/PyPOTS">
    <img src="https://raw.githubusercontent.com/WenjieDu/PyPOTS/main/docs/_static/figs/PyPOTS_logo.svg?sanitize=true" width="200" align="right">
</a>

## <p align="center">Welcome to PyPOTS</p>
**<p align="center">A Python Toolbox for Data Mining on Partially-Observed Time Series</p>**

<p align="center">
    <img alt="Python version" src="https://img.shields.io/badge/Python-v3.7--3.10-88ada6?logo=python&logoColor=white">
    <img alt="powered by Pytorch" src="https://img.shields.io/badge/PyTorch-❤️-bbcdc5?logo=pytorch&logoColor=white">
    <a href="https://pypi.org/project/">
        <img alt="the latest release version" src="https://img.shields.io/github/v/release/wenjiedu/pypots?color=e0eee8&include_prereleases&label=Release">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/blob/main/LICENSE">
        <img alt="GPL-v3 license" src="https://img.shields.io/badge/License-GPL--v3-c0ebd7">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/blob/main/README.md#-community">
        <img alt="Community" src="https://img.shields.io/badge/join_us-community!-7fecad">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/graphs/contributors">
        <img alt="GitHub contributors" src="https://img.shields.io/github/contributors/wenjiedu/pypots?color=7bcfa6&label=Contributors&logo=GitHub">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/stargazers">
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/wenjiedu/pypots?logo=Github&color=7bcfa6&label=Stars">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/forks">
        <img alt="GitHub Repo forks" src="https://img.shields.io/github/forks/wenjiedu/pypots?logo=Github&color=2edfa3&label=Forks">
    </a>
    <a href="https://codeclimate.com/github/WenjieDu/PyPOTS">
        <img alt="Code Climate maintainability" src="https://img.shields.io/codeclimate/maintainability-percentage/WenjieDu/PyPOTS?color=25f8cb&label=Maintainability&logo=codeclimate">
    </a>
    <a href="https://coveralls.io/github/WenjieDu/PyPOTS">
        <img alt="Coveralls coverage" src="https://img.shields.io/coverallsCoverage/github/WenjieDu/PyPOTS?branch=main&logo=coveralls&color=00e09e&label=Coverage">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/actions/workflows/testing_ci.yml">
        <img alt="GitHub Testing" src="https://img.shields.io/github/actions/workflow/status/wenjiedu/pypots/testing_ci.yml?logo=github&color=48c0a3&label=CI">
    </a>
    <a href="https://doi.org/10.5281/zenodo.6823221">
        <img alt="Zenodo DOI" src="https://img.shields.io/badge/DOI-10.5281/zenodo.6823221-21a675">
    </a>
    <a href="https://anaconda.org/conda-forge/pypots">
        <img alt="Conda downloads" src="https://img.shields.io/conda/dn/conda-forge/pypots?label=Conda%20Downloads&color=057748&logo=anaconda&logoColor=white">
    </a>
    <a href="https://pypi.org/project/pypots">
        <img alt="PyPI downloads" src="https://static.pepy.tech/personalized-badge/pypots?period=total&units=international_system&left_color=grey&right_color=teal&left_text=PyPI%20Downloads&logo=github">
    </a>

</p>

⦿ `Motivation`: Due to all kinds of reasons like failure of collection sensors, communication error,
and unexpected malfunction, missing values are common to see in time series from the real-world environment.
This makes partially-observed time series (POTS) a pervasive problem in open-world modeling and prevents advanced
data analysis. Although this problem is important, the area of data mining on POTS still lacks a dedicated toolkit.
PyPOTS is created to fill in this blank.

⦿ `Mission`: PyPOTS is born to become a handy toolbox that is going to make data mining on POTS easy rather than
tedious, to help engineers and researchers focus more on the core problems in their hands rather than on how to deal
with the missing parts in their data. PyPOTS will keep integrating classical and the latest state-of-the-art data mining
algorithms for partially-observed multivariate time series. For sure, besides various algorithms, PyPOTS is going to
have unified APIs together with detailed documentation and interactive examples across algorithms as tutorials.

<a href="https://github.com/WenjieDu/TSDB">
    <img src="https://raw.githubusercontent.com/WenjieDu/TSDB/main/docs/_static/figs/TSDB_logo.svg?sanitize=true" align="left" width="160" alt="TSDB logo"/>
</a>

To make various open-source time-series datasets readily available to our users,
PyPOTS gets supported by project [TSDB (Time-Series Data Base)](https://github.com/WenjieDu/TSDB),
a toolbox making loading time-series datasets super easy!

Visit [TSDB](https://github.com/WenjieDu/TSDB) right now to know more about this handy tool 🛠!
It now supports a total of 119 open-source datasets.
<br clear="left">

The rest of this readme file is organized as follows:
[**❖ Installation**](#-installation),
[**❖ Usage**](#-usage),
[**❖ Available Algorithms**](#-available-algorithms),
[**❖ Citing PyPOTS**](#-citing-pypots),
[**❖ Contribution**](#-contribution),
[**❖ Community**](#-community).


## ❖ Installation
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
<a href="https://github.com/WenjieDu/BrewPOTS">
    <img src="https://raw.githubusercontent.com/WenjieDu/BrewPOTS/main/figs/BrewPOTS_logo.jpg" align="left" width="160" alt="BrewPOTS logo"/>
</a>

PyPOTS tutorials have been released. Considering the future workload, I separate the tutorials into a single repo,
and you can find them in [BrewPOTS](https://github.com/WenjieDu/BrewPOTS).
Take a look at it now, and brew your POTS dataset into a cup of coffee! 🤓

If you have further questions, please refer to PyPOTS documentation 📑[docs.pypots.com](https://docs.pypots.com).
Besides, you can also [raise an issue](https://github.com/WenjieDu/PyPOTS/issues) or [ask in our community](#-community).

We present you a usage example of imputing missing values in time series with PyPOTS below, you can click it to view.

<details>
<summary><b>Click here to see an example applying SAITS on PhysioNet2012 for imputation:</b></summary>

``` python
import numpy as np
from sklearn.preprocessing import StandardScaler
from pypots.data import load_specific_dataset, mcar, masked_fill
from pypots.imputation import SAITS
from pypots.utils.metrics import cal_mae
# Data preprocessing. Tedious, but PyPOTS can help. 🤓
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
# Model training. This is PyPOTS showtime. 💪
saits = SAITS(n_steps=48, n_features=37, n_layers=2, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=10)
saits.fit(dataset)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
imputation = saits.impute(dataset)  # impute the originally-missing values and artificially-missing values
mae = cal_mae(imputation, X_intact, indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
```
</details>


## ❖ Available Algorithms
PyPOTS supports imputation, classification, clustering, and forecasting tasks on multivariate time series with missing values. The currently available algorithms of four tasks are cataloged in the following table with four partitions. The paper references are all listed at the bottom of this readme file. Please refer to them if you want more details.

|   ***`Imputation`***   |      🚥      |                                                                                        🚥                                                                                         |    🚥    |
|:----------------------:|:------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------:|
|        **Type**        |  **Abbr.**   |                                                                    **Full name of the algorithm/model/paper**                                                                     | **Year** |
|       Neural Net       |    SAITS     |                                                               Self-Attention-based Imputation for Time Series [^1]                                                                |   2023   |
|       Neural Net       | Transformer  | Attention is All you Need [^2];<br>Self-Attention-based Imputation for Time Series [^1];<br><sub>Note: proposed in [^2], and re-implemented as an imputation model in [^1].</sub> |   2017   |
|       Neural Net       |    BRITS     |                                                              Bidirectional Recurrent Imputation for Time Series [^3]                                                              |   2018   |
|         Naive          |     LOCF     |                                                                         Last Observation Carried Forward                                                                          |    -     |
| ***`Classification`*** |      🚥      |                                                                                        🚥                                                                                         |    🚥    |
|        **Type**        |  **Abbr.**   |                                                                    **Full name of the algorithm/model/paper**                                                                     | **Year** |
|       Neural Net       |    BRITS     |                                                              Bidirectional Recurrent Imputation for Time Series [^3]                                                              |   2018   |
|       Neural Net       |    GRU-D     |                                                  Recurrent Neural Networks for Multivariate Time Series with Missing Values [^4]                                                  |   2018   |
|       Neural Net       |   Raindrop   |                                                    Graph-Guided Network for Irregularly Sampled Multivariate Time Series [^5]                                                     |   2022   |
|   ***`Clustering`***   |      🚥      |                                                                                        🚥                                                                                         |    🚥    |
|        **Type**        |  **Abbr.**   |                                                                    **Full name of the algorithm/model/paper**                                                                     | **Year** |
|       Neural Net       |     CRLI     |                                                      Clustering Representation Learning on Incomplete time-series data [^6]                                                       |   2021   |
|       Neural Net       |    VaDER     |                                                                  Variational Deep Embedding with Recurrence [^7]                                                                  |   2019   |
|  ***`Forecasting`***   |      🚥      |                                                                                        🚥                                                                                         |    🚥    |
|        **Type**        |  **Abbr.**   |                                                                    **Full name of the algorithm/model/paper**                                                                     | **Year** |
|     Probabilistic      |     BTTF     |                                                                    Bayesian Temporal Tensor Factorization [^8]                                                                    |   2021   |


## ❖ Citing PyPOTS
We are pursuing to publish a short paper introducing PyPOTS in prestigious academic venues, e.g. JMLR (track for
[Machine Learning Open Source Software](https://www.jmlr.org/mloss/)). Before that, PyPOTS is using its DOI from Zenodo
for reference. If you use PyPOTS in your research, please cite it as below and 🌟star this repository to make others
notice this work. 🤗

``` bibtex
@software{Du2022PyPOTS,
author = {Wenjie Du},
title = {{PyPOTS: A Python Toolbox for Data Mining on Partially-Observed Time Series}},
year = {2022},
howpublished = {\url{https://github.com/WenjieDu/PyPOTS}},
url = {\url{https://github.com/WenjieDu/PyPOTS}},
doi = {10.5281/zenodo.6823221},
}
```

or

> Wenjie Du. (2022).
> PyPOTS: A Python Toolbox for Data Mining on Partially-Observed Time Series.
> Zenodo. https://doi.org/10.5281/zenodo.6823221


## ❖ Contribution
You're very welcome to contribute to this exciting project!

By committing your code, you'll

1. make your well-established model out-of-the-box for PyPOTS users to run,
   and help your work obtain more exposure and impact.
   Take a look at our [inclusion criteria](https://docs.pypots.com/en/latest/faq.html#inclusion-criteria);
2. be listed as one of [PyPOTS contributors](https://github.com/WenjieDu/PyPOTS/graphs/contributors):
   <img align="center" src="https://contrib.rocks/image?repo=wenjiedu/pypots">;
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
    <img alt="PyPOTS stargazers" src="https://reporoster.com/stars/dark/WenjieDu/PyPOTS">
</a>
<br>
<a href="https://github.com/WenjieDu/PyPOTS/network/members">
    <img alt="PyPOTS forkers" src="https://reporoster.com/forks/dark/WenjieDu/PyPOTS">
</a>
</details>


## ❖ Community
We care about the feedback from our users, so we're building PyPOTS community on

- [Slack](https://pypots-dev.slack.com). General discussion, Q&A, and our development team are here;
- [LinkedIn](https://www.linkedin.com/company/pypots). Official announcements and news are here;
- [WeChat (微信公众号)](https://mp.weixin.qq.com/s/m6j83SJNgz-xySSZd-DTBw). We also run a group chat on WeChat,
  and you can get the QR code from the official account after following it;

If you have any suggestions or want to contribute ideas or share time-series related papers, join us and tell.
PyPOTS community is open, transparent, and surely friendly. Let's work together to build and improve PyPOTS 💪!


## ❖ Attention 👀
‼️ PyPOTS is currently under developing. If you like it and look forward to its growth, <ins>please give PyPOTS a star
and watch it to keep you posted on its progress and to let me know that its development is meaningful</ins>.
If you have any additional questions or have interests in collaboration, please take a look at
[my GitHub profile](https://github.com/WenjieDu) and feel free to contact me 🤝.

Thank you all for your attention! 😃


[^1]: Du, W., Cote, D., & Liu, Y. (2023). [SAITS: Self-Attention-based Imputation for Time Series](https://doi.org/10.1016/j.eswa.2023.119619). *Expert systems with applications*.
[^2]: Vaswani, A., Shazeer, N.M., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., & Polosukhin, I. (2017). [Attention is All you Need](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html). *NeurIPS 2017*.
[^3]: Cao, W., Wang, D., Li, J., Zhou, H., Li, L., & Li, Y. (2018). [BRITS: Bidirectional Recurrent Imputation for Time Series](https://papers.nips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html). *NeurIPS 2018*.
[^4]: Che, Z., Purushotham, S., Cho, K., Sontag, D.A., & Liu, Y. (2018). [Recurrent Neural Networks for Multivariate Time Series with Missing Values](https://www.nature.com/articles/s41598-018-24271-9). *Scientific Reports*.
[^5]: Zhang, X., Zeman, M., Tsiligkaridis, T., & Zitnik, M. (2022). [Graph-Guided Network for Irregularly Sampled Multivariate Time Series](https://arxiv.org/abs/2110.05357). *ICLR 2022*.
[^6]: Ma, Q., Chen, C., Li, S., & Cottrell, G. W. (2021). [Learning Representations for Incomplete Time Series Clustering](https://ojs.aaai.org/index.php/AAAI/article/view/17070). *AAAI 2021*.
[^7]: Jong, J.D., Emon, M.A., Wu, P., Karki, R., Sood, M., Godard, P., Ahmad, A., Vrooman, H.A., Hofmann-Apitius, M., & Fröhlich, H. (2019). [Deep learning for clustering of multivariate clinical patient trajectories with missing values](https://academic.oup.com/gigascience/article/8/11/giz134/5626377). *GigaScience*.
[^8]: Chen, X., & Sun, L. (2021). [Bayesian Temporal Factorization for Multidimensional Time Series Prediction](https://arxiv.org/abs/1910.06366). *IEEE transactions on pattern analysis and machine intelligence*.

<details>
<summary>🏠 Visits</summary>
<a href="https://github.com/WenjieDu/PyPOTS">
    <img alt="PyPOTS visits" align="left" src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FPyPOTS%2FPyPOTS&count_bg=%23009A0A&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Hits&edge_flat=false">
</a>
</details>
