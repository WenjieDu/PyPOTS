<a href='https://github.com/WenjieDu/PyPOTS'><img src='https://raw.githubusercontent.com/WenjieDu/PyPOTS/main/docs/figs/PyPOTS%20logo.svg?sanitize=true' width='200' align='right' /></a>

## <p align='center'>Welcome to PyPOTS</p>
**<p align='center'>A Python Toolbox for Data Mining on Partially-Observed Time Series</p>**

<p align='center'>
    <!-- Python version -->
    <img src='https://img.shields.io/badge/python-v3-yellow?color=a4e2c6'>
    <!-- PyTorch-->
    <img src='https://img.shields.io/static/v1?label=PyTorch&message=%E2%9D%A4%EF%B8%8F&color=7bcfa6&logo=pytorch'>
    <!-- PyPI version -->
    <a alt='PyPI download number' href='https://pypi.org/project/pypots'>
        <img alt="PyPI" src="https://img.shields.io/pypi/v/pypots?color=7fecad&label=PyPI&logo=pypi&logoColor=white">
    </a>
    <!-- on Anaconda -->
    <a alt='on anaconda' href='https://anaconda.org/conda-forge/pypots'>
        <img alt="on anaconda" src="https://img.shields.io/conda/pn/conda-forge/pypots?color=3de1ad&label=Conda&logo=anaconda" />
    </a>
    <!-- License -->
    <a alt='GPL3 license' href='https://github.com/WenjieDu/PyPOTS/blob/main/LICENSE'>
        <img src='https://img.shields.io/badge/License-GPL--v3-00e09e'>
    </a>
    <!-- Repo size -->
    <img src="https://img.shields.io/github/repo-size/WenjieDu/PyPOTS?color=48c0a3">
    <!-- Code of Conduct -->
    <a alt='CODE_OF_CONDUCT' href='https://github.com/WenjieDu/PyPOTS/blob/main/CODE_OF_CONDUCT.md'> 
        <img src='https://img.shields.io/badge/Contributor%20Covenant-v2.1-21a675.svg'>
    </a>
    <!-- Slack Workspace -->
    <a alt='Slack Workspace' href='https://join.slack.com/t/pypots-dev/shared_invite/zt-1gq6ufwsi-p0OZdW~e9UW_IA4_f1OfxA'> 
        <img src='https://img.shields.io/badge/Slack-PyPOTS-grey?logo=slack&color=549688'>
    </a>
    <!-- Zenodo DOI -->
    <a alt='Zenodo DOI' href='https://doi.org/10.5281/zenodo.6823221'>
        <img src='https://zenodo.org/badge/DOI/10.5281/zenodo.6823221.svg'>
    </a>
    <!-- PyPI download number -->
    <a alt='PyPI download number' href='https://pepy.tech/project/pypots'>
        <img src='https://static.pepy.tech/personalized-badge/pypots?period=total&units=international_system&left_color=grey&right_color=navy&left_text=Downloads'>
    </a>
    <!-- GitHub Testing -->
    <a alt='GitHub Testing' href='https://github.com/WenjieDu/PyPOTS/actions/workflows/testing.yml'> 
        <img src='https://github.com/WenjieDu/PyPOTS/actions/workflows/testing.yml/badge.svg'>
    </a>
    <!-- Coveralls report -->
    <a alt='Coveralls report' href='https://coveralls.io/github/WenjieDu/PyPOTS'> 
        <img src='https://img.shields.io/coverallsCoverage/github/WenjieDu/PyPOTS?branch=main&logo=coveralls&labelColor=#0aa344'>
    </a>
</p>

⦿ `Motivation`: Due to all kinds of reasons like failure of collection sensors, communication error, and unexpected malfunction, missing values are common to see in time series from the real-world environment. This makes partially-observed time series (POTS) a pervasive problem in open-world modeling and prevents advanced data analysis. Although this problem is important, the area of data mining on POTS still lacks a dedicated toolkit. PyPOTS is created to fill in this blank.

⦿ `Mission`: PyPOTS is born to become a handy toolbox that is going to make data mining on POTS easy rather than tedious, to help engineers and researchers focus more on the core problems in their hands rather than on how to deal with the missing parts in their data. PyPOTS will keep integrating classical and the latest state-of-the-art data mining algorithms for partially-observed multivariate time series. For sure, besides various algorithms, PyPOTS is going to have unified APIs together with detailed documentation and interactive examples across algorithms as tutorials.

<a href='https://github.com/WenjieDu/TSDB'><img src="https://raw.githubusercontent.com/WenjieDu/TSDB/main/docs/figs/TSDB%20logo.svg?sanitize=true" align='left' width='190'/></a>
To make various open-source time-series datasets readily available to our users, PyPOTS gets supported by project [TSDB (Time-Series DataBase)](https://github.com/WenjieDu/TSDB), a toolbox making loading time-series datasets super easy! 

Visit [TSDB](https://github.com/WenjieDu/TSDB) right now to know more about this handy tool 🛠! It now supports a total of 119 open-source datasets.
<br clear='left'>

## ❖ Installation
PyPOTS now is available on <a alt='Anaconda' href='https://anaconda.org/conda-forge/pypots'><img align='center' src='https://img.shields.io/badge/Anaconda--lightgreen?style=social&logo=anaconda'></a>❗️ 

Install it with `conda install pypots`, you may need to specify the channel with option `-c conda-forge`

Install the latest release from PyPI:
> pip install pypots

or install from the source code with the latest features not officially released in a version:
> pip install `https://github.com/WenjieDu/PyPOTS/archive/main.zip`

<details open>
<summary><b>Below is an example applying SAITS in PyPOTS to impute missing values in the dataset PhysioNet2012:</b></summary>

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
X = X.drop('RecordID', axis = 1)
X = StandardScaler().fit_transform(X.to_numpy())
X = X.reshape(num_samples, 48, -1)
X_intact, X, missing_mask, indicating_mask = mcar(X, 0.1) # hold out 10% observed values as ground truth
X = masked_fill(X, 1 - missing_mask, np.nan)
dataset = {"X": X}
# Model training. This is PyPOTS showtime. 💪
saits = SAITS(n_steps=48, n_features=37, n_layers=2, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=10)
saits.fit(dataset)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
imputation = saits.impute(dataset)  # impute the originally-missing values and artificially-missing values
mae = cal_mae(imputation, X_intact, indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
```
</details>

## ❖ Available Algorithms
| Task                          | Type           | Algorithm                                                                | Year | Reference |        
|-------------------------------|----------------|--------------------------------------------------------------------------|------|-----------|
| Imputation                    | Neural Network | SAITS (Self-Attention-based Imputation for Time Series)                  | 2023 | [^1]      |
| Imputation                    | Neural Network | Transformer                                                              | 2017 | [^1] [^2] |
| Imputation,<br>Classification | Neural Network | BRITS (Bidirectional Recurrent Imputation for Time Series)               | 2018 | [^3]      |
| Imputation                    | Naive          | LOCF (Last Observation Carried Forward)                                  | -    | -         |
| Classification                | Neural Network | GRU-D                                                                    | 2018 | [^4]      |
| Classification                | Neural Network | Raindrop                                                                 | 2022 | [^5]      |
| Clustering                    | Neural Network | CRLI (Clustering Representation Learning on Incomplete time-series data) | 2021 | [^6]      |
| Clustering                    | Neural Network | VaDER (Variational Deep Embedding with Recurrence)                       | 2019 | [^7]      |
| Forecasting                   | Probabilistic  | BTTF (Bayesian Temporal Tensor Factorization)                            | 2021 | [^8]      |

## ❖ Reference
If you find PyPOTS is helpful to your research, please cite it as below and ⭐️star this repository to make others notice this work. 🤗

```bibtex
@misc{du2022PyPOTS,
author = {Wenjie Du},
title = {{PyPOTS: A Python Toolbox for Data Mining on Partially-Observed Time Series}},
howpublished = {\url{https://github.com/wenjiedu/pypots}},
year = {2022},
doi = {10.5281/zenodo.6823221},
}
```

or

`Wenjie Du. (2022). PyPOTS: A Python Toolbox for Data Mining on Partially-Observed Time Series. Zenodo. https://doi.org/10.5281/zenodo.6823221`

## ❖ Attention 👀
The documentation and tutorials are under construction. And a short paper introducing PyPOTS is on the way! 🚀 Stay tuned please!

‼️ PyPOTS is currently under developing. If you like it and look forward to its growth, <ins>please give PyPOTS a star and watch it to keep you posted on its progress and to let me know that its development is meaningful</ins>. If you have any feedback, or want to contribute ideas/suggestions or share time-series related algorithms/papers, please join PyPOTS community and chat on <a alt='Slack Workspace' href='https://join.slack.com/t/pypots-dev/shared_invite/zt-1gq6ufwsi-p0OZdW~e9UW_IA4_f1OfxA'><img align='center' src='https://img.shields.io/badge/Slack-PyPOTS-grey?logo=slack&labelColor=4A154B&color=62BCE5'></a>, or create an issue. If you have any additional questions or have interests in collaboration, please take a look at [my GitHub profile](https://github.com/WenjieDu) and feel free to contact me 🤝.

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
<img align='left' src='https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FPyPOTS%2FPyPOTS&count_bg=%23009A0A&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Hits&edge_flat=false'>
</details>

