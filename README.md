<a href="https://github.com/WenjieDu/PyPOTS">
    <img src="https://pypots.com/figs/pypots_logos/PyPOTS/logo_FFBG.svg" width="210" align="right">
</a>

<h3 align="center">Welcome to PyPOTS</h3>

<p align="center"><i>a Python toolbox for machine learning on Partially-Observed Time Series</i></p>

<p align="center">
    <a href="https://docs.pypots.com/en/latest/install.html#reasons-of-version-limitations-on-dependencies">
       <img alt="Python version" src="https://img.shields.io/badge/Python-v3.8+-F8C6B5?logo=python&logoColor=white">
    </a>
    <a href="https://landscape.pytorch.org/?item=modeling--specialized--pypots">
        <img alt="Pytorch landscape" src="https://img.shields.io/badge/PyTorch%20Landscape-EE4C2C?logo=pytorch&logoColor=white">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/blob/main/LICENSE">
        <img alt="BSD-3 license" src="https://img.shields.io/badge/License-BSD--3-E9BB41?logo=opensourceinitiative&logoColor=white">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS?tab=coc-ov-file">
        <img alt="Code of Conduct" src="https://img.shields.io/badge/Contributor_Covenant-2.1-4baaaa">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS#-community">
        <img alt="Community" src="https://img.shields.io/badge/join_us-community!-C8A062">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/releases">
        <img alt="the latest release version" src="https://img.shields.io/github/v/release/wenjiedu/pypots?color=EE781F&include_prereleases&label=Release&logo=github&logoColor=white">
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
    <a href="https://sonarcloud.io/component_measures?id=WenjieDu_PyPOTS&metric=sqale_rating&view=list">
        <img alt="maintainability" src="https://sonarcloud.io/api/project_badges/measure?project=WenjieDu_PyPOTS&metric=sqale_rating">
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
    <a href="https://github.com/psf/black">
        <img alt="Code Style" src="https://img.shields.io/badge/Code_Style-black-000000">
    </a>
    <a href="https://anaconda.org/conda-forge/pypots">
        <img alt="Conda downloads" src="https://pypots.com/figs/downloads_badges/conda_pypots_downloads.svg">
    </a>
    <a href="https://pepy.tech/project/pypots">
        <img alt="PyPI downloads" src="https://pypots.com/figs/downloads_badges/pypi_pypots_downloads.svg">
    </a>
    <a href="https://arxiv.org/abs/2305.18811">
        <img alt="arXiv DOI" src="https://img.shields.io/badge/DOI-10.48550/arXiv.2305.18811-F8F7F0">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/blob/main/README_zh.md">
        <img alt="README in Chinese" src="https://pypots.com/figs/pypots_logos/readme/CN.svg">
    </a>
   <a href="https://github.com/WenjieDu/PyPOTS/blob/main/README.md">
        <img alt="README in English" src="https://pypots.com/figs/pypots_logos/readme/US.svg">
    </a>
</p>

⦿ `Motivation`: Due to all kinds of reasons like failure of collection sensors, communication error,
and unexpected malfunction, missing values are common to see in time series from the real-world environment.
This makes partially-observed time series (POTS) a pervasive problem in open-world modeling and prevents advanced
data analysis. Although this problem is important, the area of machine learning on POTS still lacks a dedicated toolkit.
PyPOTS is created to fill in this blank.

⦿ `Mission`: PyPOTS (pronounced "Pie Pots") is born to become a handy toolbox that is going to make machine learning on
POTS easy rather than tedious, to help engineers and researchers focus more on the core problems in their hands rather
than on how to deal with the missing parts in their data. PyPOTS will keep integrating classical and the latest
state-of-the-art machine learning algorithms for partially-observed multivariate time series. For sure, besides various
algorithms, PyPOTS is going to have unified APIs together with detailed documentation and interactive examples across
algorithms as tutorials.

🤗 **Please** star this repo to help others notice PyPOTS if you think it is a useful toolkit.
**Please** kindly [cite PyPOTS](https://github.com/WenjieDu/PyPOTS#-citing-pypots) in your publications if it helps with
your research.
This really means a lot to our open-source research. Thank you!

The rest of this readme file is organized as follows:
[**❖ Available Algorithms**](#-available-algorithms),
[**❖ PyPOTS Ecosystem**](#-pypots-ecosystem),
[**❖ Installation**](#-installation),
[**❖ Usage**](#-usage),
[**❖ Citing PyPOTS**](#-citing-pypots),
[**❖ Contribution**](#-contribution),
[**❖ Community**](#-community).

## ❖ Available Algorithms

PyPOTS supports imputation, classification, clustering, forecasting, and anomaly detection tasks on multivariate
partially-observed time series with missing values. The table below shows the availability of each algorithm
(sorted by Year) in PyPOTS for different tasks. The symbol `✅` indicates the algorithm is available for the
corresponding task (note that models will be continuously updated in the future to handle tasks that are not
currently supported. Stay tuned❗️).

🌟 Since **v0.2**, all neural-network models in PyPOTS has got hyperparameter-optimization support.
This functionality is implemented with the [Microsoft NNI](https://github.com/microsoft/nni) framework. You may want to
refer to our time-series imputation survey and benchmark
repo [Awesome_Imputation](https://github.com/WenjieDu/Awesome_Imputation)
to see how to config and tune the hyperparameters.

🔥 Note that all models whose name with `🧑‍🔧` in the table (e.g. Transformer, iTransformer, Informer etc.) are not
originally proposed as algorithms for POTS data in their papers, and they cannot directly accept time series with
missing values as input, let alone imputation. **To make them applicable to POTS data, we specifically apply the
embedding strategy and training approach (ORT+MIT) the same as we did in
[the SAITS paper](https://arxiv.org/pdf/2202.08516)[^1].**

The task types are abbreviated as follows:
**`IMPU`**: Imputation;
**`FORE`**: Forecasting;
**`CLAS`**: Classification;
**`CLUS`**: Clustering;
**`ANOD`**: Anomaly Detection.
In addition to the 5 tasks, PyPOTS also provides TS2Vec[^48] for time series representation learning and vectorization.
The paper references and links are all listed at the bottom of this file.

| **Type**      | **Algo**                                                                                                                         | **IMPU** | **FORE** | **CLAS** | **CLUS** | **ANOD** | **Year - Venue**                                   |
|:--------------|:---------------------------------------------------------------------------------------------------------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:---------------------------------------------------|
| LLM&TSFM      | <a href="https://time-series.ai"><img src="https://time-series.ai/static/figs/robot.svg" width="26px"> Time-Series.AI</a>  [^36] |    ✅     |    ✅     |    ✅     |    ✅     |    ✅     | <a href="https://time-series.ai">Join waitlist</a> |
| Neural Net    | TimeMixer++[^49]                                                                                                                 |    ✅     |          |          |          |    ✅     | `2025 - ICLR`                                      |
| LLM           | Time-LLM🧑‍🔧[^45]                                                                                                               |    ✅     |    ✅     |          |          |          | `2024 - ICLR`                                      |
| TSFM          | MOMENT[^47]                                                                                                                      |    ✅     |    ✅     |          |          |          | `2024 - ICML`                                      |
| Neural Net    | TSLANet[^51]                                                                                                                     |    ✅     |          |          |          |          | `2024 - ICML`                                      |
| Neural Net    | TEFN🧑‍🔧[^39]                                                                                                                   |    ✅     |    ✅     |    ✅     |          |    ✅     | `2024 - arXiv`                                     |
| Neural Net    | FITS🧑‍🔧[^41]                                                                                                                   |    ✅     |    ✅     |          |          |          | `2024 - ICLR`                                      |
| Neural Net    | TimeMixer[^37]                                                                                                                   |    ✅     |    ✅     |          |          |     ✅     | `2024 - ICLR`                                      |
| Neural Net    | iTransformer🧑‍🔧[^24]                                                                                                           |    ✅     |          |    ✅     |          |    ✅      | `2024 - ICLR`                                      |
| Neural Net    | ModernTCN[^38]                                                                                                                   |    ✅     |    ✅     |          |          |          | `2024 - ICLR`                                      |
| Neural Net    | ImputeFormer🧑‍🔧[^34]                                                                                                           |    ✅     |          |          |          |    ✅     | `2024 - KDD`                                       |
| Neural Net    | TOTEM [^50]                                                                                                                      |    ✅     |          |          |          |          | `2024 - TMLR`                                      |
| Neural Net    | SAITS[^1]                                                                                                                        |    ✅     |          |    ✅     |          |    ✅     | `2023 - ESWA`                                      |
| LLM           | GPT4TS[^46]                                                                                                                      |    ✅     |    ✅     |          |          |          | `2023 - NeurIPS`                                   |
| Neural Net    | FreTS🧑‍🔧[^23]                                                                                                                  |    ✅     |          |          |          |          | `2023 - NeurIPS`                                   |
| Neural Net    | Koopa🧑‍🔧[^29]                                                                                                                  |    ✅     |          |          |          |          | `2023 - NeurIPS`                                   |
| Neural Net    | Crossformer🧑‍🔧[^16]                                                                                                            |    ✅     |          |          |          |     ✅     | `2023 - ICLR`                                      |
| Neural Net    | TimesNet[^14]                                                                                                                    |    ✅     |    ✅     |    ✅     |          |    ✅     | `2023 - ICLR`                                      |
| Neural Net    | PatchTST🧑‍🔧[^18]                                                                                                               |    ✅     |          |    ✅     |          |    ✅     | `2023 - ICLR`                                      |
| Neural Net    | ETSformer🧑‍🔧[^19]                                                                                                              |    ✅     |          |          |          |     ✅     | `2023 - ICLR`                                      |
| Neural Net    | MICN🧑‍🔧[^27]                                                                                                                   |    ✅     |    ✅     |          |          |          | `2023 - ICLR`                                      |
| Neural Net    | DLinear🧑‍🔧[^17]                                                                                                                |    ✅     |    ✅     |          |          |    ✅     | `2023 - AAAI`                                      |
| Neural Net    | TiDE🧑‍🔧[^28]                                                                                                                   |    ✅     |          |          |          |          | `2023 - TMLR`                                      |
| Neural Net    | CSAI[^42]                                                                                                                        |    ✅     |          |    ✅     |          |          | `2023 - arXiv`                                     |
| Neural Net    | SegRNN🧑‍🔧[^43]                                                                                                                 |    ✅     |    ✅     |          |          |    ✅     | `2023 - arXiv`                                     |
| Neural Net    | TS2Vec[^48]                                                                                                                      |          |          |    ✅     |          |          | `2022 - AAAI`                                      |
| Neural Net    | SCINet🧑‍🔧[^30]                                                                                                                 |    ✅     |          |          |          |    ✅     | `2022 - NeurIPS`                                   |
| Neural Net    | Nonstationary Tr.🧑‍🔧[^25]                                                                                                      |    ✅     |          |          |          |     ✅     | `2022 - NeurIPS`                                   |
| Neural Net    | FiLM🧑‍🔧[^22]                                                                                                                   |    ✅     |    ✅     |          |          |     ✅     | `2022 - NeurIPS`                                   |
| Neural Net    | RevIN_SCINet🧑‍🔧[^31]                                                                                                           |    ✅     |          |          |          |          | `2022 - ICLR`                                      |
| Neural Net    | Pyraformer🧑‍🔧[^26]                                                                                                             |    ✅     |          |          |          |     ✅     | `2022 - ICLR`                                      |
| Neural Net    | Raindrop[^5]                                                                                                                     |          |          |    ✅     |          |          | `2022 - ICLR`                                      |
| Neural Net    | FEDformer🧑‍🔧[^20]                                                                                                              |    ✅     |          |          |          |     ✅     | `2022 - ICML`                                      |
| Neural Net    | Autoformer🧑‍🔧[^15]                                                                                                             |    ✅     |          |    ✅     |          |    ✅     | `2021 - NeurIPS`                                   |
| Neural Net    | CSDI[^12]                                                                                                                        |    ✅     |    ✅     |          |          |          | `2021 - NeurIPS`                                   |
| Neural Net    | Informer🧑‍🔧[^21]                                                                                                               |    ✅     |          |          |          |    ✅      | `2021 - AAAI`                                      |
| Neural Net    | US-GAN[^10]                                                                                                                      |    ✅     |          |          |          |          | `2021 - AAAI`                                      |
| Neural Net    | CRLI[^6]                                                                                                                         |          |          |          |    ✅     |          | `2021 - AAAI`                                      |
| Probabilistic | BTTF[^8]                                                                                                                         |          |    ✅     |          |          |          | `2021 - TPAMI`                                     |
| Neural Net    | StemGNN🧑‍🔧[^33]                                                                                                                |    ✅     |          |          |          |          | `2020 - NeurIPS`                                   |
| Neural Net    | Reformer🧑‍🔧[^32]                                                                                                               |    ✅     |          |          |          |    ✅     | `2020 - ICLR`                                      |
| Neural Net    | GP-VAE[^11]                                                                                                                      |    ✅     |          |          |          |          | `2020 - AISTATS`                                   |
| Neural Net    | VaDER[^7]                                                                                                                        |          |          |          |    ✅     |          | `2019 - GigaSci.`                                  |
| Neural Net    | M-RNN[^9]                                                                                                                        |    ✅     |          |          |          |          | `2019 - TBME`                                      |
| Neural Net    | BRITS[^3]                                                                                                                        |    ✅     |          |    ✅     |          |          | `2018 - NeurIPS`                                   |
| Neural Net    | GRU-D[^4]                                                                                                                        |    ✅     |          |    ✅     |          |          | `2018 - Sci. Rep.`                                 |
| Neural Net    | TCN🧑‍🔧[^35]                                                                                                                    |    ✅     |          |          |          |          | `2018 - arXiv`                                     |
| Neural Net    | Transformer🧑‍🔧[^2]                                                                                                             |    ✅     |    ✅     |          |          |     ✅     | `2017 - NeurIPS`                                   |
| MF            | TRMF[^44]                                                                                                                        |    ✅     |          |          |          |          | `2016 - NeurIPS`                                   |
| Naive         | Lerp[^40]                                                                                                                        |    ✅     |          |          |          |          |                                                    |
| Naive         | LOCF/NOCB                                                                                                                        |    ✅     |          |          |          |          |                                                    |
| Naive         | Mean                                                                                                                             |    ✅     |          |          |          |          |                                                    |
| Naive         | Median                                                                                                                           |    ✅     |          |          |          |          |                                                    |

🙋 Differences between `LLM (Large Language Model)` and `TSFM (Time-Series Foundation Model)` in the above table:
`LLM` refers to the models that are pre-trained on large-scale text data and can be fine-tuned for specific tasks.
`TSFM` refers to the models that are pre-trained on large-scale time series data, inspired by recent achievements
of foundation models in CV and NLP.

💯 Contribute your model right now to increase your research impact! PyPOTS downloads are increasing rapidly
(**[1M+ in total and 2k+ daily on PyPI so far](https://www.pepy.tech/projects/pypots)**),
and your work will be widely used and cited by the community.
Refer to the [contribution guide](https://github.com/WenjieDu/PyPOTS#-contribution) to see how to include your model in
PyPOTS.

## ❖ PyPOTS Ecosystem

At PyPOTS, things are related to coffee, which we're familiar with. Yes, this is a coffee universe!
As you can see, there is a coffee pot in the PyPOTS logo. And what else? Please read on ;-)

<a href="https://github.com/WenjieDu/TSDB">
    <img src="https://pypots.com/figs/pypots_logos/TSDB/logo_FFBG.svg" align="left" width="140" alt="TSDB logo"/>
</a>

👈 Time series datasets are taken as coffee beans at PyPOTS, and POTS datasets are incomplete coffee beans with missing
parts that have their own meanings. To make various public time-series datasets readily available to users,
<i>Time Series Data Beans (TSDB)</i> is created to make loading time-series datasets super easy!
Visit [TSDB](https://github.com/WenjieDu/TSDB) right now to know more about this handy tool 🛠, and it now supports a
total of 172 open-source datasets!

<a href="https://github.com/WenjieDu/PyGrinder">
    <img src="https://pypots.com/figs/pypots_logos/PyGrinder/logo_FFBG.svg" align="right" width="140" alt="PyGrinder logo"/>
</a>

👉 To simulate the real-world data beans with missingness, the ecosystem library
[PyGrinder](https://github.com/WenjieDu/PyGrinder), a toolkit helping grind your coffee beans into incomplete ones, is
created. Missing patterns fall into three categories according to Robin's theory[^13]:
MCAR (missing completely at random), MAR (missing at random), and MNAR (missing not at random).
PyGrinder supports all of them and additional functionalities related to missingness.
With PyGrinder, you can introduce synthetic missing values into your datasets with a single line of code.

<a href="https://github.com/WenjieDu/BenchPOTS">
    <img src="https://pypots.com/figs/pypots_logos/BenchPOTS/logo_FFBG.svg" align="left" width="140" alt="BenchPOTS logo"/>
</a>

👈 To fairly evaluate the performance of PyPOTS algorithms, the benchmarking suite
[BenchPOTS](https://github.com/WenjieDu/BenchPOTS) is created, which provides standard and unified data-preprocessing
pipelines to prepare datasets for measuring the performance of different POTS algorithms on various tasks.

<a href="https://github.com/WenjieDu/BrewPOTS">
    <img src="https://pypots.com/figs/pypots_logos/BrewPOTS/logo_FFBG.svg" align="right" width="140" alt="BrewPOTS logo"/>
</a>

👉 Now the beans, grinder, and pot are ready, please have a seat on the bench and let's think about how to brew us a cup
of coffee. Tutorials are necessary! Considering the future workload, PyPOTS tutorials are released in a single repo,
and you can find them in [BrewPOTS](https://github.com/WenjieDu/BrewPOTS).
Take a look at it now, and learn how to brew your POTS datasets!

<p align="center">
<a href="https://pypots.com/ecosystem/">
    <img src="https://pypots.com/figs/pypots_logos/Ecosystem/PyPOTS_Ecosystem_Pipeline.png" width="95%"/>
</a>
<br>
<b> ☕️ Welcome to the universe of PyPOTS. Enjoy it and have fun!</b>
</p>

## ❖ Installation

You can refer to [the installation instruction](https://docs.pypots.com/en/latest/install.html) in PyPOTS documentation
for a guideline with more details.

PyPOTS is available on both [PyPI](https://pypi.python.org/pypi/pypots)
and [Anaconda](https://anaconda.org/conda-forge/pypots).
You can install PyPOTS like below as well as
[TSDB](https://github.com/WenjieDu/TSDB),[PyGrinder](https://github.com/WenjieDu/PyGrinder),
[BenchPOTS](https://github.com/WenjieDu/BenchPOTS), and [AI4TS](https://github.com/WenjieDu/AI4TS):

``` bash
# via pip
pip install pypots            # the first time installation
pip install pypots --upgrade  # update pypots to the latest version
# install from the latest source code with the latest features but may be not officially released yet
pip install https://github.com/WenjieDu/PyPOTS/archive/main.zip

# via conda
conda install conda-forge::pypots  # the first time installation
conda update  conda-forge::pypots  # update pypots to the latest version

# via docker
docker run -it --name pypots wenjiedu/pypots  # docker will auto pull our built image and run a instance for you
# after things settled, you can run python in the container to access the well-configured environment for running pypots
# if you'd like to detach from the container, press ctrl-P + ctrl-Q 
# run `docker attach pypots` to enter the container again. 
```

## ❖ Usage

Besides [BrewPOTS](https://github.com/WenjieDu/BrewPOTS), you can also find a simple and quick-start tutorial notebook
on Google Colab
<a href="https://colab.research.google.com/drive/1HEFjylEy05-r47jRy0H9jiS_WhD0UWmQ">
<img src="https://img.shields.io/badge/GoogleColab-PyPOTS_Tutorials-F9AB00?logo=googlecolab&logoColor=white" alt="Colab tutorials" align="center"/>
</a>. If you have further questions, please refer to PyPOTS documentation [docs.pypots.com](https://docs.pypots.com).
You can also [raise an issue](https://github.com/WenjieDu/PyPOTS/issues) or [ask in our community](#-community).

We present you a usage example of imputing missing values in time series with PyPOTS below, you can click it to view.

<details open>
<summary><b>Click here to see an example applying SAITS on PhysioNet2012 for imputation:</b></summary>

``` python
import numpy as np
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar, calc_missing_rate
from benchpots.datasets import preprocess_physionet2012
data = preprocess_physionet2012(subset='set-a',rate=0.1) # Our ecosystem libs will automatically download and extract it
train_X, val_X, test_X = data["train_X"], data["val_X"], data["test_X"]
print(train_X.shape)  # (n_samples, n_steps, n_features)
print(val_X.shape)  # samples (n_samples) in train set and val set are different, but they have the same sequence len (n_steps) and feature dim (n_features)
print(f"We have {calc_missing_rate(train_X):.1%} values missing in train_X")  
train_set = {"X": train_X}  # in training set, simply put the incomplete time series into it
val_set = {
    "X": val_X,
    "X_ori": data["val_X_ori"],  # in validation set, we need ground truth for evaluation and picking the best model checkpoint
}
test_set = {"X": test_X}  # in test set, only give the testing incomplete time series for model to impute
test_X_ori = data["test_X_ori"]  # test_X_ori bears ground truth for evaluation
indicating_mask = np.isnan(test_X) ^ np.isnan(test_X_ori)  # mask indicates the values that are missing in X but not in X_ori, i.e. where the gt values are 

from pypots.imputation import SAITS  # import the model you want to use
from pypots.nn.functional import calc_mae
saits = SAITS(n_steps=train_X.shape[1], n_features=train_X.shape[2], n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64, d_ffn=128, dropout=0.1, epochs=5)
saits.fit(train_set, val_set)  # train the model on the dataset
imputation = saits.impute(test_set)  # impute the originally-missing values and artificially-missing values
mae = calc_mae(imputation, np.nan_to_num(test_X_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
saits.save("save_it_here/saits_physionet2012.pypots")  # save the model for future use
saits.load("save_it_here/saits_physionet2012.pypots")  # reload the serialized model file for following imputation or training
```

</details>

## ❖ Citing PyPOTS

> [!TIP]
> **[Updates in Jun 2024]** 😎 The 1st comprehensive time-seres imputation benchmark paper
[TSI-Bench: Benchmarking Time Series Imputation](https://arxiv.org/abs/2406.12747) now is public available.
> The code is open source in the repo [Awesome_Imputation](https://github.com/WenjieDu/Awesome_Imputation).
> With nearly 35,000 experiments, we provide a comprehensive benchmarking study on 28 imputation methods, 3 missing
> patterns (points, sequences, blocks),
> various missing rates, and 8 real-world datasets.
>
> **[Updates in Feb 2024]** 🎉 Our survey
> paper [Deep Learning for Multivariate Time Series Imputation: A Survey](https://arxiv.org/abs/2402.04059) has been
> released on arXiv.
> We comprehensively review the literature of the state-of-the-art deep-learning imputation methods for time series,
> provide a taxonomy for them, and discuss the challenges and future directions in this field.

The paper introducing PyPOTS is available [on arXiv](https://arxiv.org/abs/2305.18811),
and a short version of it is accepted by the 9th SIGKDD international workshop on Mining and Learning from Time
Series ([MiLeTS'23](https://kdd-milets.github.io/milets2023/))).
**Additionally**, PyPOTS has been included as a [PyTorch Ecosystem](https://landscape.pytorch.org/?item=modeling--specialized--pypots) project.
We are pursuing to publish it in prestigious academic venues, e.g. JMLR (track for
[Machine Learning Open Source Software](https://www.jmlr.org/mloss/)). If you use PyPOTS in your work,
please cite it as below and 🌟star this repository to make others notice this library. 🤗

There are scientific research projects using PyPOTS and referencing in their papers.
Here is [an incomplete list of them](https://scholar.google.com/scholar?as_ylo=2022&q=%E2%80%9CPyPOTS%E2%80%9D&hl=en).

```bibtex
@article{du2023pypots,
    title = {{PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series}},
    author = {Wenjie Du},
    journal = {arXiv preprint arXiv:2305.18811},
    year = {2023},
}
```

or
> Wenjie Du.
> PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series.
> arXiv, abs/2305.18811, 2023.

## ❖ Contribution

You're very welcome to contribute to this exciting project!

By committing your code, you'll

1. make your well-established model out-of-the-box for PyPOTS users to run,
   and help your work obtain more exposure and impact.
   Take a look at our [inclusion criteria](https://docs.pypots.com/en/latest/faq.html#inclusion-criteria).
   You can utilize the `template` folder in each task package (e.g.
   [pypots/imputation/template](https://github.com/WenjieDu/PyPOTS/tree/main/pypots/imputation/template)) to quickly
   start;
2. become one of [PyPOTS contributors](https://github.com/WenjieDu/PyPOTS/graphs/contributors) and
   be listed as a volunteer developer [on the PyPOTS website](https://pypots.com/about/#volunteer-developers);
3. get mentioned in PyPOTS [release notes](https://github.com/WenjieDu/PyPOTS/releases);

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

- [Slack](https://join.slack.com/t/pypots-org/shared_invite/zt-1gq6ufwsi-p0OZdW~e9UW_IA4_f1OfxA). General discussion,
  Q&A, and our development team are here;
- [LinkedIn](https://www.linkedin.com/company/pypots). Official announcements and news are here;
- [WeChat (微信公众号)](https://mp.weixin.qq.com/s/X3ukIgL1QpNH8ZEXq1YifA). We also run a group chat on WeChat,
  and you can get the QR code from the official account after following it;

If you have any suggestions or want to contribute ideas or share time-series related papers, join us and tell.
PyPOTS community is open, transparent, and surely friendly. Let's work together to build and improve PyPOTS!


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
[Graph-Guided Network for Irregularly Sampled Multivariate Time Series](https://arxiv.org/abs/2110.05357).
*ICLR 2022*.
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
<a href="https://time-series.ai"><img src="http://time-series.ai/static/figs/robot.svg" width="20px">
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
[^48]: Yue, Z., Wang, Y., Duan, J., Yang, T., Huang, C., Tong, Y., & Xu, B. (2022).
[TS2Vec: Towards Universal Representation of Time Series](https://ojs.aaai.org/index.php/AAAI/article/view/20881).
*AAAI 2022*.
[^49]: Wang, S., Li, J., Shi, X., Ye, Z., Mo, B., Lin, W., Ju, S., Chu, Z. & Jin, M. (2025).
[TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis](https://openreview.net/forum?id=1CLzLXSFNn).
*ICLR 2025*.
[^50]: Talukder, S., Yue, Y., & Gkioxari, G. (2024).
[TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis](https://openreview.net/forum?id=QlTLkH6xRC).
*TMLR 2024*.
[^51]: Eldele, E., Ragab, M., Chen, Z., Wu, M., & Li, X. (2024).
[TSLANet: Rethinking Transformers for Time Series Representation Learning](https://proceedings.mlr.press/v235/eldele24a.html).
*ICML 2024*.