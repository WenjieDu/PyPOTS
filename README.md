<img src="https://pypots.com/figs/icons/translate.svg" width="18"> English | [简体中文](https://github.com/WenjieDu/PyPOTS/blob/main/i18n/README_zh.md) | [日本語](https://github.com/WenjieDu/PyPOTS/blob/main/i18n/README_ja.md) | [한국어](https://github.com/WenjieDu/PyPOTS/blob/main/i18n/README_ko.md) | [Deutsch](https://github.com/WenjieDu/PyPOTS/blob/main/i18n/README_de.md) | [Français](https://github.com/WenjieDu/PyPOTS/blob/main/i18n/README_fr.md) | [Español](https://github.com/WenjieDu/PyPOTS/blob/main/i18n/README_es.md) | [Português (Brasil)](https://github.com/WenjieDu/PyPOTS/blob/main/i18n/README_pt-BR.md) | [Русский](https://github.com/WenjieDu/PyPOTS/blob/main/i18n/README_ru.md) | [Bahasa Indonesia](https://github.com/WenjieDu/PyPOTS/blob/main/i18n/README_id.md)

#  PyPOTS <a href="https://github.com/WenjieDu/PyPOTS"><img src="https://pypots.com/figs/pypots_logos/PyPOTS/logo_FFBG.svg" width="200" align='right'></a>

> A Python Toolbox for Unified Analysis of Real-World Time Series

<p>
    <a href="https://docs.pypots.com/en/latest/install.html#reasons-of-version-limitations-on-dependencies">
       <img alt="Python version" src="https://img.shields.io/badge/Python-v3.10+-F8C6B5?logo=python&logoColor=white">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/releases">
        <img alt="the latest release version" src="https://img.shields.io/github/v/release/wenjiedu/pypots?color=EE781F&include_prereleases&label=Release&logo=github&logoColor=white">
    </a>
    <a href="https://docs.pypots.com">
        <img alt="Docs building" src="https://img.shields.io/readthedocs/pypots?logo=readthedocs&label=Docs&logoColor=white&color=395260">
    </a>
    <a href="https://pepy.tech/project/pypots">
        <img alt="PyPI downloads" src="https://pypots.com/figs/downloads_badges/pypi_pypots_downloads.svg">
    </a>
    <a href="https://anaconda.org/conda-forge/pypots">
        <img alt="Conda downloads" src="https://pypots.com/figs/downloads_badges/conda_pypots_downloads.svg">
    </a>
    <a href="https://landscape.pytorch.org/?item=modeling--specialized--pypots">
        <img alt="Pytorch landscape" src="https://img.shields.io/badge/PyTorch%20Landscape-EE4C2C?logo=pytorch&logoColor=white">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/blob/main/LICENSE">
        <img alt="BSD-3 license" src="https://img.shields.io/badge/License-BSD--3-E9BB41?logo=opensourceinitiative&logoColor=white">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS#-community">
        <img alt="Community" src="https://img.shields.io/badge/join_us-community!-C8A062">
    </a>
    <a href="https://star-history.com/#wenjiedu/pypots">
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/wenjiedu/pypots?logo=None&color=6BB392&label=%E2%98%85%20Stars">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/network/members">
        <img alt="GitHub Repo forks" src="https://img.shields.io/github/forks/wenjiedu/pypots?logo=forgejo&logoColor=black&label=Forks">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/graphs/contributors">
        <img alt="GitHub contributors" src="https://img.shields.io/github/contributors/wenjiedu/pypots?color=D8E699&label=Contributors&logo=GitHub">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/actions/workflows/testing_ci.yml">
        <img alt="GitHub Testing" src="https://img.shields.io/github/actions/workflow/status/wenjiedu/pypots/testing_ci.yml?logo=circleci&color=C8D8E1&label=CI">
    </a>
    <a href="https://coveralls.io/github/WenjieDu/PyPOTS?branch=full_test">
        <img alt="Coveralls coverage" src="https://img.shields.io/coverallsCoverage/github/WenjieDu/PyPOTS?branch=full_test&logo=coveralls&color=75C1C4&label=Coverage">
    </a>
    <a href="https://sonarcloud.io/component_measures?id=WenjieDu_PyPOTS&metric=sqale_rating&view=list">
        <img alt="maintainability" src="https://sonarcloud.io/api/project_badges/measure?project=WenjieDu_PyPOTS&metric=sqale_rating">
    </a>
    <a href="https://github.com/astral-sh/ruff">
        <img alt="Code Style" src="https://img.shields.io/badge/Code_Style-Ruff-D7FF64?logo=ruff">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS?tab=coc-ov-file">
        <img alt="Code of Conduct" src="https://img.shields.io/badge/Contributor_Covenant-2.1-4baaaa">
    </a>
    <a href="https://deepwiki.com/WenjieDu/PyPOTS">
        <img alt="Ask DeepWiki" src="https://pypots.com/figs/misc/ask_deepwiki.svg">
    </a>
    <a href="https://arxiv.org/abs/2305.18811">
        <img alt="arXiv DOI" src="https://img.shields.io/badge/DOI-10.48550/arXiv.2305.18811-F8F7F0">
    </a>
</p>

PyPOTS (pronounced "Pie Pots") is an open-source Python toolbox for time series analysis, offering
**50+ models and methods with unified APIs** for imputation, forecasting, classification, clustering, anomaly detection,
and representation learning. It is designed for real-world time series, including those with missing values.

A time series is not the underlying system itself, but a collection of observations of that system.
Due to the inherent limitations of measurement and sampling, real-world time series are fundamentally partial
observations: measured variables capture only certain aspects of the underlying system, observations are collected
at discrete and potentially irregular intervals, and some measurements may be missing.

**Being unobserved does not necessarily mean being nonexistent. Partial observation is the norm, not the exception.**

We conceptualize real-world time series as **Partially-Observed Time Series (POTS)**. From this perspective,
time series modeling is fundamentally about learning the underlying system and its dynamics from partial observations.
PyPOTS is built upon this philosophy to enable unified time series analysis for real-world data.

[Website](https://pypots.com) · [Documentation](https://docs.pypots.com) · [Tutorials](https://github.com/WenjieDu/BrewPOTS) · [Community](https://github.com/WenjieDu/PyPOTS#-community)

[**Key Features**](https://github.com/WenjieDu/PyPOTS#-key-features) · [**Installation**](https://github.com/WenjieDu/PyPOTS#-installation) · [**Usage**](https://github.com/WenjieDu/PyPOTS#-usage) · [**Model Zoo**](https://github.com/WenjieDu/PyPOTS#-model-zoo) · [**PyPOTS Ecosystem**](https://github.com/WenjieDu/PyPOTS#-pypots-ecosystem) · [**Citing PyPOTS**](https://github.com/WenjieDu/PyPOTS#-citing-pypots) · [**Contribution**](https://github.com/WenjieDu/PyPOTS#-contribution)

## ❖ Key Features

- **Multi-task support:** Imputation, forecasting, classification, clustering, anomaly detection, and representation learning in one toolbox.
- **Native missing-value support:** Missingness-aware models can learn directly from incomplete time series without prior imputation.
- **Unified APIs:** Consistent training and prediction interfaces across models, making experimentation and model comparison easier.
- **Rich model zoo:** 50+ models and methods spanning classical approaches, deep learning, LLM-based models, and time series foundation models (TSFMs).
- **<a href="https://landscape.pytorch.org/?item=modeling--specialized--pypots"><img src="https://pypots.com/figs/misc/pytorch.svg" width="26px" align="center">PyTorch Ecosystem</a>:** PyPOTS is built on PyTorch and featured in the PyTorch Landscape.
- **Hardware acceleration:** CPU and NVIDIA GPU support, including multi-GPU training via PyTorch DataParallel and automatic mixed precision for supported models.
- **Flexible data handling:** In-memory data and lazy loading from HDF5 files.
- **Hyperparameter optimization:** Model tuning powered by [Optuna](https://github.com/optuna/optuna).
- **Reproducible research:** Datasets, missingness simulation, standardized preprocessing, and evaluation tools across the PyPOTS ecosystem.
- **Extensible architecture:** Modular model components and standardized templates for integrating new algorithms.

## ❖ Installation

PyPOTS requires Python 3.10 or later and is available on [PyPI](https://pypi.org/project/pypots/)
and [Anaconda](https://anaconda.org/conda-forge/pypots). See the
[installation guide](https://docs.pypots.com/en/latest/install.html) for detailed dependency and environment requirements.

Choose **one** of the following installation methods according to your environment.
Upgrade commands are only needed if you already have PyPOTS installed.

**Option 1: pip (recommended)**

```bash
pip install pypots
# To upgrade: pip install pypots --upgrade
```

**Option 2: Conda**

```bash
conda install conda-forge::pypots
# To upgrade: conda update conda-forge::pypots
```

**Option 3: Docker**

```bash
docker run -it --name pypots wenjiedu/pypots
```

Press Ctrl-P followed by Ctrl-Q to detach without stopping the container.
To reconnect, run `docker attach pypots`.

**Option 4: Latest source code on main (may include unreleased changes)**

```bash
pip install https://github.com/WenjieDu/PyPOTS/archive/main.zip
```

## ❖ Usage

### Quick Start

This self-contained SAITS imputation example uses synthetic data and requires no dataset download. Inputs have shape `(n_samples, n_steps, n_features)`, with missing values represented by `NaN`.

```python
import numpy as np
from pypots.imputation import SAITS

# Generate synthetic time series with missing values
rng = np.random.default_rng(42)
X = rng.normal(size=(16, 24, 8)).astype(np.float32)
X[rng.random(X.shape) < 0.1] = np.nan
train_X, test_X = X[:12], X[12:]

# Initialize a SAITS imputation model
model = SAITS(
    n_steps=24, n_features=8,
    n_layers=2, d_model=64, n_heads=4,
    d_k=16, d_v=16, d_ffn=128,
    epochs=2, device="cpu",
)
# Train the model
model.fit({"X": train_X})

# Impute the test set
imputed_X = model.impute({"X": test_X})
print(imputed_X.shape)  # (4, 24, 8)
```

### Explore PyPOTS by Task

| Task | What it does | Resources |
|:--|:--|:--|
| Imputation | Estimate missing observations | [Examples](https://github.com/WenjieDu/PyPOTS/tree/main/examples/imputation) · [API](https://docs.pypots.com/en/latest/pypots.imputation.html) |
| Forecasting | Predict future values | [Examples](https://github.com/WenjieDu/PyPOTS/tree/main/examples/forecasting) · [API](https://docs.pypots.com/en/latest/pypots.forecasting.html) |
| Classification | Assign labels to time series | [Examples](https://github.com/WenjieDu/PyPOTS/tree/main/examples/classification) · [API](https://docs.pypots.com/en/latest/pypots.classification.html) |
| Clustering | Discover groups of similar time series | [Examples](https://github.com/WenjieDu/PyPOTS/tree/main/examples/clustering) · [API](https://docs.pypots.com/en/latest/pypots.clustering.html) |
| Anomaly detection | Identify abnormal observations or patterns | [Examples](https://github.com/WenjieDu/PyPOTS/tree/main/examples/anomaly_detection) · [API](https://docs.pypots.com/en/latest/pypots.anomaly_detection.html) |
| Representation learning | Learn latent representations of time series | [Examples](https://github.com/WenjieDu/PyPOTS/tree/main/examples/representation) · [API](https://docs.pypots.com/en/latest/pypots.representation.html) |

### Tutorials and a Real-World Example

Besides [BrewPOTS](https://github.com/WenjieDu/BrewPOTS), you can also find a simple and quick-start tutorial notebook
on Google Colab
<a href="https://colab.research.google.com/drive/1HEFjylEy05-r47jRy0H9jiS_WhD0UWmQ">
<img src="https://img.shields.io/badge/GoogleColab-PyPOTS_Tutorials-F9AB00?logo=googlecolab&logoColor=white" alt="Colab tutorials"/>
</a>. If you have further questions, please refer to PyPOTS documentation [docs.pypots.com](https://docs.pypots.com).
You can also [raise an issue](https://github.com/WenjieDu/PyPOTS/issues) or [ask in our community](https://github.com/WenjieDu/PyPOTS#-community).

For a comprehensive hands-on introduction to PyPOTS, explore our
[KDD 2026 Tutorial: End-to-End Learning for Partially-Observed Time Series with PyPOTS](https://pypots.com/events/KDD2026Tutorial/).

The PhysioNet2012 example below demonstrates training, validation, testing, and imputation evaluation.

<details>
<summary><b>Click here to see an example applying SAITS on PhysioNet2012 for imputation:</b></summary>

``` python
import numpy as np
from pygrinder import calc_missing_rate
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

<a name="-available-algorithms"></a>

## ❖ Model Zoo

The table below lists the models and methods implemented in PyPOTS, sorted by publication year, alongside the Time-Series.AI waitlist entry.
Each linked ✅ leads to a standalone, task-specific Python example. An empty cell indicates
that the corresponding implementation is not currently available in PyPOTS.

Models marked with `🧑‍🔧` were adapted to handle explicit missing values. Depending on the task,
these adaptations use missingness-aware embeddings and training strategies, including observed
reconstruction and masked imputation (ORT + MIT) for imputation, as introduced in [SAITS](https://arxiv.org/abs/2202.08516)[^1].
Input formats and support for irregular sampling depend on the individual model.

Task abbreviations: **IMPT** (Imputation), **FCST** (Forecasting), **CLAF** (Classification),
**CLUS** (Clustering), **ANOD** (Anomaly Detection), and **REPR** (Representation Learning).
Paper references appear at the end of this README.

| Type | Model | IMPT | FCST | CLAF | CLUS | ANOD | REPR | Publication |
|:--|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--|
| LLM&TSFM | <a href="https://time-series.ai"><img src="https://pypots.com/figs/timeseriesai/logo.png" width="26px" align="center">Time-Series.AI</a>  [^36] | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | <a href="https://docs.google.com/forms/d/1Ff2ndYUFQEL3tIcwtcR8lWeopQ2vTXX6D_x8WGFKH6E">Join waitlist</a> |
| Neural Net | HELIX [^55] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/helix_imputation_example.py) |  |  |  |  |  | `2026 - ICML` |
| Neural Net | MixLinear🧑‍🔧[^52] |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/forecasting/mixlinear_forecasting_example.py) |  |  |  |  | `2026 - ICLR` |
| Neural Net | SegRNN🧑‍🔧[^43] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/segrnn_imputation_example.py) | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/forecasting/segrnn_forecasting_example.py) |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/segrnn_anomalydetection_example.py) |  | `2026 - IoT-J` |
| TSFM | TS-ICL[^56] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/tsicl_imputation_example.py) | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/forecasting/tsicl_forecasting_example.py) |  |  |  |  | `2026 - arXiv` |
| Neural Net | LIFTS[^57] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/lifts_imputation_example.py) | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/forecasting/lifts_forecasting_example.py) |  |  |  |  | `2026 - arXiv` |
| Neural Net | TEFN🧑‍🔧[^39] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/tefn_imputation_example.py) | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/forecasting/tefn_forecasting_example.py) | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/classification/tefn_classification_example.py) |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/tefn_anomalydetection_example.py) |  | `2025 - TPAMI` |
| Neural Net | TimeMixer++[^49] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/timemixerpp_imputation_example.py) | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/forecasting/timemixerpp_forecasting_example.py) |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/timemixerpp_anomalydetection_example.py) |  | `2025 - ICLR` |
| LLM | Time-LLM🧑‍🔧[^45] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/timellm_imputation_example.py) | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/forecasting/timellm_forecasting_example.py) |  |  |  |  | `2024 - ICLR` |
| TSFM | MOMENT[^47] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/moment_imputation_example.py) | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/forecasting/moment_forecasting_example.py) |  |  |  |  | `2024 - ICML` |
| Neural Net | TSLANet[^51] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/tslanet_imputation_example.py) |  |  |  |  |  | `2024 - ICML` |
| Neural Net | FITS🧑‍🔧[^41] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/fits_imputation_example.py) | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/forecasting/fits_forecasting_example.py) |  |  |  |  | `2024 - ICLR` |
| Neural Net | TimeMixer[^37] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/timemixer_imputation_example.py) | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/forecasting/timemixer_forecasting_example.py) |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/timemixer_anomalydetection_example.py) |  | `2024 - ICLR` |
| Neural Net | iTransformer🧑‍🔧[^24] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/itransformer_imputation_example.py) |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/classification/itransformer_classification_example.py) |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/itransformer_anomalydetection_example.py) |  | `2024 - ICLR` |
| Neural Net | ModernTCN[^38] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/moderntcn_imputation_example.py) | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/forecasting/moderntcn_forecasting_example.py) |  |  |  |  | `2024 - ICLR` |
| Neural Net | ImputeFormer🧑‍🔧[^34] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/imputeformer_imputation_example.py) |  |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/imputeformer_anomalydetection_example.py) |  | `2024 - KDD` |
| Neural Net | TOTEM[^50] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/totem_imputation_example.py) |  |  |  |  |  | `2024 - TMLR` |
| Neural Net | TKAN🧑‍🔧[^54] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/tkan_imputation_example.py) |  |  |  |  |  | `2024 - arXiv` |
| Neural Net | SAITS[^1] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/saits_imputation_example.py) |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/classification/saits_classification_example.py) |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/saits_anomalydetection_example.py) |  | `2023 - ESWA` |
| LLM | GPT4TS[^46] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/gpt4ts_imputation_example.py) | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/forecasting/gpt4ts_forecasting_example.py) |  |  |  |  | `2023 - NeurIPS` |
| Neural Net | FreTS🧑‍🔧[^23] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/frets_imputation_example.py) |  |  |  |  |  | `2023 - NeurIPS` |
| Neural Net | Koopa🧑‍🔧[^29] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/koopa_imputation_example.py) |  |  |  |  |  | `2023 - NeurIPS` |
| Neural Net | Crossformer🧑‍🔧[^16] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/crossformer_imputation_example.py) |  |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/crossformer_anomalydetection_example.py) |  | `2023 - ICLR` |
| Neural Net | TimesNet[^14] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/timesnet_imputation_example.py) | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/forecasting/timesnet_forecasting_example.py) | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/classification/timesnet_classification_example.py) |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/timesnet_anomalydetection_example.py) |  | `2023 - ICLR` |
| Neural Net | PatchTST🧑‍🔧[^18] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/patchtst_imputation_example.py) |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/classification/patchtst_classification_example.py) |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/patchtst_anomalydetection_example.py) |  | `2023 - ICLR` |
| Neural Net | ETSformer🧑‍🔧[^19] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/etsformer_imputation_example.py) |  |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/etsformer_anomalydetection_example.py) |  | `2023 - ICLR` |
| Neural Net | MICN🧑‍🔧[^27] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/micn_imputation_example.py) | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/forecasting/micn_forecasting_example.py) |  |  |  |  | `2023 - ICLR` |
| Neural Net | DLinear🧑‍🔧[^17] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/dlinear_imputation_example.py) | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/forecasting/dlinear_forecasting_example.py) |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/dlinear_anomalydetection_example.py) |  | `2023 - AAAI` |
| Neural Net | TiDE🧑‍🔧[^28] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/tide_imputation_example.py) |  |  |  |  |  | `2023 - TMLR` |
| Neural Net | CSAI[^42] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/csai_imputation_example.py) |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/classification/csai_classification_example.py) |  |  |  | `2023 - arXiv` |
| Neural Net | TS2Vec[^48] |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/classification/ts2vec_classification_example.py) |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/representation/ts2vec_representation_example.py) | `2022 - AAAI` |
| Neural Net | SCINet🧑‍🔧[^30] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/scinet_imputation_example.py) |  |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/scinet_anomalydetection_example.py) |  | `2022 - NeurIPS` |
| Neural Net | Non-stationary Transformer🧑‍🔧[^25] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/nonstationary_transformer_imputation_example.py) |  |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/nonstationary_transformer_anomalydetection_example.py) |  | `2022 - NeurIPS` |
| Neural Net | FiLM🧑‍🔧[^22] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/film_imputation_example.py) | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/forecasting/film_forecasting_example.py) |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/film_anomalydetection_example.py) |  | `2022 - NeurIPS` |
| Neural Net | RevIN_SCINet🧑‍🔧[^31] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/revin_scinet_imputation_example.py) |  |  |  |  |  | `2022 - ICLR` |
| Neural Net | Pyraformer🧑‍🔧[^26] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/pyraformer_imputation_example.py) |  |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/pyraformer_anomalydetection_example.py) |  | `2022 - ICLR` |
| Neural Net | Raindrop[^5] |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/classification/raindrop_classification_example.py) |  |  |  | `2022 - ICLR` |
| Neural Net | FEDformer🧑‍🔧[^20] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/fedformer_imputation_example.py) |  |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/fedformer_anomalydetection_example.py) |  | `2022 - ICML` |
| Neural Net | Autoformer🧑‍🔧[^15] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/autoformer_imputation_example.py) |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/classification/autoformer_classification_example.py) |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/autoformer_anomalydetection_example.py) |  | `2021 - NeurIPS` |
| Neural Net | CSDI[^12] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/csdi_imputation_example.py) | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/forecasting/csdi_forecasting_example.py) |  |  |  |  | `2021 - NeurIPS` |
| Neural Net | Informer🧑‍🔧[^21] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/informer_imputation_example.py) |  |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/informer_anomalydetection_example.py) |  | `2021 - AAAI` |
| Neural Net | US-GAN[^10] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/usgan_imputation_example.py) |  |  |  |  |  | `2021 - AAAI` |
| Neural Net | CRLI[^6] |  |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/clustering/crli_clustering_example.py) |  |  | `2021 - AAAI` |
| Probabilistic | BTTF[^8] |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/forecasting/bttf_forecasting_example.py) |  |  |  |  | `2021 - TPAMI` |
| Neural Net | StemGNN🧑‍🔧[^33] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/stemgnn_imputation_example.py) |  |  |  |  |  | `2020 - NeurIPS` |
| Neural Net | SeFT[^53] |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/classification/seft_classification_example.py) |  |  |  | `2020 - ICML` |
| Neural Net | Reformer🧑‍🔧[^32] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/reformer_imputation_example.py) |  |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/reformer_anomalydetection_example.py) |  | `2020 - ICLR` |
| Neural Net | GP-VAE[^11] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/gpvae_imputation_example.py) |  |  |  |  |  | `2020 - AISTATS` |
| Neural Net | VaDER[^7] |  |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/clustering/vader_clustering_example.py) |  |  | `2019 - GigaSci.` |
| Neural Net | M-RNN[^9] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/mrnn_imputation_example.py) |  |  |  |  |  | `2019 - TBME` |
| Neural Net | BRITS[^3] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/brits_imputation_example.py) |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/classification/brits_classification_example.py) |  |  |  | `2018 - NeurIPS` |
| Neural Net | GRU-D[^4] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/grud_imputation_example.py) |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/classification/grud_classification_example.py) |  |  |  | `2018 - Sci. Rep.` |
| Neural Net | TCN🧑‍🔧[^35] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/tcn_imputation_example.py) |  |  |  |  |  | `2018 - arXiv` |
| Neural Net | Transformer🧑‍🔧[^2] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/transformer_imputation_example.py) | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/forecasting/transformer_forecasting_example.py) |  |  | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/anomaly_detection/transformer_anomalydetection_example.py) |  | `2017 - NeurIPS` |
| MF | TRMF[^44] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/trmf_imputation_example.py) |  |  |  |  |  | `2016 - NeurIPS` |
| Naive | Lerp[^40] | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/lerp_imputation_example.py) |  |  |  |  |  |  |
| Naive | LOCF/NOCB | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/locf_imputation_example.py) |  |  |  |  |  |  |
| Naive | Mean | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/mean_imputation_example.py) |  |  |  |  |  |  |
| Naive | Median | [✅](https://github.com/WenjieDu/PyPOTS/blob/main/examples/imputation/median_imputation_example.py) |  |  |  |  |  |  |

**Model types:** LLM-based models build on language models pretrained on text, whereas TSFMs build on models
pretrained on time series data. These categories describe the underlying modeling approach.

Want to add a model? See the [contribution guide](https://github.com/WenjieDu/PyPOTS#-contribution) to make your research more accessible,
reproducible, and discoverable by the time series community.

## ❖ PyPOTS Ecosystem

Welcome to the PyPOTS coffee universe! Our ecosystem takes inspiration from brewing coffee,
with each library playing a distinct role in the time series analysis pipeline.

<a href="https://github.com/WenjieDu/TSDB">
    <img src="https://pypots.com/figs/pypots_logos/TSDB/logo_FFBG.svg" align="left" width="140" alt="TSDB logo"/>
</a>

👈 Time series datasets are our coffee beans, and POTS datasets are beans with missing parts.
[Time Series Data Beans (TSDB)](https://github.com/WenjieDu/TSDB) makes public time series datasets easy to load.
Visit its repository for the latest list of supported datasets.

<a href="https://github.com/WenjieDu/PyGrinder">
    <img src="https://pypots.com/figs/pypots_logos/PyGrinder/logo_FFBG.svg" align="right" width="140" alt="PyGrinder logo"/>
</a>

👉 [PyGrinder](https://github.com/WenjieDu/PyGrinder) grinds your coffee beans into incomplete ones by simulating missing values.
It supports all three missingness mechanisms defined by Rubin[^13], namely MCAR (missing completely at random),
MAR (missing at random), and MNAR (missing not at random). Introduce synthetic missing values in a single line of code.

<a href="https://github.com/WenjieDu/BenchPOTS">
    <img src="https://pypots.com/figs/pypots_logos/BenchPOTS/logo_FFBG.svg" align="left" width="140" alt="BenchPOTS logo"/>
</a>

👈 Take a seat on the bench! [BenchPOTS](https://github.com/WenjieDu/BenchPOTS) provides standardized
data preprocessing pipelines for fair and reproducible evaluation of time series algorithms across tasks.

<a href="https://github.com/WenjieDu/BrewPOTS">
    <img src="https://pypots.com/figs/pypots_logos/BrewPOTS/logo_FFBG.svg" align="right" width="140" alt="BrewPOTS logo"/>
</a>

👉 Ready to brew? [BrewPOTS](https://github.com/WenjieDu/BrewPOTS) brings together tutorials and notebooks
to help you learn how to brew your POTS datasets with PyPOTS.

<p align="center">
<a href="https://pypots.com/ecosystem/">
    <img src="https://pypots.com/figs/pypots_logos/Ecosystem/PyPOTS_Ecosystem_Pipeline.png" width="95%"/>
</a>
<br>
<b> ☕️ Welcome to the universe of PyPOTS. Enjoy it and have fun!</b>
</p>

## ❖ Citing PyPOTS

PyPOTS was first introduced at the [MiLeTS workshop at KDD 2023](https://kdd-milets.github.io/milets2023/).
If you use PyPOTS in your research, please cite the [toolkit paper](https://arxiv.org/abs/2305.18811v2).


```bibtex
@article{du2023pypots,
title = {{PyPOTS: A Python Toolkit for Data Mining on Partially-Observed Time Series}},
author = {Wenjie Du},
journal = {SIGKDD MiLeTS Workshop},
url = {https://arxiv.org/abs/2305.18811v1},
year = {2023},
}
```

```bibtex
@article{du2025pypots,
title = {{PyPOTS: A Python Toolkit for Machine Learning on Partially-Observed Time Series}},
author = {Wenjie Du and Yiyuan Yang and Linglong Qian and Jun Wang and Qingsong Wen},
journal = {arXiv preprint arXiv:2305.18811v2},
url = {https://arxiv.org/abs/2305.18811v2},
year = {2025},
}
```

There are scientific research projects using PyPOTS and referencing it in their papers.
Explore [research citing or using PyPOTS here](https://scholar.google.com/scholar?as_ylo=2022&q=%E2%80%9CPyPOTS%E2%80%9D&hl=en).

### Related Research

- **[SAITS](https://doi.org/10.1016/j.eswa.2023.119619)** (ESWA 2023, 1K+ citations): The research project that gave rise to PyPOTS. Its original implementation of self-attention-based time series imputation laid the foundation for the PyPOTS codebase.
- **[TSI-Bench](https://arxiv.org/abs/2406.12747)** (arXiv preprint 2024): A standardized time series imputation benchmark spanning 34,804 experiments, 28 algorithms, and 8 datasets under diverse missingness settings. Code repository: [Awesome_Imputation](https://github.com/WenjieDu/Awesome_Imputation).
- **[Deep Learning for Multivariate Time Series Imputation: A Survey](https://doi.org/10.24963/ijcai.2025/1187)** (IJCAI 2025): A taxonomy of imputation methods, toolkits, and research directions.
- **[HELIX](https://openreview.net/forum?id=FN20iuPnEU)** (ICML 2026 Spotlight): Learnable identities and cross-dimensional synthesis for time series imputation.
- **[End-to-End Learning for Partially-Observed Time Series with PyPOTS](https://doi.org/10.1145/3770855.3816470)** (KDD 2026 Hands-on Tutorial): Practical end-to-end workflows for learning from partially observed time series, from data preparation and multi-task modeling to evaluation and custom model integration. [Tutorial materials](https://pypots.com/events/KDD2026Tutorial/).

## ❖ Contribution

We welcome contributions from researchers, engineers, and users:

- **New models:** Review our [inclusion criteria](https://docs.pypots.com/en/latest/faq.html#inclusion-criteria) and use a task-specific template (e.g., [imputation](https://github.com/WenjieDu/PyPOTS/tree/main/pypots/imputation/template)).
- **Bug fixes and features:** Report reproducible issues or propose improvements through [GitHub Issues](https://github.com/WenjieDu/PyPOTS/issues).
- **Documentation and examples:** Improve tutorials, API documentation, usage examples, and translations.
- **Testing:** Add regression tests and help improve the reliability and reproducibility of PyPOTS.
- **Security:** Report vulnerabilities privately as described in [SECURITY.md](https://github.com/WenjieDu/PyPOTS/blob/main/SECURITY.md).

Before submitting a pull request, we recommend enabling [pre-commit](https://pre-commit.com/) in your local clone to automatically format and check your changes:

```bash
pip install pre-commit
pre-commit install
```

The hooks run automatically when you commit. If a hook modifies files, review and stage the changes, then commit again. GitHub CI checks Ruff linting and formatting even if you do not install the hooks. See the [developer guide](https://docs.pypots.com/en/latest/dev_getting_started.html#using-pre-commit) for details.

Contributors are recognized on the [GitHub contributors page](https://github.com/WenjieDu/PyPOTS/graphs/contributors),
the [PyPOTS website](https://pypots.com/about/#volunteer-developers), and in release notes where applicable.
Please follow our [Code of Conduct](https://github.com/WenjieDu/PyPOTS/blob/main/CODE_OF_CONDUCT.md).

If you find PyPOTS useful, consider starring the repository to help others discover it.
Explore the [affiliations of our users](https://pypots.com/users/).

## ❖ Community

Join the PyPOTS community to ask questions, discuss research, share feedback, and collaborate:

- [Discord](https://discord.gg/DXpCTdrvCE): General discussion, Q&A, and development.
- ~~Slack~~ (retired): We have moved to Discord because Slack's free plan does not retain conversation history indefinitely.
- [LinkedIn](https://www.linkedin.com/company/pypots): Official announcements and news.
- [WeChat (微信公众号)](https://mp.weixin.qq.com/s/X3ukIgL1QpNH8ZEXq1YifA): Follow the official account to access the community group QR code.
- [GitHub Issues](https://github.com/WenjieDu/PyPOTS/issues): Bug reports and feature requests.

We welcome feedback and ideas from everyone working on real-world time series.

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

[^36]: Time-Series.AI is building next-generation AI solutions for unified time series analysis, powered by foundation models and AI agents. Join our waitlist for early access.
<a href="https://time-series.ai"><img src="https://pypots.com/figs/timeseriesai/logo.png" width="20px">Time-Series.AI</a>

[^37]: Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., ... & ZHOU, J. (2024).
[TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting](https://openreview.net/forum?id=7oLshfEIC2).
*ICLR 2024*.

[^38]: Luo, D., & Wang X. (2024).
[ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis](https://openreview.net/forum?id=vpJMJerXHU).
*ICLR 2024*.

[^39]: Zhan, T., He, Y., Deng, Y., Li, Z., Du, W., & Wen, Q. (2025).
[Time Evidence Fusion Network: Multi-source View in Long-Term Time Series Forecasting](https://doi.org/10.1109/TPAMI.2025.3596905).
*TPAMI 2025*.

[^40]: [Wikipedia: Linear interpolation](https://en.wikipedia.org/wiki/Linear_interpolation)

[^41]: Xu, Z., Zeng, A., & Xu, Q. (2024).
[FITS: Modeling Time Series with 10k parameters](https://openreview.net/forum?id=bWcnvZ3qMb).
*ICLR 2024*.

[^42]: Qian, L., Ibrahim, Z., Ellis, H. L., Zhang, A., Zhang, Y., Wang, T., & Dobson, R. (2023).
[Knowledge Enhanced Conditional Imputation for Healthcare Time-series](https://arxiv.org/abs/2312.16713).
*arXiv 2023*.

[^43]: Lin, S., Lin, W., Wu, W., Zhao, F., Mo, R., & Zhang, H. (2026).
[SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting](https://arxiv.org/abs/2308.11200).
*IEEE IoT-J 2026*.

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

[^52]: Ma, A., Luo, D., & Sha, M. (2026).
[MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with 0.1K Parameters](https://openreview.net/forum?id=QUj0KuCumD).
*ICLR 2026*.

[^53]: Horn, M., Moor, M., Bock, C., Rieck, B., & Borgwardt, K. (2020).
[Set Functions for Time Series](https://proceedings.mlr.press/v119/horn20a).
*ICML 2020*.

[^54]: Genet, R., & Inzirillo, H. (2024).
[TKAN: Temporal Kolmogorov-Arnold Networks](https://arxiv.org/abs/2405.07344).
*arXiv 2024*.

[^55]: Zhang, F., Du, W., Zhang, H., Yu, K., & Qu, S. (2026).
[HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation](https://openreview.net/forum?id=FN20iuPnEU).
*ICML 2026*.

[^56]: Le Naour, E., Nabil, T., & Petralia, A. (2026).
[TS-ICL: A Flexible Time-Indexed Foundation Model for Time Series via In-Context Learning](https://arxiv.org/abs/2606.05878).
*arXiv 2026*.

[^57]: Wu, S., Wei, Y., & Wang, K. (2026).
[Latent-Driven Imputation and Forecasting for Partially Observed Time Series](https://arxiv.org/abs/2610.03916).
*arXiv 2026*.
