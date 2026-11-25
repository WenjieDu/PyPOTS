<img src="https://pypots.com/figs/icons/translate.svg" width="18"> [English](../README.md) | [简体中文](README_zh.md) | [日本語](README_ja.md) | 한국어 | [Deutsch](README_de.md) | [Français](README_fr.md) | [Español](README_es.md) | [Português (Brasil)](README_pt-BR.md) | [Русский](README_ru.md) | [Bahasa Indonesia](README_id.md)

#  PyPOTS <a href="https://github.com/WenjieDu/PyPOTS"><img src="https://pypots.com/figs/pypots_logos/PyPOTS/logo_FFBG.svg" width="200" align='right'></a>

<!-- Translation source: English README at c69b7742. -->

> 실제 세계의 시계열을 통합적으로 분석하는 Python 툴박스

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
    <a href="#-커뮤니티">
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

PyPOTS(발음은 "Pie Pots")는 시계열 분석을 위한 오픈 소스 Python 툴박스입니다. 결측값 대치, 예측, 분류, 군집화, 이상 탐지, 표현 학습을 위해 **통합 API를 갖춘 50개 이상의 모델과 방법**을 제공합니다. 결측값을 포함하는 실제 세계의 시계열을 대상으로 설계되었습니다.

시계열은 기저 시스템 자체가 아니라 해당 시스템에 대한 관측의 모음입니다. 측정과 샘플링에 내재된 한계로 인해 실제 시계열은 본질적으로 부분 관측입니다. 측정 변수는 시스템의 일부 측면만 반영하며, 관측은 이산적이고 불규칙할 수 있는 간격으로 이루어지고 일부 측정값은 누락될 수 있습니다.

**관측되지 않았다고 해서 반드시 존재하지 않는 것은 아닙니다. 부분 관측은 예외가 아니라 일반적인 상태입니다.**

우리는 실제 세계의 시계열을 **부분 관측 시계열(Partially-Observed Time Series, POTS)**로 개념화합니다. 이러한 관점에서 시계열 모델링의 본질은 부분 관측으로부터 기저 시스템과 그 동역학을 학습하는 것입니다. PyPOTS는 이러한 철학을 바탕으로 실제 데이터의 통합 시계열 분석을 지원합니다.

[웹사이트](https://pypots.com) · [문서](https://docs.pypots.com) · [튜토리얼](https://github.com/WenjieDu/BrewPOTS) · [커뮤니티](#-커뮤니티)

[**주요 기능**](#-주요-기능) · [**설치**](#-설치) · [**사용법**](#-사용법) · [**모델 목록**](#-모델-목록) · [**PyPOTS 생태계**](#-pypots-생태계) · [**PyPOTS 인용**](#-pypots-인용) · [**기여**](#-기여)

## ❖ 주요 기능

- **다중 작업 지원:** 결측값 대치, 예측, 분류, 군집화, 이상 탐지, 표현 학습을 하나의 툴박스에서 수행합니다.
- **결측값 기본 지원:** 결측을 고려하는 모델은 사전 대치 없이 불완전한 시계열로부터 직접 학습할 수 있습니다.
- **통합 API:** 일관된 학습 및 예측 인터페이스로 실험과 모델 비교를 간소화합니다.
- **풍부한 모델 목록:** 전통적 방법, 딥러닝, LLM 기반 모델 및 시계열 파운데이션 모델(TSFM)을 포함한 50개 이상의 모델과 방법.
- **<a href="https://landscape.pytorch.org/?item=modeling--specialized--pypots"><img src="https://pypots.com/figs/misc/pytorch.svg" width="26px" align="center">PyTorch Ecosystem</a>:** PyTorch를 기반으로 구축되었으며 PyTorch Landscape에 등재되었습니다.
- **하드웨어 가속:** CPU, NVIDIA CUDA GPU 및 Ascend NPU를 지원하며, 지원 모델에서는 PyTorch DistributedDataParallel(DDP)을 통한 다중 장치 학습, CUDA DataParallel 및 자동 혼합 정밀도를 사용할 수 있습니다. 설정 방법과 제한 사항은 [가속 가이드](https://docs.pypots.com/en/latest/install.html#acceleration)를 참고하세요.
- **유연한 데이터 처리:** 메모리 내 데이터 및 HDF5 파일의 지연 로딩을 지원합니다.
- **하이퍼파라미터 최적화:** [Optuna](https://github.com/optuna/optuna)를 이용한 모델 튜닝.
- **재현 가능한 연구:** 생태계 전반의 데이터셋, 결측 시뮬레이션, 표준화된 전처리 및 평가 도구.
- **확장 가능한 아키텍처:** 모듈식 구성 요소와 표준 템플릿으로 새 알고리즘을 통합할 수 있습니다.

## ❖ 설치

PyPOTS는 Python 3.10 이상이 필요하며 [PyPI](https://pypi.org/project/pypots/)와 [Anaconda](https://anaconda.org/conda-forge/pypots)에서 제공됩니다. 자세한 의존성과 환경 요구 사항은 [설치 가이드](https://docs.pypots.com/en/latest/install.html)를 참조하세요.

환경에 맞는 다음 설치 방법 중 **하나**를 선택하세요. 업그레이드 명령은 이미 PyPOTS를 설치한 경우에만 필요합니다.

**방법 1: pip (권장)**

```bash
pip install pypots
# 업그레이드하려면: pip install pypots --upgrade
```

**방법 2: Conda**

```bash
conda install conda-forge::pypots
# 업그레이드하려면: conda update conda-forge::pypots
```

**방법 3: Docker**

```bash
docker run -it --name pypots wenjiedu/pypots
```

Ctrl-P를 누른 다음 Ctrl-Q를 누르면 컨테이너를 중지하지 않고 분리할 수 있습니다. 다시 연결하려면 `docker attach pypots`를 실행하세요.

**방법 4: main의 최신 소스 코드 (미출시 변경 사항이 포함될 수 있음)**

```bash
pip install https://github.com/WenjieDu/PyPOTS/archive/main.zip
```

## ❖ 사용법

### 빠른 시작

이 독립적인 SAITS 대치 예제는 합성 데이터를 사용하므로 데이터셋을 다운로드할 필요가 없습니다. 입력 형태는 `(n_samples, n_steps, n_features)`이며 결측값은 `NaN`으로 표시합니다.

```python
import numpy as np
from pypots.imputation import SAITS

# 결측값이 있는 합성 시계열 생성
rng = np.random.default_rng(42)
X = rng.normal(size=(16, 24, 8)).astype(np.float32)
X[rng.random(X.shape) < 0.1] = np.nan
train_X, test_X = X[:12], X[12:]

# SAITS 대치 모델 초기화
model = SAITS(
    n_steps=24, n_features=8,
    n_layers=2, d_model=64, n_heads=4,
    d_k=16, d_v=16, d_ffn=128,
    epochs=2, device="cpu",
)
# 모델 학습
model.fit({"X": train_X})

# 테스트 세트 대치
imputed_X = model.impute({"X": test_X})
print(imputed_X.shape)  # (4, 24, 8)
```

### 작업별 PyPOTS 살펴보기

| 작업 | 기능 | 리소스 |
|:--|:--|:--|
| 결측값 대치 | 누락된 관측값 추정 | [예제](https://github.com/WenjieDu/PyPOTS/tree/main/examples/imputation) · [API](https://docs.pypots.com/en/latest/pypots.imputation.html) |
| 예측 | 미래 값 예측 | [예제](https://github.com/WenjieDu/PyPOTS/tree/main/examples/forecasting) · [API](https://docs.pypots.com/en/latest/pypots.forecasting.html) |
| 분류 | 시계열에 클래스 레이블 부여 | [예제](https://github.com/WenjieDu/PyPOTS/tree/main/examples/classification) · [API](https://docs.pypots.com/en/latest/pypots.classification.html) |
| 군집화 | 유사한 시계열 그룹 탐색 | [예제](https://github.com/WenjieDu/PyPOTS/tree/main/examples/clustering) · [API](https://docs.pypots.com/en/latest/pypots.clustering.html) |
| 이상 탐지 | 비정상 관측 또는 패턴 식별 | [예제](https://github.com/WenjieDu/PyPOTS/tree/main/examples/anomaly_detection) · [API](https://docs.pypots.com/en/latest/pypots.anomaly_detection.html) |
| 표현 학습 | 시계열의 잠재 표현 학습 | [예제](https://github.com/WenjieDu/PyPOTS/tree/main/examples/representation) · [API](https://docs.pypots.com/en/latest/pypots.representation.html) |

### 튜토리얼과 실제 데이터 예제

[BrewPOTS](https://github.com/WenjieDu/BrewPOTS) 외에도 [Google Colab](https://colab.research.google.com/drive/1HEFjylEy05-r47jRy0H9jiS_WhD0UWmQ) <a href="https://colab.research.google.com/drive/1HEFjylEy05-r47jRy0H9jiS_WhD0UWmQ"> <img src="https://img.shields.io/badge/GoogleColab-PyPOTS_Tutorials-F9AB00?logo=googlecolab&logoColor=white" alt="Colab tutorials"/> </a>에서 빠른 시작 튜토리얼을 볼 수 있습니다. 궁금한 점은 [PyPOTS 문서](https://docs.pypots.com)를 참고하거나 [이슈 등록](https://github.com/WenjieDu/PyPOTS/issues) 또는 [커뮤니티 질문](#-커뮤니티)을 이용하세요.

체계적인 실습 입문을 원한다면 [KDD 2026 Tutorial: End-to-End Learning for Partially-Observed Time Series with PyPOTS](https://pypots.com/events/KDD2026Tutorial/)을 참조하세요.

아래 PhysioNet2012 예제는 학습, 검증, 테스트 및 대치 평가를 보여줍니다.

<details>
<summary><b>SAITS로 PhysioNet2012를 대치하는 예제 보기:</b></summary>

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

## ❖ 모델 목록

다음 표는 PyPOTS에 구현된 모델과 방법을 발표 연도순으로 나열하며, 첫 행에는 Time-Series.AI 대기 목록이 있습니다. 링크가 있는 ✅는 해당 작업의 독립 실행형 Python 예제를 가리키고, 빈 셀은 해당 구현이 아직 없음을 뜻합니다.

`🧑‍🔧` 표시가 있는 모델은 명시적인 결측값을 처리하도록 조정되었습니다. 작업에 따라 결측을 고려하는 임베딩 및 학습 전략을 적용하며, 대치의 경우 [SAITS](https://arxiv.org/abs/2202.08516)[^1]에서 도입한 관측값 재구성 및 마스킹 대치(ORT + MIT) 등을 사용합니다. 입력 형식과 불규칙 샘플링 지원 여부는 모델에 따라 다릅니다.

작업 약어: **IMPT**(결측값 대치), **FCST**(예측), **CLAF**(분류), **CLUS**(군집화), **ANOD**(이상 탐지), **REPR**(표현 학습). 논문 참고 문헌은 문서 끝에 있습니다.

| 유형 | 모델 | IMPT | FCST | CLAF | CLUS | ANOD | REPR | 발표 |
|:--|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--|
| LLM&TSFM | <a href="https://time-series.ai"><img src="https://pypots.com/figs/timeseriesai/logo.png" width="26px" align="center">Time-Series.AI</a>  [^36] | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | <a href="https://docs.google.com/forms/d/1Ff2ndYUFQEL3tIcwtcR8lWeopQ2vTXX6D_x8WGFKH6E">대기 목록 등록</a> |
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

**모델 유형:** LLM 기반 모델은 텍스트로 사전 학습된 언어 모델을 활용하고 TSFM은 시계열 데이터로 사전 학습된 모델을 활용합니다. 이 분류는 기반 모델링 접근법을 나타냅니다.

모델을 추가하고 싶다면 [기여 가이드](#-기여)를 참고하여 연구의 접근성, 재현성, 발견 가능성을 높여 보세요.

## ❖ PyPOTS 생태계

PyPOTS의 커피 세계에 오신 것을 환영합니다! 커피를 내리는 과정에서 영감을 받은 생태계의 각 라이브러리는 시계열 분석에서 서로 다른 역할을 합니다.

<a href="https://github.com/WenjieDu/TSDB">
    <img src="https://pypots.com/figs/pypots_logos/TSDB/logo_FFBG.svg" align="left" width="140" alt="TSDB logo"/>
</a>

👈 시계열 데이터셋은 커피 원두, POTS 데이터셋은 일부가 빠진 원두입니다. [Time Series Data Beans (TSDB)](https://github.com/WenjieDu/TSDB)는 공개 시계열 데이터셋을 쉽게 불러올 수 있게 합니다. 지원 데이터셋의 최신 목록은 저장소를 참고하세요.

<a href="https://github.com/WenjieDu/PyGrinder">
    <img src="https://pypots.com/figs/pypots_logos/PyGrinder/logo_FFBG.svg" align="right" width="140" alt="PyGrinder logo"/>
</a>

👉 [PyGrinder](https://github.com/WenjieDu/PyGrinder)는 결측값을 시뮬레이션하여 원두를 불완전하게 만듭니다. Rubin[^13]이 정의한 세 가지 결측 메커니즘인 MCAR(완전 무작위 결측), MAR(무작위 결측), MNAR(비무작위 결측)을 모두 지원하며 한 줄의 코드로 인공 결측값을 도입할 수 있습니다.

<a href="https://github.com/WenjieDu/BenchPOTS">
    <img src="https://pypots.com/figs/pypots_logos/BenchPOTS/logo_FFBG.svg" align="left" width="140" alt="BenchPOTS logo"/>
</a>

👈 벤치에 앉아 보세요! [BenchPOTS](https://github.com/WenjieDu/BenchPOTS)는 여러 작업에서 시계열 알고리즘을 공정하고 재현 가능하게 평가할 수 있도록 표준화된 데이터 전처리 파이프라인을 제공합니다.

<a href="https://github.com/WenjieDu/BrewPOTS">
    <img src="https://pypots.com/figs/pypots_logos/BrewPOTS/logo_FFBG.svg" align="right" width="140" alt="BrewPOTS logo"/>
</a>

👉 커피를 내릴 준비가 되었나요? [BrewPOTS](https://github.com/WenjieDu/BrewPOTS)는 PyPOTS로 POTS 데이터셋을 다루는 방법을 배울 수 있는 튜토리얼과 노트북을 제공합니다.

<p align="center">
<a href="https://pypots.com/ecosystem/">
    <img src="https://pypots.com/figs/pypots_logos/Ecosystem/PyPOTS_Ecosystem_Pipeline.png" width="95%"/>
</a>
<br>
<b> ☕️ PyPOTS의 세계에 오신 것을 환영합니다. 즐겁게 활용하세요!</b>
</p>

## ❖ PyPOTS 인용

PyPOTS는 KDD 2023의 [MiLeTS 워크숍](https://kdd-milets.github.io/milets2023/)에서 처음 소개되었습니다. 연구에서 PyPOTS를 사용한다면 [툴킷 논문](https://arxiv.org/abs/2305.18811v2)을 인용해 주세요.

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

PyPOTS를 사용하고 논문에서 인용하는 연구 프로젝트들이 있습니다. [PyPOTS를 인용하거나 사용하는 연구](https://scholar.google.com/scholar?as_ylo=2022&q=%E2%80%9CPyPOTS%E2%80%9D&hl=en)를 살펴보세요.

### 관련 연구

- **[SAITS](https://doi.org/10.1016/j.eswa.2023.119619)** (ESWA 2023, 인용 1,000회 이상): PyPOTS의 기원이 된 연구 프로젝트입니다. 자기 주의 기반 시계열 대치의 원래 구현이 PyPOTS 코드베이스의 토대를 마련했습니다.
- **[TSI-Bench](https://arxiv.org/abs/2406.12747)** (arXiv 프리프린트 2024): 다양한 결측 조건에서 34,804건의 실험, 28개 알고리즘, 8개 데이터셋을 포함하는 표준화된 시계열 결측값 대치 벤치마크. 코드 저장소: [Awesome_Imputation](https://github.com/WenjieDu/Awesome_Imputation).
- **[Deep Learning for Multivariate Time Series Imputation: A Survey](https://doi.org/10.24963/ijcai.2025/1187)** (IJCAI 2025): 대치 방법의 분류 체계, 툴킷 및 연구 방향.
- **[HELIX](https://openreview.net/forum?id=FN20iuPnEU)** (ICML 2026 Spotlight): 학습 가능한 정체성 표현과 차원 간 합성을 이용한 시계열 대치.
- **[End-to-End Learning for Partially-Observed Time Series with PyPOTS](https://doi.org/10.1145/3770855.3816470)** (KDD 2026 실습 튜토리얼): 데이터 준비와 다중 작업 모델링부터 평가, 사용자 정의 모델 통합까지의 실습 워크플로. [튜토리얼 자료](https://pypots.com/events/KDD2026Tutorial/).

## ❖ 기여

연구자, 엔지니어, 사용자의 기여를 환영합니다.

- **새 모델:** [채택 기준](https://docs.pypots.com/en/latest/faq.html#inclusion-criteria)을 확인하고 작업별 템플릿(예: [결측값 대치](https://github.com/WenjieDu/PyPOTS/tree/main/pypots/imputation/template))을 사용하세요.
- **버그 수정과 기능:** [GitHub Issues](https://github.com/WenjieDu/PyPOTS/issues)를 통해 재현 가능한 문제나 개선안을 제안하세요.
- **문서와 예제:** 튜토리얼, API 문서, 사용 예제 및 번역을 개선하세요.
- **테스트:** 회귀 테스트를 추가하여 신뢰성과 재현성을 높이세요.
- **보안:** [SECURITY.md](https://github.com/WenjieDu/PyPOTS/blob/main/SECURITY.md)의 안내에 따라 취약점을 비공개로 신고하세요.

풀 리퀘스트를 제출하기 전에 로컬에 클론한 저장소에서 [pre-commit](https://pre-commit.com/)을 활성화하여 변경 사항을 자동으로 포맷하고 검사하는 것을 권장합니다:

```bash
pip install pre-commit
pre-commit install
```

훅은 커밋할 때 자동으로 실행됩니다. 훅이 파일을 수정하면 변경 사항을 검토하고 다시 스테이징한 다음 커밋을 재시도하세요. 훅을 설치하지 않아도 GitHub CI에서 Ruff 린팅 및 포맷 검사를 수행합니다. 자세한 내용은 [개발자 가이드](https://docs.pypots.com/en/latest/dev_getting_started.html#using-pre-commit)를 참고하세요.

기여자는 [GitHub 기여자 페이지](https://github.com/WenjieDu/PyPOTS/graphs/contributors), [PyPOTS 웹사이트](https://pypots.com/about/#volunteer-developers) 및 해당되는 릴리스 노트에 소개됩니다. [행동 강령](https://github.com/WenjieDu/PyPOTS/blob/main/CODE_OF_CONDUCT.md)을 준수해 주세요.

PyPOTS가 유용하다면 더 많은 사람이 발견할 수 있도록 저장소에 별을 눌러 주세요. [사용자 소속 기관](https://pypots.com/users/)도 확인해 보세요.

## ❖ 커뮤니티

PyPOTS 커뮤니티에 참여하여 질문하고, 연구를 논의하고, 의견을 공유하며 협업하세요.

- [Discord](https://discord.gg/DXpCTdrvCE): 일반 토론, 질의응답 및 개발.
- ~~Slack~~ (운영 종료): 무료 플랜에서 대화 기록을 무기한 보관하지 못하므로 Discord로 이전했습니다.
- [LinkedIn](https://www.linkedin.com/company/pypots): 공식 공지 및 소식.
- [WeChat (공식 계정)](https://mp.weixin.qq.com/s/X3ukIgL1QpNH8ZEXq1YifA): 공식 계정을 팔로우하여 커뮤니티 그룹 QR 코드를 확인하세요.
- [GitHub Issues](https://github.com/WenjieDu/PyPOTS/issues): 버그 신고 및 기능 요청.

실제 세계의 시계열을 다루는 모든 분의 의견과 아이디어를 환영합니다.

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

[^36]: Time-Series.AI는 파운데이션 모델과 AI 에이전트를 기반으로 통합 시계열 분석을 위한 차세대 AI 솔루션을 개발하고 있습니다. 얼리 액세스를 원하시면 대기 목록에 등록해 주세요.
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
