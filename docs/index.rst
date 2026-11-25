.. PyPOTS documentation index page created by Wenjie Du <wenjay.du@gmail.com>

.. |translate| image:: https://pypots.com/figs/icons/translate.svg
   :width: 18px
   :alt: README translations

Welcome to PyPOTS docs!
===============================

|translate| English | `简体中文 <https://github.com/WenjieDu/PyPOTS/blob/main/i18n/README_zh.md>`__ | `日本語 <https://github.com/WenjieDu/PyPOTS/blob/main/i18n/README_ja.md>`__ | `한국어 <https://github.com/WenjieDu/PyPOTS/blob/main/i18n/README_ko.md>`__ | `Deutsch <https://github.com/WenjieDu/PyPOTS/blob/main/i18n/README_de.md>`__ | `Français <https://github.com/WenjieDu/PyPOTS/blob/main/i18n/README_fr.md>`__ | `Español <https://github.com/WenjieDu/PyPOTS/blob/main/i18n/README_es.md>`__ | `Português (Brasil) <https://github.com/WenjieDu/PyPOTS/blob/main/i18n/README_pt-BR.md>`__ | `Русский <https://github.com/WenjieDu/PyPOTS/blob/main/i18n/README_ru.md>`__ | `Bahasa Indonesia <https://github.com/WenjieDu/PyPOTS/blob/main/i18n/README_id.md>`__

.. image:: https://pypots.com/figs/pypots_logos/PyPOTS/logo_FFBG.svg?sanitize=true
   :height: 156
   :align: right
   :target: https://github.com/WenjieDu/PyPOTS
   :alt: PyPOTS logo

**A Python Toolbox for Unified Analysis of Real-World Time Series**

.. image:: https://img.shields.io/badge/Python-v3.10+-F8C6B5?logo=python&logoColor=white
   :alt: Python version
   :target: https://docs.pypots.com/en/latest/install.html#reasons-of-version-limitations-on-dependencies

.. image:: https://img.shields.io/github/v/release/wenjiedu/pypots?color=EE781F&include_prereleases&label=Release&logo=github&logoColor=white
   :alt: the latest release version
   :target: https://github.com/WenjieDu/PyPOTS/releases

.. image:: https://img.shields.io/readthedocs/pypots?logo=readthedocs&label=Docs&logoColor=white&color=395260
   :alt: Docs building
   :target: https://docs.pypots.com

.. image:: https://pypots.com/figs/downloads_badges/pypi_pypots_downloads.svg
   :alt: PyPI downloads
   :target: https://pepy.tech/project/pypots

.. image:: https://pypots.com/figs/downloads_badges/conda_pypots_downloads.svg
   :alt: Conda downloads
   :target: https://anaconda.org/conda-forge/pypots

.. image:: https://img.shields.io/badge/PyTorch%20Landscape-EE4C2C?logo=pytorch&logoColor=white
   :alt: Pytorch landscape
   :target: https://landscape.pytorch.org/?item=modeling--specialized--pypots

.. image:: https://img.shields.io/badge/License-BSD--3-E9BB41?logo=opensourceinitiative&logoColor=white
   :alt: BSD-3 license
   :target: https://github.com/WenjieDu/PyPOTS/blob/main/LICENSE

.. image:: https://img.shields.io/badge/join_us-community!-C8A062
   :alt: Community
   :target: https://github.com/WenjieDu/PyPOTS#-community

.. image:: https://img.shields.io/github/stars/wenjiedu/pypots?logo=None&color=6BB392&label=%E2%98%85%20Stars
   :alt: GitHub Repo stars
   :target: https://star-history.com/#wenjiedu/pypots

.. image:: https://img.shields.io/github/forks/wenjiedu/pypots?logo=forgejo&logoColor=black&label=Forks
   :alt: GitHub Repo forks
   :target: https://github.com/WenjieDu/PyPOTS/network/members

.. image:: https://img.shields.io/github/contributors/wenjiedu/pypots?color=D8E699&label=Contributors&logo=GitHub
   :alt: GitHub contributors
   :target: https://github.com/WenjieDu/PyPOTS/graphs/contributors

.. image:: https://img.shields.io/github/actions/workflow/status/wenjiedu/pypots/testing_ci.yml?logo=circleci&color=C8D8E1&label=CI
   :alt: GitHub Testing
   :target: https://github.com/WenjieDu/PyPOTS/actions/workflows/testing_ci.yml

.. image:: https://img.shields.io/coverallsCoverage/github/WenjieDu/PyPOTS?branch=full_test&logo=coveralls&color=75C1C4&label=Coverage
   :alt: Coveralls coverage
   :target: https://coveralls.io/github/WenjieDu/PyPOTS?branch=full_test

.. image:: https://sonarcloud.io/api/project_badges/measure?project=WenjieDu_PyPOTS&metric=sqale_rating
   :alt: maintainability
   :target: https://sonarcloud.io/component_measures?id=WenjieDu_PyPOTS&metric=sqale_rating&view=list

.. image:: https://img.shields.io/badge/Code_Style-Ruff-D7FF64?logo=ruff
   :alt: Code Style
   :target: https://github.com/astral-sh/ruff

.. image:: https://img.shields.io/badge/Contributor_Covenant-2.1-4baaaa
   :alt: Code of Conduct
   :target: https://github.com/WenjieDu/PyPOTS?tab=coc-ov-file

.. image:: https://pypots.com/figs/misc/ask_deepwiki.svg
   :alt: Ask DeepWiki
   :target: https://deepwiki.com/WenjieDu/PyPOTS

.. image:: https://img.shields.io/badge/DOI-10.48550/arXiv.2305.18811-F8F7F0
   :alt: arXiv DOI
   :target: https://arxiv.org/abs/2305.18811

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


❖ Key Features
^^^^^^^^^^^^^^

- **Multi-task support:** Imputation, forecasting, classification, clustering, anomaly detection, and representation learning in one toolbox.
- **Native missing-value support:** Missingness-aware models can learn directly from incomplete time series without prior imputation.
- **Unified APIs:** Consistent training and prediction interfaces across models, making experimentation and model comparison easier.
- **Rich model zoo:** 50+ models and methods spanning classical approaches, deep learning, LLM-based models, and time series foundation models (TSFMs).
- **PyTorch Ecosystem:** PyPOTS is built on PyTorch and featured in the `PyTorch Landscape <https://landscape.pytorch.org/?item=modeling--specialized--pypots>`__.
- **Hardware acceleration:** CPU, NVIDIA CUDA GPUs, and Ascend NPUs; multi-device training via PyTorch DistributedDataParallel (DDP), CUDA DataParallel, and automatic mixed precision for supported models. See the :ref:`acceleration guide <hardware-acceleration>` for setup and limitations.
- **Flexible data handling:** In-memory data and lazy loading from HDF5 files.
- **Hyperparameter optimization:** Model tuning powered by `Optuna <https://github.com/optuna/optuna>`__.
- **Reproducible research:** Datasets, missingness simulation, standardized preprocessing, and evaluation tools across the PyPOTS ecosystem.
- **Extensible architecture:** Modular model components and standardized templates for integrating new algorithms.

❖ Installation
^^^^^^^^^^^^^^

PyPOTS requires Python 3.10 or later and is available on
`PyPI <https://pypi.org/project/pypots/>`__ and
`Anaconda <https://anaconda.org/conda-forge/pypots>`__.

.. code-block:: bash

   pip install pypots

See :doc:`install` for pip, Conda, Docker and source installation, dependency
requirements, and hardware acceleration. See :doc:`npu` and :doc:`distributed`
for Ascend NPU and distributed training.

❖ Usage
^^^^^^^

This self-contained SAITS imputation example uses synthetic data and requires no dataset download.
Inputs have shape ``(n_samples, n_steps, n_features)``, with missing values represented by NaN.

.. code-block:: python

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

Explore PyPOTS by Task
""""""""""""""""""""""

.. list-table::
   :widths: 20 35 45
   :header-rows: 1

   * - Task
     - What it does
     - Resources
   * - Imputation
     - Estimate missing observations
     - `Examples <https://github.com/WenjieDu/PyPOTS/tree/main/examples/imputation>`__ · :doc:`API <pypots.imputation>`
   * - Forecasting
     - Predict future values
     - `Examples <https://github.com/WenjieDu/PyPOTS/tree/main/examples/forecasting>`__ · :doc:`API <pypots.forecasting>`
   * - Classification
     - Assign labels to time series
     - `Examples <https://github.com/WenjieDu/PyPOTS/tree/main/examples/classification>`__ · :doc:`API <pypots.classification>`
   * - Clustering
     - Discover groups of similar time series
     - `Examples <https://github.com/WenjieDu/PyPOTS/tree/main/examples/clustering>`__ · :doc:`API <pypots.clustering>`
   * - Anomaly detection
     - Identify abnormal observations or patterns
     - `Examples <https://github.com/WenjieDu/PyPOTS/tree/main/examples/anomaly_detection>`__ · :doc:`API <pypots.anomaly_detection>`
   * - Representation learning
     - Learn latent representations of time series
     - `Examples <https://github.com/WenjieDu/PyPOTS/tree/main/examples/representation>`__ · :doc:`API <pypots.representation>`

See :doc:`tutorials` for the PhysioNet2012 example, BrewPOTS notebooks,
Google Colab and the KDD 2026 hands-on tutorial.

❖ Model Zoo
^^^^^^^^^^^

The table below lists the models and methods implemented in PyPOTS, sorted by publication year, alongside the Time-Series.AI waitlist entry.
Each linked ✅ leads to a standalone, task-specific Python example. An empty cell indicates
that the corresponding implementation is not currently available in PyPOTS.

Models marked with ``🧑‍🔧`` were adapted to handle explicit missing values. Depending on the task,
these adaptations use missingness-aware embeddings and training strategies, including observed
reconstruction and masked imputation (ORT + MIT) for imputation, as introduced in SAITS :cite:`du2023SAITS`.
Input formats and support for irregular sampling depend on the individual model.

Task abbreviations: **IMPT** (Imputation), **FCST** (Forecasting), **CLAF** (Classification),
**CLUS** (Clustering), **ANOD** (Anomaly Detection), and **REPR** (Representation Learning).
Paper references appear in :doc:`references`.

.. include:: algo_table.rst

Time-Series.AI is building next-generation AI solutions for unified time series
analysis, powered by foundation models and AI agents. Join our waitlist for early access.

**Model types:** LLM-based models build on language models pretrained on text, whereas TSFMs build on models
pretrained on time series data. These categories describe the underlying modeling approach.

Want to add a model? See the :ref:`contribution guide <contributing-to-pypots>` to make your research more accessible,
reproducible, and discoverable by the time series community.

❖ PyPOTS Ecosystem
^^^^^^^^^^^^^^^^^^

Welcome to the PyPOTS coffee universe! Our ecosystem takes inspiration from brewing coffee,
with each library playing a distinct role in the time series analysis pipeline.

.. image:: https://pypots.com/figs/pypots_logos/TSDB/logo_FFBG.svg
   :width: 140
   :alt: TSDB logo
   :align: left
   :target: https://github.com/WenjieDu/TSDB

👈 Time series datasets are our coffee beans, and POTS datasets are beans with missing parts.
`Time Series Data Beans (TSDB) <https://github.com/WenjieDu/TSDB>`__ makes public time series datasets easy to load.
Visit its repository for the latest list of supported datasets.

.. image:: https://pypots.com/figs/pypots_logos/PyGrinder/logo_FFBG.svg
   :width: 140
   :alt: PyGrinder logo
   :align: right
   :target: https://github.com/WenjieDu/PyGrinder

👉 `PyGrinder <https://github.com/WenjieDu/PyGrinder>`__ grinds your coffee beans into incomplete ones by simulating missing values.
It supports all three missingness mechanisms defined by Rubin :cite:`rubin1976missing`, namely MCAR (missing completely at random),
MAR (missing at random), and MNAR (missing not at random). Introduce synthetic missing values in a single line of code.

.. image:: https://pypots.com/figs/pypots_logos/BenchPOTS/logo_FFBG.svg
   :width: 140
   :alt: BenchPOTS logo
   :align: left
   :target: https://github.com/WenjieDu/BenchPOTS

👈 Take a seat on the bench! `BenchPOTS <https://github.com/WenjieDu/BenchPOTS>`__ provides standardized
data preprocessing pipelines for fair and reproducible evaluation of time series algorithms across tasks.

.. image:: https://pypots.com/figs/pypots_logos/BrewPOTS/logo_FFBG.svg
   :width: 140
   :alt: BrewPOTS logo
   :align: right
   :target: https://github.com/WenjieDu/BrewPOTS

👉 Ready to brew? `BrewPOTS <https://github.com/WenjieDu/BrewPOTS>`__ brings together tutorials and notebooks
to help you learn how to brew your POTS datasets with PyPOTS.

.. image:: https://pypots.com/figs/pypots_logos/Ecosystem/PyPOTS_Ecosystem_Pipeline.png
   :width: 95%
   :alt: PyPOTS ecosystem pipeline
   :align: center
   :target: https://pypots.com/ecosystem/

**☕️ Welcome to the universe of PyPOTS. Enjoy it and have fun!**

❖ Citing PyPOTS
^^^^^^^^^^^^^^^

PyPOTS was first introduced at the `MiLeTS workshop at KDD 2023 <https://kdd-milets.github.io/milets2023/>`__.
If you use PyPOTS in your research, please cite the `toolkit paper <https://arxiv.org/abs/2305.18811v2>`__.

.. code-block:: bibtex

   @article{du2023pypots,
   title = {{PyPOTS: A Python Toolkit for Data Mining on Partially-Observed Time Series}},
   author = {Wenjie Du},
   journal = {SIGKDD MiLeTS Workshop},
   url = {https://arxiv.org/abs/2305.18811v1},
   year = {2023},
   }

.. code-block:: bibtex

   @article{du2025pypots,
   title = {{PyPOTS: A Python Toolkit for Machine Learning on Partially-Observed Time Series}},
   author = {Wenjie Du and Yiyuan Yang and Linglong Qian and Jun Wang and Qingsong Wen},
   journal = {arXiv preprint arXiv:2305.18811v2},
   url = {https://arxiv.org/abs/2305.18811v2},
   year = {2025},
   }

There are scientific research projects using PyPOTS and referencing it in their papers.
Explore `research citing or using PyPOTS here <https://scholar.google.com/scholar?as_ylo=2022&q=%E2%80%9CPyPOTS%E2%80%9D&hl=en>`__.

Related Research
""""""""""""""""

- `SAITS <https://doi.org/10.1016/j.eswa.2023.119619>`__ (ESWA 2023, 1K+ citations): The research project that gave rise to PyPOTS. Its original implementation of self-attention-based time series imputation laid the foundation for the PyPOTS codebase.
- `TSI-Bench <https://arxiv.org/abs/2406.12747>`__ (arXiv preprint 2024): A standardized time series imputation benchmark spanning 34,804 experiments, 28 algorithms, and 8 datasets under diverse missingness settings. Code repository: `Awesome_Imputation <https://github.com/WenjieDu/Awesome_Imputation>`__.
- `Deep Learning for Multivariate Time Series Imputation: A Survey <https://doi.org/10.24963/ijcai.2025/1187>`__ (IJCAI 2025): A taxonomy of imputation methods, toolkits, and research directions.
- `HELIX <https://openreview.net/forum?id=FN20iuPnEU>`__ (ICML 2026 Spotlight): Learnable identities and cross-dimensional synthesis for time series imputation.
- `End-to-End Learning for Partially-Observed Time Series with PyPOTS <https://doi.org/10.1145/3770855.3816470>`__ (KDD 2026 Hands-on Tutorial): Practical end-to-end workflows for learning from partially observed time series, from data preparation and multi-task modeling to evaluation and custom model integration. `Tutorial materials <https://pypots.com/events/KDD2026Tutorial/>`__.

.. _contributing-to-pypots:

❖ Contribution
^^^^^^^^^^^^^^

We welcome contributions from researchers, engineers, and users:

- **New models:** Review our `inclusion criteria <https://docs.pypots.com/en/latest/faq.html#inclusion-criteria>`__ and use a task-specific template (e.g., `imputation <https://github.com/WenjieDu/PyPOTS/tree/main/pypots/imputation/template>`__).
- **Bug fixes and features:** Report reproducible issues or propose improvements through `GitHub Issues <https://github.com/WenjieDu/PyPOTS/issues>`__.
- **Documentation and examples:** Improve tutorials, API documentation, usage examples, and translations.
- **Testing:** Add regression tests and help improve the reliability and reproducibility of PyPOTS.
- **Security:** Report vulnerabilities privately as described in `SECURITY.md <https://github.com/WenjieDu/PyPOTS/blob/main/SECURITY.md>`__.

Before submitting a pull request, we recommend enabling `pre-commit <https://pre-commit.com/>`__ in your local clone to automatically format and check your changes:

.. code-block:: bash

   pip install pre-commit
   pre-commit install

The hooks run automatically when you commit. If a hook modifies files, review and stage the changes, then commit again. GitHub CI checks Ruff linting and formatting even if you do not install the hooks. See the `developer guide <https://docs.pypots.com/en/latest/dev_getting_started.html#using-pre-commit>`__ for details.

Contributors are recognized on the `GitHub contributors page <https://github.com/WenjieDu/PyPOTS/graphs/contributors>`__,
the `PyPOTS website <https://pypots.com/about/#volunteer-developers>`__, and in release notes where applicable.
Please follow our `Code of Conduct <https://github.com/WenjieDu/PyPOTS/blob/main/CODE_OF_CONDUCT.md>`__.

If you find PyPOTS useful, consider starring the repository to help others discover it.
Explore the `affiliations of our users <https://pypots.com/users/>`__.

❖ Community
^^^^^^^^^^^

Join the PyPOTS community to ask questions, discuss research, share feedback, and collaborate:

- `Discord <https://discord.gg/DXpCTdrvCE>`__: General discussion, Q&A, and development.
- **Slack** (retired): We have moved to Discord because Slack's free plan does not retain conversation history indefinitely.
- `LinkedIn <https://www.linkedin.com/company/pypots>`__: Official announcements and news.
- `WeChat (微信公众号) <https://mp.weixin.qq.com/s/X3ukIgL1QpNH8ZEXq1YifA>`__: Follow the official account to access the community group QR code.
- `GitHub Issues <https://github.com/WenjieDu/PyPOTS/issues>`__: Bug reports and feature requests.

We welcome feedback and ideas from everyone working on real-world time series.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install
   tutorials

.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: User Docs of PyPOTS

   model_api
   npu
   distributed
   pypots
   tsdb
   pygrinder
   benchpots

.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: Developer Docs of PyPOTS

   dev_getting_started
   dev_architecture
   dev_integration_guide
   dev_quality

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   faq
   milestones
   references
