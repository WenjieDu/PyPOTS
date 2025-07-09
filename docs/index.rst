.. PyPOTS documentation index page created by Wenjie Du <wenjay.du@gmail.com>

Welcome to PyPOTS docs!
===============================
.. image:: https://pypots.com/figs/pypots_logos/PyPOTS/logo_FFBG.svg?sanitize=true
   :height: 156
   :align: right
   :target: https://github.com/WenjieDu/PyPOTS
   :alt: PyPOTS logo

**A Python Toolbox for Machine Learning on Partially-Observed Time Series**

.. image:: https://img.shields.io/badge/Python-v3.8+-E97040?logo=python&logoColor=white
   :alt: Python version
   :target: https://docs.pypots.com/en/latest/install.html#reasons-of-version-limitations-on-dependencies

.. image:: https://img.shields.io/badge/PyTorch%20Landscape-EE4C2C?logo=pytorch&logoColor=white
   :alt: Pytorch landscape
   :target: https://landscape.pytorch.org/?item=modeling--specialized--pypots

.. image:: https://img.shields.io/github/v/release/wenjiedu/pypots?color=EE781F&include_prereleases&label=Release&logo=github&logoColor=white
   :alt: the latest release version
   :target: https://github.com/WenjieDu/PyPOTS/releases

.. image:: https://img.shields.io/badge/License-BSD--3-E9BB41?logo=opensourceinitiative&logoColor=white
   :alt: BSD-3 license
   :target: https://github.com/WenjieDu/PyPOTS/blob/main/LICENSE

.. image:: https://img.shields.io/badge/Contributor_Covenant-2.1-4baaaa
   :alt: Code of Conduct
   :target: https://github.com/WenjieDu/PyPOTS?tab=coc-ov-file

.. image:: https://img.shields.io/badge/join_us-community!-C8A062
   :alt: Community
   :target: https://github.com/WenjieDu/PyPOTS#-community

.. image:: https://img.shields.io/github/contributors/wenjiedu/pypots?color=D8E699&label=Contributors&logo=GitHub
   :alt: GitHub Contributors
   :target: https://github.com/WenjieDu/PyPOTS/graphs/contributors

.. image:: https://img.shields.io/github/stars/wenjiedu/pypots?logo=None&color=6BB392&label=%E2%98%85%20Stars
   :alt: GitHub Repo stars
   :target: https://star-history.com/#wenjiedu/pypots

.. image:: https://img.shields.io/github/forks/wenjiedu/pypots?logo=forgejo&logoColor=black&label=Forks
   :alt: GitHub Repo forks
   :target: https://github.com/WenjieDu/PyPOTS/network/members

.. image:: https://sonarcloud.io/api/project_badges/measure?project=WenjieDu_PyPOTS&metric=sqale_rating
   :alt: maintainability
   :target: https://sonarcloud.io/component_measures?id=WenjieDu_PyPOTS&metric=sqale_rating&view=list

.. image:: https://img.shields.io/coverallsCoverage/github/WenjieDu/PyPOTS?branch=full_test&logo=coveralls&color=75C1C4&label=Coverage
   :alt: Coveralls coverage
   :target: https://coveralls.io/github/WenjieDu/PyPOTS?branch=full_test

.. image:: https://img.shields.io/github/actions/workflow/status/wenjiedu/pypots/testing_ci.yml?logo=circleci&color=C8D8E1&label=CI
   :alt: GitHub Testing
   :target: https://github.com/WenjieDu/PyPOTS/actions/workflows/testing_ci.yml

.. image:: https://img.shields.io/readthedocs/pypots?logo=readthedocs&label=Docs&logoColor=white&color=395260
   :alt: Docs building
   :target: https://readthedocs.org/projects/pypots/builds

.. image:: https://img.shields.io/badge/Code_Style-black-000000
   :alt: Code Style
   :target: https://github.com/psf/black

.. image:: https://pypots.com/figs/downloads_badges/conda_pypots_downloads.svg
   :alt: Conda downloads
   :target: https://anaconda.org/conda-forge/pypots

.. image:: https://pypots.com/figs/downloads_badges/pypi_pypots_downloads.svg
   :alt: PyPI downloads
   :target: https://pepy.tech/project/pypots

.. image:: https://img.shields.io/badge/DOI-10.48550/arXiv.2305.18811-F8F7F0
   :alt: arXiv DOI
   :target: https://arxiv.org/abs/2305.18811

.. image:: https://pypots.com/figs/pypots_logos/readme/CN.svg
   :alt: README in Chinese
   :target: https://github.com/WenjieDu/PyPOTS/blob/main/README_zh.md

.. image:: https://pypots.com/figs/pypots_logos/readme/US.svg
   :alt: README in English
   :target: https://github.com/WenjieDu/PyPOTS/blob/main/README.md

-----------------


⦿ `Motivation`: Due to all kinds of reasons like failure of collection sensors, communication error, and unexpected malfunction, missing values are common to see in time series from the real-world environment.
This makes partially-observed time series (POTS) a pervasive problem in open-world modeling and prevents advanced data analysis.
Although this problem is important, the area of data mining on POTS still lacks a dedicated toolkit. PyPOTS is created to fill in this blank.

⦿ `Mission`: PyPOTS is born to become a handy toolbox that is going to make data mining on POTS easy rather than tedious,
to help engineers and researchers focus more on the core problems in their hands rather than on how to deal with the missing parts in their data.
PyPOTS will keep integrating classical and the latest state-of-the-art data mining algorithms for partially-observed multivariate time series.
For sure, besides various algorithms, PyPOTS is going to have unified APIs together with detailed documentation and interactive examples across algorithms as tutorials.

🤗 **Please** star this repo to help others notice PyPOTS if you think it is a useful toolkit.
**Please** properly `cite PyPOTS <https://docs.pypots.com/en/latest/milestones.html#citing-pypots>`_ in your publications
if it helps with your research. This really means a lot to our open-source research. Thank you!

The rest of this readme file is organized as follows:
`❖ Available Algorithms <#id1>`_,
`❖ PyPOTS Ecosystem <#id37>`_,
`❖ Installation <#id39>`_,
`❖ Usage <#id41>`_,
`❖ Citing PyPOTS <#id43>`_,
`❖ Contribution <#id44>`_,
`❖ Community <#id45>`_.


❖ Available Algorithms
^^^^^^^^^^^^^^^^^^^^^^^
PyPOTS supports imputation, classification, clustering, forecasting, and anomaly detection tasks on multivariate partially-observed
time series with missing values. The table below shows the availability of each algorithm in PyPOTS for different tasks.
The symbol ✅ indicates the algorithm is available for the corresponding task (note that models will be continuously updated
in the future to handle tasks that are not currently supported. Stay tuned❗️).

🌟 Since **v0.2**, all neural-network models in PyPOTS has got hyperparameter-optimization support.
This functionality is implemented with the `Microsoft NNI <https://github.com/microsoft/nni>`_ framework. You may want to refer to our time-series
imputation survey repo `Awesome_Imputation <https://github.com/WenjieDu/Awesome_Imputation>`_ to see how to config and
tune the hyperparameters.

🔥 Note that all models whose name with `🧑‍🔧` in the table (e.g. Transformer, iTransformer, Informer etc.) are not originally
proposed as algorithms for POTS data in their papers, and they cannot directly accept time series with missing values as input, let alone imputation.
To make them applicable to POTS data, we specifically apply the embedding strategy and training approach (ORT+MIT)
the same as we did in `the SAITS paper <https://arxiv.org/pdf/2202.08516)>`_ :cite:`du2023SAITS`.

The task types are abbreviated as follows: **IMPU**: Imputation; **FORE**: Forecasting;
**CLAS**: Classification; **CLUS**: Clustering; **ANOD**: Anomaly Detection.
In addition to the 5 tasks, PyPOTS also provides TS2Vec :cite:`yue2022ts2vec` for time series representation learning and vectorization.
The paper references are all listed at the bottom of this readme file.

+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Type           | Algorithm                                                 | IMPU | FORE | CLAS | CLUS | ANOD | Year - Venue          |
+================+===========================================================+======+======+======+======+======+=======================+
| Neural Net     | TimeMixer++        :cite:`wang2025timemixerpp`            |  ✅  |      |      |      |  ✅  | ``2025 - ICLR``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| LLM            | Time-LLM🧑‍🔧        :cite:`jin2024timellm`               |  ✅  |  ✅  |      |      |      | ``2024 - ICLR``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| TSFM           | MOMENT🧑‍🔧        :cite:`goswami2024moment`              |  ✅  |  ✅  |      |      |      | ``2024 - ICML``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | TSLANet            :cite:`eldele2024tslanet`              |  ✅  |      |      |      |      | ``2024 - ICML``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | TEFN🧑‍🔧           :cite:`zhan2024tefn`                  |  ✅  |  ✅  |  ✅  |      |  ✅  | ``2024 - arXiv``      |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | FITS🧑‍🔧           :cite:`xu2024fits`                    |  ✅  |  ✅  |      |      |      | ``2024 - ICLR``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | TimeMixer      :cite:`wang2024timemixer`                  |  ✅  |  ✅  |      |      |      | ``2024 - ICLR``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | iTransformer🧑‍🔧 :cite:`liu2024itransformer`             |  ✅  |      |  ✅  |      |      | ``2024 - ICLR``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | ModernTCN :cite:`luo2024moderntcn`                        |  ✅  |  ✅  |      |      |      | ``2024 - ICLR``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | ImputeFormer      :cite:`nie2024imputeformer`             |  ✅  |      |      |      |  ✅  | ``2024 - KDD``        |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | TOTEM            :cite:`talukder2024totem`                |  ✅  |      |      |      |      | ``2024 - TMLR``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | SAITS :cite:`du2023SAITS`                                 |  ✅  |      |  ✅  |      |  ✅  | ``2023 - ESWA``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| LLM            | GPT4TS :cite:`zhou2023gpt4ts`                             |  ✅  |  ✅  |      |      |      | ``2023 - NeurIPS``    |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | FreTS🧑‍🔧 :cite:`yi2023frets`                            |  ✅  |      |      |      |      | ``2023 - NeurIPS``    |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | Koopa🧑‍🔧 :cite:`liu2023koopa`                           |  ✅  |      |      |      |      | ``2023 - NeurIPS``    |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | Crossformer🧑‍🔧 :cite:`zhang2023crossformer`             |  ✅  |      |      |      |      | ``2023 - ICLR``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | TimesNet :cite:`wu2023timesnet`                           |  ✅  |  ✅  |  ✅  |      |  ✅  | ``2023 - ICLR``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | PatchTST🧑‍🔧 :cite:`nie2023patchtst`                     |  ✅  |      |  ✅  |      |  ✅  | ``2023 - ICLR``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | ETSformer🧑‍🔧 :cite:`woo2023etsformer`                   |  ✅  |      |      |      |      | ``2023 - ICLR``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | MICN🧑‍🔧 :cite:`wang2023micn`                            |  ✅  |  ✅  |      |      |      | ``2023 - ICLR``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | DLinear🧑‍🔧 :cite:`zeng2023dlinear`                      |  ✅  |  ✅  |      |      |  ✅  | ``2023 - AAAI``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | TiDE🧑‍🔧 :cite:`das2023tide`                             |  ✅  |      |      |      |      | ``2023 - TMLR``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | CSAI :cite:`qian2023csai`                                 |  ✅  |      |  ✅  |      |      | ``2023 - arXiv``      |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | SegRNN🧑‍🔧 :cite:`lin2023segrnn`                         |  ✅  |  ✅  |      |      |  ✅  | ``2023 - arXiv``      |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | TS2Vec :cite:`yue2022ts2vec`                              |      |      |  ✅  |      |      | ``2022 - AAAI``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | SCINet🧑‍🔧 :cite:`liu2022scinet`                         |  ✅  |      |      |      |  ✅  | ``2022 - NeurIPS``    |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | Nonstationary Tr🧑‍🔧 :cite:`liu2022nonstationary`        |  ✅  |      |      |      |      | ``2022 - NeurIPS``    |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | FiLM🧑‍🔧 :cite:`zhou2022film`                            |  ✅  |  ✅  |      |      |      | ``2022 - NeurIPS``    |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | RevIN_SCINet🧑‍🔧 :cite:`kim2022revin`                    |  ✅  |      |      |      |      | ``2022 - ICLR``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | Pyraformer🧑‍🔧 :cite:`liu2022pyraformer`                 |  ✅  |      |      |      |      | ``2022 - ICLR``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | Raindrop :cite:`zhang2022Raindrop`                        |      |      |  ✅  |      |      | ``2022 - ICLR``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | FEDformer🧑‍🔧 :cite:`zhou2022fedformer`                  |  ✅  |      |      |      |      | ``2022 - ICML``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | Autoformer🧑‍🔧 :cite:`wu2021autoformer`                  |  ✅  |      |  ✅  |      |  ✅  | ``2021 - NeurIPS``    |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | CSDI :cite:`tashiro2021csdi`                              |  ✅  |  ✅  |      |      |      | ``2021 - NeurIPS``    |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | Informer🧑‍🔧 :cite:`zhou2021informer`                    |  ✅  |      |      |      |      | ``2021 - AAAI``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | US-GAN :cite:`miao2021SSGAN`                              |  ✅  |      |      |      |      | ``2021 - AAAI``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | CRLI :cite:`ma2021CRLI`                                   |      |      |      |  ✅  |      | ``2021 - AAAI``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Probabilistic  | BTTF :cite:`chen2021BTMF`                                 |      |  ✅  |      |      |      | ``2021 - TPAMI``      |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | StemGNN🧑‍🔧 :cite:`cao2020stemgnn`                       |  ✅  |      |      |      |      | ``2020 - NeurIPS``    |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | Reformer🧑‍🔧 :cite:`kitaev2020reformer`                  |  ✅  |      |      |      |  ✅  | ``2020 - ICLR``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | GP-VAE :cite:`fortuin2020gpvae`                           |  ✅  |      |      |      |      | ``2020 - AISTATS``    |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | VaDER :cite:`dejong2019VaDER`                             |      |      |      |  ✅  |      | ``2019 - GigaSci.``   |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | M-RNN :cite:`yoon2019MRNN`                                |  ✅  |      |      |      |      | ``2019 - TBME``       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | BRITS :cite:`cao2018BRITS`                                |  ✅  |      |  ✅  |      |      | ``2018 - NeurIPS``    |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | GRU-D :cite:`che2018GRUD`                                 |      |      |  ✅  |      |      | ``2018 - Sci. Rep.``  |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | TCN🧑‍🔧 :cite:`bai2018tcn`                               |  ✅  |      |      |      |      | ``2018 - arXiv``      |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Neural Net     | Transformer🧑‍🔧 :cite:`vaswani2017Transformer`           |  ✅  |  ✅  |      |      |      | ``2017 - NeurIPS``    |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| MF             | TRMF :cite:`yu2016trmf`                                   |  ✅  |      |      |      |      | ``2016 - NeurIPS``    |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Naive          | Lerp (Linear Interpolation)                               |  ✅  |      |      |      |      |                       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Naive          | LOCF/NOCB                                                 |  ✅  |      |      |      |      |                       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Naive          | Median                                                    |  ✅  |      |      |      |      |                       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+
| Naive          | Mean                                                      |  ✅  |      |      |      |      |                       |
+----------------+-----------------------------------------------------------+------+------+------+------+------+-----------------------+

🙋 Differences between ``LLM (Large Language Model)`` and ``TSFM (Time-Series Foundation Model)`` in the above table:
``LLM`` refers to the models that are pre-trained on large-scale text data and can be fine-tuned for specific tasks.
``TSFM`` refers to the models that are pre-trained on large-scale time series data, inspired by recent achievements
of foundation models in CV and NLP.

💯 Contribute your model right now to increase your research impact! PyPOTS downloads are increasing rapidly (`1M+ in total and 2k+ daily on PyPI so far <https://www.pepy.tech/projects/pypots>`_),
and your work will be widely used and cited by the community.
Refer to the `contribution guide <#id44>`_ to see how to include your model in PyPOTS.


❖ PyPOTS Ecosystem
^^^^^^^^^^^^^^^^^^^
At PyPOTS, things are related to coffee, which we're familiar with. Yes, this is a coffee universe!
As you can see, there is a coffee pot in the PyPOTS logo.
And what else? Please read on ;-)

.. image:: https://pypots.com/figs/pypots_logos/TSDB/logo_FFBG.svg
   :width: 150
   :alt: TSDB logo
   :align: left
   :target: https://github.com/WenjieDu/TSDB

👈 Time series datasets are taken as coffee beans at PyPOTS, and POTS datasets are incomplete coffee beans with missing parts that have their own meanings.
To make various public time-series datasets readily available to users,
*Time Series Data Beans (TSDB)* is created to make loading time-series datasets super easy!
Visit `TSDB <https://github.com/WenjieDu/TSDB>`_ right now to know more about this handy tool 🛠, and it now supports a total of 172 open-source datasets!

.. image:: https://pypots.com/figs/pypots_logos/PyGrinder/logo_FFBG.svg
   :width: 150
   :alt: PyGrinder logo
   :align: right
   :target: https://github.com/WenjieDu/PyGrinder

👉 To simulate the real-world data beans with missingness, the ecosystem library `PyGrinder <https://github.com/WenjieDu/PyGrinder>`_,
a toolkit helping grind your coffee beans into incomplete ones, is created. Missing patterns fall into three categories according to Robin's theory :cite:`rubin1976missing`:
MCAR (missing completely at random), MAR (missing at random), and MNAR (missing not at random).
PyGrinder supports all of them and additional functionalities related to missingness.
With PyGrinder, you can introduce synthetic missing values into your datasets with a single line of code.

.. image:: https://pypots.com/figs/pypots_logos/BenchPOTS/logo_FFBG.svg
   :width: 150
   :alt: BrewPOTS logo
   :align: left
   :target: https://github.com/WenjieDu/BenchPOTS

👈 To fairly evaluate the performance of PyPOTS algorithms, the benchmarking suite [BenchPOTS](https://github.com/WenjieDu/BenchPOTS) is created,
which provides standard and unified data-preprocessing pipelines to prepare datasets for measuring the performance of different
POTS algorithms on various tasks.

.. image:: https://pypots.com/figs/pypots_logos/BrewPOTS/logo_FFBG.svg
   :width: 150
   :alt: BrewPOTS logo
   :align: right
   :target: https://github.com/WenjieDu/BrewPOTS

👉 Now the beans, grinder, and pot are ready, please have a seat on the bench and let's think about how to brew us a cup of coffee.
Tutorials are necessary! Considering the future workload, PyPOTS tutorials is released in a single repo,
and you can find them in `BrewPOTS <https://github.com/WenjieDu/BrewPOTS>`_.
Take a look at it now, and learn how to brew your POTS datasets!

**☕️ Welcome to the universe of PyPOTS. Enjoy it and have fun!**

.. image:: https://pypots.com/figs/pypots_logos/Ecosystem/PyPOTS_Ecosystem_Pipeline.png
   :width: 95%
   :alt: BrewPOTS logo
   :align: center
   :target: https://pypots.com/ecosystem/


❖ Installation
^^^^^^^^^^^^^^^
PyPOTS is available on both `PyPI <https://pypi.python.org/pypi/pypots>`_ and `Anaconda <https://anaconda.org/conda-forge/pypots>`_.
Furthermore, we also provide a `Docker image <https://hub.docker.com/r/wenjie_du/pypots>`_ for PyPOTS.

Refer to the page `Installation <install.html>`_ to see different ways of installing PyPOTS.


❖ Usage
^^^^^^^^
Besides `BrewPOTS <https://github.com/WenjieDu/BrewPOTS>`_, you can also find a simple and quick-start tutorial notebook
on Google Colab with `this link <https://colab.research.google.com/drive/1HEFjylEy05-r47jRy0H9jiS_WhD0UWmQ>`_.
You can also `raise an issue <https://github.com/WenjieDu/PyPOTS/issues>`_ or `ask in our community <#id21>`_.

Additionally, we present you a usage example of imputing missing values in time series with PyPOTS in
`Section Quick-start Examples <https://docs.pypots.com/en/latest/examples.html>`_, you can click it to view.


❖ Citing PyPOTS
^^^^^^^^^^^^^^^^
**[Updates in Jun 2023]** 🎉A short version of the PyPOTS paper is accepted by the 9th SIGKDD international workshop on
Mining and Learning from Time Series (`MiLeTS'23 <https://kdd-milets.github.io/milets2023/>`_).
Besides, PyPOTS has been included as a `PyTorch Ecosystem <https://landscape.pytorch.org/?item=modeling--specialized--pypots>`_ project.

The paper introducing PyPOTS is available on arXiv at `this URL <https://arxiv.org/abs/2305.18811>`_.,
and we are pursuing to publish it in prestigious academic venues, e.g. JMLR (track for
`Machine Learning Open Source Software <https://www.jmlr.org/mloss/>`_). If you use PyPOTS in your work,
please cite it as below and 🌟star `PyPOTS repository <https://github.com/WenjieDu/PyPOTS>`_ to make others notice this library. 🤗

.. code-block:: bibtex
   :linenos:

   @article{du2023pypots,
   title={{PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series}},
   author={Wenjie Du},
   journal={arXiv preprint arXiv:2305.18811},
   year={2023},
   }

or

..

   Wenjie Du.
   PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series.
   arXiv, abs/2305.18811, 2023.


❖ Contribution
^^^^^^^^^^^^^^^
You're very welcome to contribute to this exciting project!

By committing your code, you'll

1. make your well-established model out-of-the-box for PyPOTS users to run,
   and help your work obtain more exposure and impact.
   Take a look at our `inclusion criteria <https://docs.pypots.com/en/latest/faq.html#inclusion-criteria>`_.
   You can utilize the ``template`` folder in each task package (e.g.
   `pypots/imputation/template <https://github.com/WenjieDu/PyPOTS/tree/main/pypots/imputation/template>`_) to quickly start;
2. become one of `PyPOTS contributors <https://github.com/WenjieDu/PyPOTS/graphs/contributors>`_ and
   be listed as a volunteer developer `on the PyPOTS website <https://pypots.com/about/#volunteer-developers>`_;
3. get mentioned in PyPOTS `release notes <https://github.com/WenjieDu/PyPOTS/releases>`_;

You can also contribute to PyPOTS by simply staring🌟 this repo to help more people notice it.
Your star is your recognition to PyPOTS, and it matters!

The lists of PyPOTS stargazers and forkers are shown below, and we're so proud to have more and more awesome users, as well as more bright ✨stars:

.. image:: https://bytecrank.com/nastyox/reporoster/php/stargazersSVG.php?theme=dark&user=WenjieDu&repo=PyPOTS
   :alt: PyPOTS stargazers
   :target: https://github.com/WenjieDu/PyPOTS/stargazers
.. image:: https://bytecrank.com/nastyox/reporoster/php/forkersSVG.php?theme=dark&user=WenjieDu&repo=PyPOTS
   :alt: PyPOTS forkers
   :target: https://github.com/WenjieDu/PyPOTS/network/members

👀 Check out a full list of our users' affiliations `on PyPOTS website here <https://pypots.com/users/>`_ !


❖ Community
^^^^^^^^^^^^
We care about the feedback from our users, so we're building PyPOTS community on

- `Slack <https://join.slack.com/t/pypots-org/shared_invite/zt-1gq6ufwsi-p0OZdW~e9UW_IA4_f1OfxA>`_. General discussion, Q&A, and our development team are here;
- `LinkedIn <https://www.linkedin.com/company/pypots>`_. Official announcements and news are here;
- `WeChat (微信公众号) <https://mp.weixin.qq.com/s/X3ukIgL1QpNH8ZEXq1YifA>`_. We also run a group chat on WeChat,
  and you can get the QR code from the official account after following it;

If you have any suggestions or want to contribute ideas or share time-series related papers, join us and tell.
PyPOTS community is open, transparent, and surely friendly. Let's work together to build and improve PyPOTS!


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install
   examples

.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: Docs of PyPOTS Ecosystem

   model_api
   pypots
   tsdb
   pygrinder
   benchpots

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   faq
   milestones
   about_us
   references
