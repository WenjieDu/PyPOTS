.. PyPOTS documentation index page
   Created by Wenjie Du <wenjay.du@gmail.com>
   License: GPL-v3

Welcome to PyPOTS docs!
===============================
.. image:: https://pypots.com/figs/pypots_logos/PyPOTS_logo_FFBG.svg?sanitize=true
   :height: 168
   :align: right
   :target: https://github.com/WenjieDu/PyPOTS
   :alt: PyPOTS logo

**A Python Toolbox for Data Mining on Partially-Observed Time Series**

.. image:: https://img.shields.io/badge/Python-v3.7--3.10-E97040?logo=python&logoColor=white
   :alt: Python version
   :target: https://docs.pypots.com/en/latest/install.html#reasons-of-version-limitations-on-dependencies

.. image:: https://img.shields.io/badge/PyTorch-❤️-F8C6B5?logo=pytorch&logoColor=white
   :alt: powered by Pytorch
   :target: https://github.com/WenjieDu/PyPOTS

.. image:: https://img.shields.io/github/v/release/wenjiedu/pypots?color=EE781F&include_prereleases&label=Release&logo=github&logoColor=white
   :alt: the latest release version
   :target: https://github.com/WenjieDu/PyPOTS/releases

.. image:: https://img.shields.io/badge/License-GPL--v3-E9BB41?logo=opensourceinitiative&logoColor=white
   :alt: GPL-v3 license
   :target: https://github.com/WenjieDu/PyPOTS/blob/main/LICENSE

.. image:: https://img.shields.io/badge/join_us-community!-C8A062
   :alt: Community
   :target: #id17

.. image:: https://img.shields.io/github/contributors/wenjiedu/pypots?color=D8E699&label=Contributors&logo=GitHub
   :alt: GitHub Contributors
   :target: https://github.com/WenjieDu/PyPOTS/graphs/contributors

.. image:: https://img.shields.io/github/stars/wenjiedu/pypots?logo=Github&color=6BB392&label=Stars
   :alt: GitHub Repo stars
   :target: https://star-history.com/#wenjiedu/pypots

.. image:: https://img.shields.io/github/forks/wenjiedu/pypots?logo=Github&color=91B821&label=Forks
   :alt: GitHub Repo forks
   :target: https://github.com/WenjieDu/PyPOTS/network/members

.. image:: https://img.shields.io/codeclimate/maintainability-percentage/WenjieDu/PyPOTS?color=3C7699&label=Maintainability&logo=codeclimate
   :alt: Code Climate maintainability
   :target: https://codeclimate.com/github/WenjieDu/PyPOTS

.. image:: https://img.shields.io/coverallsCoverage/github/WenjieDu/PyPOTS?branch=main&logo=coveralls&color=75C1C4&label=Coverage
   :alt: Coveralls coverage
   :target: https://coveralls.io/github/WenjieDu/PyPOTS

.. image:: https://img.shields.io/github/actions/workflow/status/wenjiedu/pypots/testing_ci.yml?logo=github&color=C8D8E1&label=CI
   :alt: GitHub Testing
   :target: https://github.com/WenjieDu/PyPOTS/actions/workflows/testing_ci.yml

.. image:: https://img.shields.io/badge/DOI-10.48550/arXiv.2305.18811-F8F7F0
   :alt: arXiv DOI
   :target: https://arxiv.org/abs/2305.18811

.. image:: https://img.shields.io/endpoint?url=https://pypots.com/figs/downloads_badges/conda_pypots_downloads.json
   :alt: Conda downloads
   :target: https://anaconda.org/conda-forge/pypots

.. image:: https://img.shields.io/endpoint?url=https://pypots.com/figs/downloads_badges/pypi_pypots_downloads.json
   :alt: PyPI downloads
   :target: https://pepy.tech/project/pypots

.. image:: https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FPyPOTS%2FPyPOTS&count_bg=%23009A0A&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visits%20since%20May%202022&edge_flat=false
   :alt: Visiting number

-----------------


⦿ `Motivation`: Due to all kinds of reasons like failure of collection sensors, communication error, and unexpected malfunction, missing values are common to see in time series from the real-world environment. This makes partially-observed time series (POTS) a pervasive problem in open-world modeling and prevents advanced data analysis. Although this problem is important, the area of data mining on POTS still lacks a dedicated toolkit. PyPOTS is created to fill in this blank.

⦿ `Mission`: PyPOTS is born to become a handy toolbox that is going to make data mining on POTS easy rather than tedious, to help engineers and researchers focus more on the core problems in their hands rather than on how to deal with the missing parts in their data. PyPOTS will keep integrating classical and the latest state-of-the-art data mining algorithms for partially-observed multivariate time series. For sure, besides various algorithms, PyPOTS is going to have unified APIs together with detailed documentation and interactive examples across algorithms as tutorials.

🤗 **Please** star this repo to help others notice PyPOTS if you think it is a useful toolkit.
**Please** properly `cite PyPOTS <https://docs.pypots.com/en/latest/milestones.html#citing-pypots>`_ in your publications
if it helps with your research. This really means a lot to our open-source research. Thank you!

.. image:: https://pypots.com/figs/pypots_logos/TSDB_logo_FFBG.svg?sanitize=true
   :width: 170
   :alt: TSDB
   :align: left
   :target: https://github.com/WenjieDu/TSDB

To make various open-source time-series datasets readily available to our users, PyPOTS gets supported by its sub-project `TSDB (Time-Series Data Base) <https://github.com/WenjieDu/TSDB>`_, a toolbox making loading time-series datasets super easy!

Visit `TSDB <https://github.com/WenjieDu/TSDB>`_ right now to know more about this handy tool 🛠! It now supports a total of 168 open-source datasets.

The rest of this readme file is organized as follows:
`❖ Installation <#id1>`_,
`❖ Usage <#id3>`_,
`❖ Available Algorithms <#id4>`_,
`❖ Citing PyPOTS <#id14>`_,
`❖ Contribution <#id15>`_,
`❖ Community <#id16>`_.


❖ Installation
^^^^^^^^^^^^^^^
PyPOTS is available on both `PyPI <https://pypi.python.org/pypi/pypots>`_ and `Anaconda <https://anaconda.org/conda-forge/pypots>`_.

Refer to the page `Installation <install.html>`_ to see different ways of installing PyPOTS.


❖ Usage
^^^^^^^^
.. image:: https://pypots.com/figs/pypots_logos/BrewPOTS_logo_FFBG.svg?sanitize=true
   :width: 160
   :alt: BrewPOTS logo
   :align: left
   :target: https://github.com/WenjieDu/BrewPOTS

PyPOTS tutorials have been released. Considering the future workload, I separate the tutorials into a single repo,
and you can find them in `BrewPOTS <https://github.com/WenjieDu/BrewPOTS>`_.
Take a look at it now, and brew your POTS dataset into a cup of coffee!

If you have further questions, please refer to PyPOTS documentation `docs.pypots.com <https://docs.pypots.com>`_.
Besides, you can also `raise an issue <https://github.com/WenjieDu/PyPOTS/issues>`_ or `ask in our community <#id14>`_.


❖ Available Algorithms
^^^^^^^^^^^^^^^^^^^^^^^
PyPOTS supports imputation, classification, clustering, and forecasting tasks on multivariate time series with missing values. The currently available algorithms of four tasks are cataloged in the following table with four partitions. The paper references are all listed at the bottom of this readme file. Please refer to them if you want more details.

============================== ================ =========================================================================  ====== =========
Task                           Type             Algorithm                                                                  Year   Reference
============================== ================ =========================================================================  ====== =========
Imputation                     Neural Network   SAITS (Self-Attention-based Imputation for Time Series)                    2022   :cite:`du2023SAITS`
Imputation                     Neural Network   Transformer                                                                2017   :cite:`vaswani2017Transformer`, :cite:`du2023SAITS`
Imputation, Classification     Neural Network   BRITS (Bidirectional Recurrent Imputation for Time Series)                 2018   :cite:`cao2018BRITS`
Imputation                     Neural Network   M-RNN (Multi-directional Recurrent Neural Network)                         2019   :cite:`yoon2019MRNN`
Imputation                     Naive            LOCF (Last Observation Carried Forward)                                    /      /
Classification                 Neural Network   GRU-D                                                                      2018   :cite:`che2018GRUD`
Classification                 Neural Network   Raindrop                                                                   2022   :cite:`zhang2022Raindrop`
Clustering                     Neural Network   CRLI (Clustering Representation Learning on Incomplete time-series data)   2021   :cite:`ma2021CRLI`
Clustering                     Neural Network   VaDER (Variational Deep Embedding with Recurrence)                         2019   :cite:`dejong2019VaDER`
Forecasting                    Probabilistic    BTTF (Bayesian Temporal Tensor Factorization)                              2021   :cite:`chen2021BTMF`
============================== ================ =========================================================================  ====== =========


❖ Citing PyPOTS
^^^^^^^^^^^^^^^^
**[Updates in Jun 2023]** 🎉A short version of the PyPOTS paper is accepted by the 9th SIGKDD international workshop on
Mining and Learning from Time Series (`MiLeTS'23 <https://kdd-milets.github.io/milets2023/>`_).
Besides, PyPOTS has been included as a `PyTorch Ecosystem <https://pytorch.org/ecosystem/>`_ project.

The paper introducing PyPOTS is available on arXiv at `this URL <https://arxiv.org/abs/2305.18811>`_.,
and we are pursuing to publish it in prestigious academic venues, e.g. JMLR (track for
`Machine Learning Open Source Software <https://www.jmlr.org/mloss/>`_). If you use PyPOTS in your work,
please cite it as below and 🌟star `PyPOTS repository <https://github.com/WenjieDu/PyPOTS>`_ to make others notice this library. 🤗

.. code-block:: bibtex
   :linenos:

   @article{du2023PyPOTS,
   title={{PyPOTS: A Python Toolbox for Data Mining on Partially-Observed Time Series}},
   author={Wenjie Du},
   year={2023},
   eprint={2305.18811},
   archivePrefix={arXiv},
   primaryClass={cs.LG},
   url={https://arxiv.org/abs/2305.18811},
   doi={10.48550/arXiv.2305.18811},
   }

or

   Wenjie Du. (2023).
   PyPOTS: A Python Toolbox for Data Mining on Partially-Observed Time Series.
   arXiv, abs/2305.18811. https://doi.org/10.48550/arXiv.2305.18811


❖ Contribution
^^^^^^^^^^^^^^^
You're very welcome to contribute to this exciting project!

By committing your code, you'll

1. make your well-established model out-of-the-box for PyPOTS users to run,
   and help your work obtain more exposure and impact.
   Take a look at our `inclusion criteria <https://docs.pypots.com/en/latest/faq.html#inclusion-criteria>`_.
   You can utilize the ``template`` folder in each task package (e.g.
   `pypots/imputation/template <https://github.com/WenjieDu/PyPOTS/tree/main/pypots/imputation/template>`_) to quickly start;
2. be listed as one of `PyPOTS contributors <https://github.com/WenjieDu/PyPOTS/graphs/contributors>`_:
3. get mentioned in our `release notes <https://github.com/WenjieDu/PyPOTS/releases>`_;

You can also contribute to PyPOTS by simply staring🌟 this repo to help more people notice it.
Your star is your recognition to PyPOTS, and it matters!

The lists of PyPOTS stargazers and forkers are shown below, and we're so proud to have more and more awesome users, as well as more bright ✨stars:

.. image:: https://reporoster.com/stars/dark/WenjieDu/PyPOTS
   :alt: PyPOTS stargazers
   :target: https://github.com/WenjieDu/PyPOTS/stargazers
.. image:: https://reporoster.com/forks/dark/WenjieDu/PyPOTS
   :alt: PyPOTS forkers
   :target: https://github.com/WenjieDu/PyPOTS/network/members

👀 Check out a full list of our users' affiliations `on PyPOTS website here <https://pypots.com/users/>`_ !

❖ Community
^^^^^^^^^^^^
We care about the feedback from our users, so we're building PyPOTS community on

- `Slack <https://join.slack.com/t/pypots-org/shared_invite/zt-1gq6ufwsi-p0OZdW~e9UW_IA4_f1OfxA>`_. General discussion, Q&A, and our development team are here;
- `LinkedIn <https://www.linkedin.com/company/pypots>`_. Official announcements and news are here;
- `WeChat (微信公众号) <https://mp.weixin.qq.com/s/sNgZmgAyxDn2sZxXoWJYMA>`_. We also run a group chat on WeChat,
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
   :caption: Code Documentation

   model_api
   pypots

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   faq
   milestones
   about_us
   references
