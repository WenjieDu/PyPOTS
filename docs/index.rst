.. PyPOTS documentation master file, created by
   sphinx-quickstart on Wed Oct 19 17:20:43 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===============================
Welcome to PyPOTS doc!
===============================
.. image:: https://raw.githubusercontent.com/WenjieDu/PyPOTS/main/docs/figs/PyPOTS%20logo.svg?sanitize=true
   :height: 180
   :align: right
   :target: https://github.com/WenjieDu/PyPOTS
   :alt: PyPOTS logo

.. centered:: A Python Toolbox for Data Mining on Partially-Observed Time Series

.. image:: https://img.shields.io/badge/python-v3-yellowgreen
   :alt: Python version
.. image:: https://img.shields.io/static/v1?label=%E2%9D%A4%EF%B8%8F&message=PyTorch&color=DC583A
   :alt: PyTorch as backend
.. image:: https://img.shields.io/pypi/v/pypots?color=green&label=PyPI
   :alt: PyPI version
   :target: https://pypi.org/project/pypots
.. image:: https://img.shields.io/badge/License-GPL--v3-green?color=79C641
   :alt: License
   :target: https://github.com/WenjieDu/PyPOTS/blob/main/LICENSE
.. image:: https://github.com/WenjieDu/PyPOTS/actions/workflows/testing.yml/badge.svg
   :alt: GitHub Testing
   :target: https://github.com/WenjieDu/PyPOTS/actions/workflows/testing.yml
.. image:: https://img.shields.io/coverallsCoverage/github/WenjieDu/PyPOTS?branch=main&logo=coveralls&labelColor=3F5767
   :alt: Coveralls report
   :target: https://coveralls.io/github/WenjieDu/PyPOTS
.. image:: https://static.pepy.tech/personalized-badge/pypots?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
   :alt: PyPI download number
   :target: https://pepy.tech/project/pypots
.. image:: https://zenodo.org/badge/475477908.svg
   :alt: Zenodo DOI
   :target: https://zenodo.org/badge/latestdoi/475477908
.. image:: https://img.shields.io/badge/Contributor%20Covenant-v2.1-4baaaa.svg
   :alt: Code of Conduct
   :target: https://github.com/WenjieDu/PyPOTS/blob/main/CODE_OF_CONDUCT.md
.. image:: https://img.shields.io/badge/Slack-PyPOTS-grey?logo=slack&labelColor=4A154B&color=62BCE5
   :alt: Slack workspace
   :target: https://join.slack.com/t/pypots-dev/shared_invite/zt-1gq6ufwsi-p0OZdW~e9UW_IA4_f1OfxA
.. image:: https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FPyPOTS%2FPyPOTS&count_bg=%23009A0A&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visits&edge_flat=false
   :alt: Visiting number


⦿ `Motivation`: Due to all kinds of reasons like failure of collection sensors, communication error, and unexpected malfunction, missing values are common to see in time series from the real-world environment. This makes partially-observed time series (POTS) a pervasive problem in open-world modeling and prevents advanced data analysis. Although this problem is important, the area of data mining on POTS still lacks a dedicated toolkit. PyPOTS is created to fill in this blank.

⦿ `Mission`: PyPOTS is born to become a handy toolbox that is going to make data mining on POTS easy rather than tedious, to help engineers and researchers focus more on the core problems in their hands rather than on how to deal with the missing parts in their data. PyPOTS will keep integrating classical and the latest state-of-the-art data mining algorithms for partially-observed multivariate time series. For sure, besides various algorithms, PyPOTS is going to have unified APIs together with detailed documentation and interactive examples across algorithms as tutorials.

.. image:: https://raw.githubusercontent.com/WenjieDu/TSDB/main/docs/figs/TSDB%20logo.svg?sanitize=true
   :width: 190
   :alt: TSDB
   :align: left
   :target: https://github.com/WenjieDu/TSDB

To make various open-source time-series datasets readily available to our users, PyPOTS gets supported by project `TSDB (Time-Series DataBase) <https://github.com/WenjieDu/TSDB>`_, a toolbox making loading time-series datasets super easy!

Visit `TSDB <https://github.com/WenjieDu/TSDB>`_ right now to know more about this handy tool 🛠! It now supports a total of 119 open-source datasets.


❖ Installation
^^^^^^^^^^^^^^^^
Install the latest release from PyPI:

   pip install pypots

Below is an example applying SAITS in PyPOTS to impute missing values in the dataset PhysioNet2012:

.. code-block:: python
   :linenos:

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
   # Model training. This is PyPOTS showtime. 💪
   saits = SAITS(n_steps=48, n_features=37, n_layers=2, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=10)
   saits.fit(X)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
   imputation = saits.impute(X)  # impute the originally-missing values and artificially-missing values
   mae = cal_mae(imputation, X_intact, indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)

❖ Available Algorithms
^^^^^^^^^^^^^^^^^^^^^^^
============================== ================ =========================================================================  ====== =========
Task                           Type             Algorithm                                                                  Year   Reference
============================== ================ =========================================================================  ====== =========
Imputation                     Neural Network   SAITS (Self-Attention-based Imputation for Time Series)                    2022   :cite:`du2022SAITS`
Imputation                     Neural Network   Transformer                                                                2017   :cite:`vaswani2017Transformer`, :cite:`du2022SAITS`
Imputation, Classification     Neural Network   BRITS (Bidirectional Recurrent Imputation for Time Series)                 2018   :cite:`cao2018BRITS`
Imputation                     Naive            LOCF (Last Observation Carried Forward)                                    /      /
Classification                 Neural Network   GRU-D                                                                      2018   :cite:`che2018GRUD`
Classification                 Neural Network   Raindrop                                                                   2022   :cite:`zhang2022Raindrop`
Clustering                     Neural Network   CRLI (Clustering Representation Learning on Incomplete time-series data)   2021   :cite:`ma2021CRLI`
Clustering                     Neural Network   VaDER (Variational Deep Embedding with Recurrence)                         2019   :cite:`dejong2019VaDER`
Forecasting                    Probabilistic    BTTF (Bayesian Temporal Tensor Factorization)                              2021   :cite:`chen2021BTMF`
============================== ================ =========================================================================  ====== =========

❖ Citing PyPOTS
^^^^^^^^^^^^^^^^
If you find PyPOTS is helpful to your research, please cite it as below and ⭐️star this repository to make others notice this work. 🤗

.. code-block:: bibtex
   :linenos:

   @misc{du2022PyPOTS,
   author = {Wenjie Du},
   title = {{PyPOTS: A Python Toolbox for Data Mining on Partially-Observed Time Series}},
   howpublished = {\url{https://github.com/wenjiedu/pypots}},
   year = {2022},
   doi = {10.5281/zenodo.6823222},
   }

or

   Wenjie Du. (2022). PyPOTS: A Python Toolbox for Data Mining on Partially-Observed Time Series. Zenodo. https://doi.org/10.5281/zenodo.6823222

❖ Attention 👀
^^^^^^^^^^^^^^^
The documentation and tutorials are under construction. And a short paper introducing PyPOTS is on the way! 🚀 Stay tuned please!

‼️ PyPOTS is currently under developing. If you like it and look forward to its growth, **please give PyPOTS a star and watch it to keep you posted on its progress and to let me know that its development is meaningful**. If you have any feedback, or want to contribute ideas/suggestions or share time-series related algorithms/papers, please join `our PyPOTS community <https://join.slack.com/t/pypots-dev/shared_invite/zt-1gq6ufwsi-p0OZdW~e9UW_IA4_f1OfxA>`_ or create an issue. If you have any additional questions or have interests in collaboration, please take a look at `my GitHub profile <https://github.com/WenjieDu>`_ and feel free to contact me 🤝.

Thank you all for your attention! 😃


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   install
   examples

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Code Documentation

   pypots

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   faq
   about_us


References
""""""""""
.. bibliography::