Quick-Start Tutorials
=====================

.. image:: https://pypots.com/figs/pypots_logos/BrewPOTS/logo_FFBG.svg
   :width: 160
   :alt: BrewPOTS logo
   :align: right
   :target: https://github.com/WenjieDu/BrewPOTS

Explore `BrewPOTS <https://github.com/WenjieDu/BrewPOTS>`__ for detailed PyPOTS
tutorials and notebooks, or start with the
`Google Colab tutorial <https://colab.research.google.com/drive/1HEFjylEy05-r47jRy0H9jiS_WhD0UWmQ>`__.

For a comprehensive hands-on introduction, explore our
`KDD 2026 Tutorial: End-to-End Learning for Partially-Observed Time Series with PyPOTS <https://pypots.com/events/KDD2026Tutorial/>`__.

Quick start with synthetic data
-------------------------------

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

Real-world example: PhysioNet2012
---------------------------------

This example demonstrates training, validation, testing, and imputation evaluation.
The ecosystem libraries automatically download and preprocess the dataset.

.. code-block:: python

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

Explore PyPOTS by Task
----------------------

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

See :doc:`install` for setup and :doc:`distributed` for distributed training.
