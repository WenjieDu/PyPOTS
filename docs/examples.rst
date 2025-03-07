Quick-start Examples
====================

.. image:: https://pypots.com/figs/pypots_logos/BrewPOTS/logo_FFBG.svg
   :width: 160
   :alt: BrewPOTS logo
   :align: right
   :target: https://github.com/WenjieDu/BrewPOTS

We put some examples here to help our users to get started quickly.

Please refer to `BrewPOTS <https://github.com/WenjieDu/BrewPOTS>`_ for detailed PyPOTS tutorials.
You can also find a simple and quick-start tutorial notebook on Google Colab

.. raw:: html

    <a href="https://colab.research.google.com/drive/1HEFjylEy05-r47jRy0H9jiS_WhD0UWmQ" target="_blank"><img src="https://img.shields.io/badge/GoogleColab-PyPOTS_Tutorials-F9AB00?logo=googlecolab&logoColor=white"></a>
    <br clear="right">


.. code-block:: python

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from pygrinder import mcar, calc_missing_rate
    from benchpots.datasets import preprocess_physionet2012

    # prepare the dataset
    data = preprocess_physionet2012(subset='set-a',rate=0.1) # Our ecosystem libs will automatically download and extract it
    train_X, val_X, test_X = data["train_X"], data["val_X"], data["test_X"]
    print(train_X.shape)  # (n_samples, n_steps, n_features)
    print(val_X.shape)  # samples (n_samples) in train set and val set are different, but they have the same sequence len (n_steps) and feature dim (n_features)
    print(f"We have {calc_missing_rate(train_X):.1%} values missing in train_X")

    # organize the dataset for PyPOTS model input
    train_set = {"X": train_X}  # in training set, simply put the incomplete time series into it
    val_set = {
        "X": val_X,
        "X_ori": data["val_X_ori"],  # in validation set, we need ground truth for evaluation and picking the best model checkpoint
    }
    test_set = {"X": test_X}  # in test set, only give the testing incomplete time series for model to impute

    # the test set for final evaluation
    test_X_ori = data["test_X_ori"]  # test_X_ori bears ground truth for evaluation
    indicating_mask = np.isnan(test_X) ^ np.isnan(test_X_ori)  # mask indicates the values that are missing in X but not in X_ori, i.e. where the gt values are

    # initialize the model
    _, n_steps, n_features = train_X.shape
    saits = SAITS(
        n_steps=n_steps,
        n_features=n_features,
        n_layers=2,
        d_model=256,
        d_ffn=128,
        n_heads=4,
        d_k=64,
        d_v=64,
        dropout=0.1,
        epochs=10,
        saving_path="examples/saits", # set the path for saving tensorboard logging file and model checkpoint
        model_saving_strategy="best", # only save the model with the best validation performance
    )

    # train the model. You can also omit the val_set if you don't need to validate the model during training
    saits.fit(train_set, val_set)
    # impute the originally-missing values and artificially-missing values
    imputation = saits.impute(test_set)
    mae = calc_mae(imputation, np.nan_to_num(test_X_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)

    # the best model has been already saved, but you can still manually save it with function save_model() as below
    saits.save(saving_path="examples/saits/manually_saved_saits_model")
    # you can load the saved model into a new initialized model
    saits.load("examples/saits/manually_saved_saits_model.pypots")
