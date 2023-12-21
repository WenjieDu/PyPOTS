Quick-start Examples
====================

.. image:: https://pypots.com/figs/pypots_logos/BrewPOTS_logo_FFBG.svg?sanitize=true
   :width: 160
   :alt: BrewPOTS logo
   :align: right
   :target: https://github.com/WenjieDu/BrewPOTS

We put some examples here to help our users to get started quickly.

Please refer to `BrewPOTS <https://github.com/WenjieDu/BrewPOTS>`_ for detailed PyPOTS tutorials.
You can also find a simple and quick-start tutorial notebook on Google Colab with
`this link <https://colab.research.google.com/drive/1HEFjylEy05-r47jRy0H9jiS_WhD0UWmQ?usp=sharing>`_.

.. raw:: html

    <br clear="right">


.. code-block:: python

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from pygrinder import mcar
    from pypots.data import load_specific_dataset
    from pypots.imputation import SAITS
    from pypots.utils.metrics import calc_mae

    # Data preprocessing. Tedious, but PyPOTS can help. 🤓
    data = load_specific_dataset('physionet_2012')  # PyPOTS will automatically download and extract it.
    X = data['X']
    num_samples = len(X['RecordID'].unique())
    X = X.drop(['RecordID', 'Time'], axis = 1)
    X = StandardScaler().fit_transform(X.to_numpy())
    X = X.reshape(num_samples, 48, -1)
    X_ori = X  # keep X_ori for validation
    X = mcar(X, 0.1)  # randomly hold out 10% observed values as ground truth
    dataset = {"X": X}  # X for model input
    print(X.shape)  # (11988, 48, 37), 11988 samples, 48 time steps, 37 features

    # initialize the model
    saits = SAITS(
        n_steps=48,
        n_features=37,
        n_layers=2,
        d_model=256,
        d_inner=128,
        n_heads=4,
        d_k=64,
        d_v=64,
        dropout=0.1,
        epochs=10,
        saving_path="examples/saits", # set the path for saving tensorboard logging file and model checkpoint
        model_saving_strategy="best", # only save the model with the best validation performance
    )

    # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
    saits.fit(dataset)
    # impute the originally-missing values and artificially-missing values
    imputation = saits.impute(dataset)
    # calculate mean absolute error on the ground truth (artificially-missing values)
    indicating_mask = np.isnan(X) ^ np.isnan(X_ori)  # indicating mask for imputation error calculation
    mae = calc_mae(imputation, np.nan_to_num(X_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)

    # the best model has been already saved, but you can still manually save it with function save_model() as below
    saits.save_model(saving_dir="examples/saits",file_name="manually_saved_saits_model")
    # you can load the saved model into a new initialized model
    saits.load_model("examples/saits/manually_saved_saits_model")
