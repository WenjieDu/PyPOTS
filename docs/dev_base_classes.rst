.. PyPOTS developer documentation - Base Classes and Inheritance

Base Classes and Inheritance
==============================

Read PyPOTS as a stack of contracts.
If you understand which layer owns which responsibility, adding a model becomes much simpler.


The Inheritance Stack
-----------------------

.. code-block:: text

   BaseModel                         ← shared shell for all models
     ├── BaseNNModel                 ← training-loop contract for NN models
     │     ├── BaseNNImputer         ← NN imputation models
     │     ├── BaseNNForecaster      ← NN forecasting models
     │     ├── BaseNNClassifier      ← NN classification models
     │     ├── BaseNNDetector        ← NN anomaly detection models
     │     ├── BaseNNClusterer       ← NN clustering models
     │     └── BaseNNRepresentor     ← NN representation learning models
     ├── BaseImputer                 ← non-NN imputation models
     ├── BaseForecaster              ← non-NN forecasting models
     ├── BaseClassifier              ← non-NN classification models
     ├── BaseDetector                ← non-NN anomaly detection models
     ├── BaseClusterer               ← non-NN clustering models
     └── BaseRepresentor             ← non-NN representation learning models

On the NN core side:

.. code-block:: text

   torch.nn.Module
     └── ModelCore                   ← base for all NN model cores
           └── _SAITS, _BRITS, ...  ← concrete model cores


What Each Layer Owns
----------------------

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Layer
     - Owns
     - You Usually Implement
   * - ``BaseModel``
     - Device setup, AMP switch, saving path, checkpoint IO, abstract public API
     - ``fit()``, ``predict()``
   * - ``BaseNNModel``
     - Training loop state, early stopping, best checkpoint tracking, TensorBoard logging, data-to-device helper
     - ``_assemble_input_*``, wrapper ``fit()``, wrapper ``predict()``, sometimes ``_train_model()``
   * - Task NN base
     - Task semantics: public result key and helper methods
     - Task-specific wrapper behavior
   * - ``BaseDataset``
     - Array/file input normalization, missing-mask generation, optional ``X_ori``/``X_pred``/``y`` loading
     - Custom dataset only if the default sample is not enough


``BaseModel``: The Outer Shell
---------------------------------

``BaseModel`` (defined in ``pypots/base.py``) is the shared shell for **all** models — both NN and non-NN.

It owns:

- **Device selection**: CPU, CUDA, multi-GPU via ``DataParallel``
- **AMP enablement**: Automatic mixed precision
- **Checkpoint path setup**: Auto-creates directories for saving
- **Save/load helpers**: ``save()`` and ``load()`` methods
- **Abstract** ``fit()`` **and** ``predict()``: These must be implemented by every model

.. code-block:: python

   from pypots.base import BaseModel

   class MyModel(BaseModel):
       def __init__(self, device=None, saving_path=None):
           super().__init__(device=device, saving_path=saving_path)

       def fit(self, train_set, val_set=None, file_type="hdf5"):
           # Your training logic
           ...

       def predict(self, test_set, file_type="hdf5"):
           # Your inference logic
           ...

**Do not** put task math here. **Do not** put optimizer stepping here.


``BaseNNModel``: The Training-Loop Contract
----------------------------------------------

``BaseNNModel`` (also in ``pypots/base.py``) is the layer that standard NN models reuse.
It extends ``BaseModel`` with training-specific behavior.

Key Attributes
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Set in __init__
   self.batch_size       # Batch size for DataLoader
   self.epochs           # Number of training epochs
   self.patience         # Early stopping patience (None = disabled)
   self.training_loss    # Criterion for training loss
   self.validation_metric # Criterion for validation metric
   self.num_workers      # DataLoader subprocesses
   self.best_model_dict  # State dict of the best model
   self.best_loss        # Best validation loss seen so far
   self.best_epoch       # Epoch of the best loss

The Default ``_train_model()`` Contract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default ``_train_model()`` follows a strict contract:

1. **Training**: calls ``_assemble_input_for_training(data)`` to build the input dict
2. **Forward pass**: passes the assembled dict into ``self.model(inputs, calc_criterion=True)``
3. **Training mode**: the returned dict **must** contain ``results["loss"]``
4. **Validation**: calls ``_assemble_input_for_validating(data)``
5. **Validation mode**: the returned dict **must** contain ``results["metric"]``
6. **Checkpointing**: best-checkpoint tracking, patience reset, and early stopping are handled automatically

This means two things for contributors:

- If you use the **standard NN path**, your core ``forward()`` must produce the keys that ``_train_model()`` expects.
- If you **override** ``_train_model()``, you must preserve best-model selection and patience semantics
  unless you have a very strong reason not to.


``ModelCore``: The NN Core Base
----------------------------------

``ModelCore`` (defined in ``pypots/nn/modules/base_model_core.py``) is the base class for all NN model cores.
It extends ``torch.nn.Module`` and defines the ``forward()`` contract:

.. code-block:: python

   from pypots.nn.modules import ModelCore
   from pypots.nn.modules.loss import Criterion

   class _MyModel(ModelCore):
       def __init__(self, training_loss: Criterion, validation_metric: Criterion):
           super().__init__()
           self.training_loss = training_loss
           self.validation_metric = validation_metric
           # Define your model's components here
           ...

       def forward(self, inputs: dict, calc_criterion: bool = False) -> dict:
           X, missing_mask = inputs["X"], inputs["missing_mask"]

           # Your model computation
           reconstruction = ...

           imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
           results = {
               "imputation": imputed_data,
               "reconstruction": reconstruction,
           }

           if calc_criterion:
               X_ori = inputs["X_ori"]
               indicating_mask = inputs["indicating_mask"]
               if self.training:
                   results["loss"] = self.training_loss(
                       reconstruction, X_ori, indicating_mask
                   )
               else:
                   results["metric"] = self.validation_metric(
                       reconstruction, X_ori, indicating_mask
                   )

           return results

The key rules for ``forward()``:

- **Input**: always a ``dict`` (assembled by the wrapper's ``_assemble_input_*`` methods)
- **Output**: always a ``dict``
- When ``calc_criterion=True`` and in training mode → include ``"loss"`` key
- When ``calc_criterion=True`` and in eval mode → include ``"metric"`` key
- Always include the task result key (e.g. ``"imputation"``)


``BaseDataset``: The Default Sample Contract
------------------------------------------------

``BaseDataset`` (in ``pypots/data/dataset/base.py``) supports both in-memory dict input
and file-backed lazy loading. It produces samples in a stable order:

.. code-block:: text

   Base items:    [idx, X, missing_mask]

   Optional items (appended in this order):
   +----------------------------+---------------------------------------+
   | Flag                       | Extra Items                           |
   +============================+=======================================+
   | return_X_ori=True          | X_ori, indicating_mask                |
   +----------------------------+---------------------------------------+
   | return_X_pred=True         | X_pred, X_pred_missing_mask           |
   +----------------------------+---------------------------------------+
   | return_y=True              | y                                     |
   +----------------------------+---------------------------------------+

So the fullest sample looks like:

.. code-block:: text

   [idx, X, missing_mask, X_ori, indicating_mask, X_pred, X_pred_missing_mask, y]

Your wrapper's ``_assemble_input_*`` methods are responsible for turning that list into
the dict expected by ``forward()``.

Array input and file-backed input follow the same logical order.
Do not let the two modes drift apart.


Task-Specific NN Differences
-------------------------------

Not every task base gives you the same amount of help.

.. list-table::
   :header-rows: 1
   :widths: 16 18 18 28 20

   * - Task
     - NN Base
     - Public Helper
     - Result Contract
     - Extra Arg
   * - Imputation
     - ``BaseNNImputer``
     - ``impute()``
     - ``predict()`` returns ``"imputation"``
     - —
   * - Forecasting
     - ``BaseNNForecaster``
     - ``forecast()``
     - ``predict()`` returns ``"forecasting"``
     - —
   * - Classification
     - ``BaseNNClassifier``
     - ``classify()`` / ``predict_proba()``
     - Result includes ``"classification"`` and ``"classification_proba"``
     - ``n_classes``
   * - Anomaly Detection
     - ``BaseNNDetector``
     - ``detect()``
     - Result includes ``"anomaly_detection"``
     - ``anomaly_rate``
   * - Clustering
     - ``BaseNNClusterer``
     - ``cluster()``
     - Result includes ``"clustering"``
     - ``n_clusters``
   * - Representation
     - ``BaseNNRepresentor``
     - ``represent()``
     - ``predict()`` returns ``"representation"``
     - —

Important differences:

- ``BaseNNForecaster`` and ``BaseNNClassifier`` already provide default ``_assemble_input_*`` helpers.
- ``BaseNNImputer``, ``BaseNNDetector``, ``BaseNNClusterer``, and ``BaseNNRepresentor`` rely more on
  the concrete model wrapper to implement assembly.
- ``BaseNNClusterer`` is especially open-ended: its ``fit()`` and ``predict()`` stay abstract in the task base.


Example Mapping
-----------------

- ``SAITS`` — standard NN imputer built on ``BaseNNImputer``
- ``USGAN`` — complex NN imputer that still inherits ``BaseNNImputer`` but overrides ``_train_model()``
- ``LOCF`` — non-NN imputer built on ``BaseImputer``

That is the main design choice in PyPOTS:
**first choose the contract layer, then write the model**.
