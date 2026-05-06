.. PyPOTS developer documentation - Standard NN Integration Path

Standard NN Integration Path
================================

Use this path when **one optimizer** and the **default** ``BaseNNModel._train_model()`` are enough.

``SAITS`` is the best reference model for this path.


When This Path Is Correct
----------------------------

Choose the standard NN path when:

- One optimizer is enough
- One main objective drives training
- You can express training and validation with ``results["loss"]`` and ``results["metric"]``
- You do not need alternating update schedules

If any of these are not true, switch early to :doc:`dev_complex_nn`.


Start From the Task Template
-------------------------------

PyPOTS ships task templates to help you get started:

.. code-block:: text

   pypots/imputation/template/
   pypots/forecasting/template/
   pypots/classification/template/
   pypots/clustering/template/

Use them as **scaffolding**, not as the final spec.
The real contract comes from the task base class.
Do not copy placeholder output names blindly.


Step-by-Step Implementation Guide
====================================


Step 1: Pick the Task Contract
---------------------------------

Before writing code, decide:

- Which **task base class** you inherit (e.g. ``BaseNNImputer``)
- Which **public helper method** must work (e.g. ``impute()``, ``forecast()``, ``classify()``)
- Which **public result key** must exist (e.g. ``"imputation"`` for imputation models)

For example, an imputation model must end up with ``"imputation"`` in the dict returned by ``predict()``.


Step 2: Implement ``core.py``
---------------------------------

Your core should focus on **model computation only**.
For the standard NN path, ``forward()`` follows this pattern:

1. Read tensors from ``inputs`` dict
2. Compute the model output
3. Return the task result key (e.g. ``"imputation"``)
4. When ``calc_criterion=True``, also return ``"loss"`` (training) or ``"metric"`` (validation)

Here is a complete example based on the SAITS pattern:

.. code-block:: python

   # pypots/imputation/your_model/core.py

   import torch.nn as nn
   from ...nn.modules import ModelCore
   from ...nn.modules.loss import Criterion

   class _YourModel(ModelCore):
       def __init__(
           self,
           n_steps: int,
           n_features: int,
           d_model: int,
           training_loss: Criterion,
           validation_metric: Criterion,
       ):
           super().__init__()
           self.training_loss = training_loss
           if validation_metric.__class__.__name__ == "Criterion":
               self.validation_metric = self.training_loss
           else:
               self.validation_metric = validation_metric

           # Define your model's components
           self.embedding = nn.Linear(n_features, d_model)
           self.backbone = nn.TransformerEncoder(
               nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model * 4),
               num_layers=2,
           )
           self.output_proj = nn.Linear(d_model, n_features)

       def forward(self, inputs: dict, calc_criterion: bool = False) -> dict:
           X, missing_mask = inputs["X"], inputs["missing_mask"]

           # Model computation
           embedded = self.embedding(X)
           encoded = self.backbone(embedded)
           reconstruction = self.output_proj(encoded)

           # Combine: keep observed values, fill missing with model output
           imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction

           results = {
               "imputation": imputed_data,
               "reconstruction": reconstruction,
           }

           # Loss / metric computation
           if calc_criterion:
               X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
               if self.training:
                   # Training: return "loss" for backpropagation
                   results["loss"] = self.training_loss(
                       reconstruction, X_ori, indicating_mask
                   )
               else:
                   # Validation: return "metric" for model selection
                   results["metric"] = self.validation_metric(
                       reconstruction, X_ori, indicating_mask
                   )

           return results


Step 3: Implement ``model.py``
---------------------------------

Your wrapper owns **orchestration**. For a standard NN model, it should do five jobs:

1. Inherit the correct task NN base
2. Instantiate the core model
3. Instantiate and initialize the optimizer
4. Implement ``_assemble_input_for_training()``, ``_assemble_input_for_validating()``, and ``_assemble_input_for_testing()``
5. Build datasets and dataloaders in ``fit()``, then call ``_train_model()``

Here is a complete example:

.. code-block:: python

   # pypots/imputation/your_model/model.py

   from typing import Union, Optional
   import numpy as np
   import torch
   from torch.utils.data import DataLoader

   from .core import _YourModel
   from ..base import BaseNNImputer
   from ...data.dataset.base import BaseDataset
   from ...data.checking import key_in_data_set
   from ...nn.modules.loss import Criterion, MAE, MSE
   from ...optim.adam import Adam
   from ...optim.base import Optimizer


   class YourModel(BaseNNImputer):
       """Your model description here.

       Parameters
       ----------
       n_steps :
           The number of time steps in the time-series data sample.

       n_features :
           The number of features in the time-series data sample.

       d_model :
           The dimension of the model's backbone.

       batch_size :
           The batch size for training and evaluating the model.

       epochs :
           The number of epochs for training the model.

       patience :
           The patience for the early-stopping mechanism.

       training_loss :
           The loss function for training. Default: MAE.

       validation_metric :
           The metric function for validation. Default: MSE.

       optimizer :
           The optimizer for model training. Default: Adam.
       """

       def __init__(
           self,
           n_steps: int,
           n_features: int,
           d_model: int = 64,
           batch_size: int = 32,
           epochs: int = 100,
           patience: Optional[int] = None,
           training_loss: Union[Criterion, type] = MAE,
           validation_metric: Union[Criterion, type] = MSE,
           optimizer: Union[Optimizer, type] = Adam,
           num_workers: int = 0,
           device: Optional[Union[str, torch.device, list]] = None,
           saving_path: Optional[str] = None,
           model_saving_strategy: Optional[str] = "best",
           verbose: bool = True,
       ):
           super().__init__(
               training_loss=training_loss,
               validation_metric=validation_metric,
               batch_size=batch_size,
               epochs=epochs,
               patience=patience,
               num_workers=num_workers,
               device=device,
               saving_path=saving_path,
               model_saving_strategy=model_saving_strategy,
               verbose=verbose,
           )

           # Store hyperparameters
           self.n_steps = n_steps
           self.n_features = n_features
           self.d_model = d_model

           # Set up the model
           self.model = _YourModel(
               n_steps=n_steps,
               n_features=n_features,
               d_model=d_model,
               training_loss=self.training_loss,
               validation_metric=self.validation_metric,
           )
           self._print_model_size()
           self._send_model_to_given_device()

           # Set up the optimizer
           if isinstance(optimizer, Optimizer):
               self.optimizer = optimizer
           else:
               self.optimizer = optimizer()
               assert isinstance(self.optimizer, Optimizer)
           self.optimizer.init_optimizer(self.model.parameters())

       def _assemble_input_for_training(self, data: list) -> dict:
           (
               indices,
               X,
               missing_mask,
               X_ori,
               indicating_mask,
           ) = self._send_data_to_given_device(data)

           inputs = {
               "X": X,
               "missing_mask": missing_mask,
               "X_ori": X_ori,
               "indicating_mask": indicating_mask,
           }
           return inputs

       def _assemble_input_for_validating(self, data: list) -> dict:
           return self._assemble_input_for_training(data)

       def _assemble_input_for_testing(self, data: list) -> dict:
           indices, X, missing_mask = self._send_data_to_given_device(data)
           inputs = {
               "X": X,
               "missing_mask": missing_mask,
           }
           return inputs

       def fit(
           self,
           train_set: Union[dict, str],
           val_set: Optional[Union[dict, str]] = None,
           file_type: str = "hdf5",
       ) -> None:
           # Step 1: Create datasets and dataloaders
           training_set = DatasetForYourModel(
               train_set, return_X_ori=False, return_y=False, file_type=file_type
           )
           train_dataloader = DataLoader(
               training_set,
               batch_size=self.batch_size,
               shuffle=True,
               num_workers=self.num_workers,
           )

           val_dataloader = None
           if val_set is not None:
               if not key_in_data_set("X_ori", val_set):
                   raise ValueError("val_set must contain 'X_ori' for validation.")
               val_dataset = DatasetForYourModel(
                   val_set, return_X_ori=True, return_y=False, file_type=file_type
               )
               val_dataloader = DataLoader(
                   val_dataset,
                   batch_size=self.batch_size,
                   shuffle=False,
                   num_workers=self.num_workers,
               )

           # Step 2: Train the model
           self._train_model(train_dataloader, val_dataloader)
           self.model.load_state_dict(self.best_model_dict)

           # Step 3: Auto-save if configured
           self._auto_save_model_if_necessary(
               confirm_saving=self.model_saving_strategy == "best"
           )

       def predict(
           self,
           test_set: Union[dict, str],
           file_type: str = "hdf5",
       ) -> dict:
           result_dict = super().predict(test_set, file_type)
           return result_dict

       def impute(
           self,
           test_set: Union[dict, str],
           file_type: str = "hdf5",
       ) -> np.ndarray:
           results = super().impute(test_set, file_type)
           return results


Step 4: Add ``data.py`` Only If Needed
-----------------------------------------

Add ``data.py`` only when ``BaseDataset`` cannot express your model's sample contract.

``SAITS`` needs ``data.py`` because masked-imputation training requires artificial masking
that ``BaseDataset`` does not provide.

If your model can work with ``BaseDataset`` directly (or reuse another model's dataset
like ``DatasetForBRITS``), do **not** add extra dataset code.


Step 5: Wire the Package
---------------------------

Create the ``__init__.py`` to export your model:

.. code-block:: python

   # pypots/imputation/your_model/__init__.py

   from .model import YourModel

   __all__ = ["YourModel"]

Then add the import to the task package's ``__init__.py``:

.. code-block:: python

   # In pypots/imputation/__init__.py, add:
   from .your_model import YourModel


Step 6: Keep ``predict()`` Boring
------------------------------------

The best ``predict()`` is usually a thin wrapper over the task base implementation.

``SAITS.predict()`` is a good example — it keeps the public API explicit,
passes through inference-time options, and reuses ``BaseNNImputer.predict()`` for the actual loop.


SAITS Walkthrough Summary
----------------------------

Read ``SAITS`` in this order for the full picture:

1. ``model.py``: wrapper, optimizer, dataloaders, input assembly
2. ``core.py``: forward contract and loss/metric outputs
3. ``data.py``: why a custom dataset exists

Key things to copy from ``SAITS``:

- Wrapper and core responsibilities stay separate
- Stage-specific input assembly is explicit
- Validation requirements are checked early (e.g. ``X_ori`` must exist in ``val_set``)
- Best-checkpoint loading happens after training


Definition of Done
---------------------

Your standard NN integration is done when **all** of these are true:

- ``fit()`` runs without overriding ``_train_model()``
- Training returns ``"loss"`` and validation returns ``"metric"``
- ``predict()`` returns the correct task result key and shape
- Save/load still works
- Targeted task tests pass

If you keep fighting the default training loop, you are probably no longer on the standard path.
Switch to :doc:`dev_complex_nn`.
