.. PyPOTS developer documentation - Non-NN Integration Path

Non-NN Integration Path
=========================

Use this path for models that should **not** use ``BaseNNModel`` at all.

``LOCF`` is the cleanest example.


When This Path Is Correct
----------------------------

Choose the non-NN path when:

- There is no gradient-based training loop
- There is no optimizer
- The model is rule-based, statistical, or algorithmic
- Wrapping it in a neural-network base class would add fake complexity

Good examples in PyPOTS:

- ``LOCF`` (Last Observed Carried Forward)
- ``Mean`` (fill with mean values)
- ``Median`` (fill with median values)
- ``Lerp`` (linear interpolation)
- ``TRMF`` (Temporal Regularized Matrix Factorization)
- ``BTTF`` (Bayesian Temporal Tensor Factorization)


``LOCF`` as the Reference Pattern
--------------------------------------

``LOCF`` inherits ``BaseImputer``, **not** ``BaseNNImputer``.

That choice immediately removes:

- Optimizer setup
- ``_train_model()``
- ``_assemble_input_*`` hooks
- Checkpoint-selection logic tied to NN training

Its implementation is direct and clean:

.. code-block:: python

   # pypots/imputation/locf/model.py (simplified)

   import warnings
   from typing import Union, Optional

   import h5py
   import numpy as np
   import torch

   from .core import locf_numpy, locf_torch
   from ..base import BaseImputer


   class LOCF(BaseImputer):
       """LOCF imputation: fills missing values with the last observed value.

       Parameters
       ----------
       first_step_imputation : str, default='zero'
           Strategy for imputing missing values at the beginning of sequences.
           Can be 'backward', 'zero', 'median', or 'nan'.
       """

       def __init__(
           self,
           first_step_imputation: str = "zero",
           device: Optional[Union[str, torch.device, list]] = None,
       ):
           super().__init__(device=device)
           assert first_step_imputation in ["nan", "zero", "backward", "median"]
           self.first_step_imputation = first_step_imputation

       def fit(
           self,
           train_set: Union[dict, str],
           val_set: Optional[Union[dict, str]] = None,
           file_type: str = "hdf5",
       ) -> None:
           """LOCF does not need training. Issues a warning."""
           warnings.warn(
               "LOCF has no parameter to train. "
               "Please run func `predict()` directly."
           )

       def predict(
           self,
           test_set: Union[dict, str],
           file_type: str = "hdf5",
           **kwargs,
       ) -> dict:
           # Handle both dict and file input
           if isinstance(test_set, str):
               with h5py.File(test_set, "r") as f:
                   X = f["X"][:]
           else:
               X = test_set["X"]

           assert len(X.shape) == 3, (
               f"Input X should have 3 dimensions "
               f"[n_samples, n_steps, n_features], "
               f"but got shape: {X.shape}"
           )

           if isinstance(X, np.ndarray):
               imputed_data = locf_numpy(X, self.first_step_imputation)
           elif isinstance(X, torch.Tensor):
               imputed_data = locf_torch(X, self.first_step_imputation)

           result_dict = {
               "imputation": imputed_data,
           }
           return result_dict

This is exactly what a non-NN wrapper should look like:
clean, explicit, and contract-driven.


Two Valid Non-NN Styles
--------------------------

Stateless Models
^^^^^^^^^^^^^^^^^^

Examples: ``LOCF``, ``Mean``, ``Median``, ``Lerp``

These models do not learn parameters from data.
``fit()`` is an explicit no-op with a warning.

.. code-block:: python

   class StatelessModel(BaseImputer):
       def fit(self, train_set, val_set=None, file_type="hdf5"):
           warnings.warn("This model has no parameters to train.")

       def predict(self, test_set, file_type="hdf5", **kwargs):
           X = test_set["X"]
           imputed_data = self._apply_algorithm(X)
           return {"imputation": imputed_data}


Stateful Models
^^^^^^^^^^^^^^^^^

Examples: ``TRMF``, ``BTTF``

These models still do **not** use the NN training loop, but they do learn
algorithm state in ``fit()``.

.. code-block:: python

   class StatefulModel(BaseImputer):
       def fit(self, train_set, val_set=None, file_type="hdf5"):
           X = train_set["X"]
           # Learn parameters from training data
           self.learned_params = self._fit_algorithm(X)

       def predict(self, test_set, file_type="hdf5", **kwargs):
           X = test_set["X"]
           imputed_data = self._apply_algorithm(X, self.learned_params)
           return {"imputation": imputed_data}

In both cases, the public contract is the same:
``predict()`` must return the task-level result key (e.g. ``"imputation"``).


Step-by-Step Implementation Guide
====================================


Step 1: Choose the Base Class
--------------------------------

Inherit the correct non-NN task base:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Task
     - Non-NN Base
     - Result Key
   * - Imputation
     - ``BaseImputer``
     - ``"imputation"``
   * - Forecasting
     - ``BaseForecaster``
     - ``"forecasting"``
   * - Classification
     - ``BaseClassifier``
     - ``"classification"``
   * - Anomaly Detection
     - ``BaseDetector``
     - ``"anomaly_detection"``
   * - Clustering
     - ``BaseClusterer``
     - ``"clustering"``


Step 2: Implement ``fit()``
------------------------------

Be explicit, even if it only warns:

.. code-block:: python

   def fit(self, train_set, val_set=None, file_type="hdf5"):
       """Train the model. For stateless models, this is a no-op."""
       warnings.warn("This model has no parameters to train.")


Step 3: Implement ``predict()``
----------------------------------

Keep it simple and contract-driven:

.. code-block:: python

   def predict(self, test_set, file_type="hdf5", **kwargs):
       # Handle both dict and file input
       if isinstance(test_set, str):
           with h5py.File(test_set, "r") as f:
               X = f["X"][:]
       else:
           X = test_set["X"]

       # Validate input shape
       assert len(X.shape) == 3, (
           f"Input X should have 3 dimensions, got {X.shape}"
       )

       # Apply your algorithm
       imputed_data = your_algorithm(X)

       return {"imputation": imputed_data}


Step 4: Implement Helper Methods
-----------------------------------

Make helper methods like ``impute()`` or ``forecast()`` return the raw array users expect:

.. code-block:: python

   def impute(self, test_set, file_type="hdf5", **kwargs):
       result = self.predict(test_set, file_type, **kwargs)
       return result["imputation"]


Step 5: Wire the Package
---------------------------

Same as the standard NN path:

.. code-block:: python

   # pypots/imputation/your_model/__init__.py
   from .model import YourModel
   __all__ = ["YourModel"]


Common Mistake
-----------------

Do **not** force a non-NN model into ``BaseNNModel`` just because most folders around it are neural models.

That usually creates:

- Fake hooks that do nothing
- Fake optimizers that are never used
- Confusing tests with unnecessary training loops
- Review confusion for maintainers

If there is no gradient, there should be no ``BaseNNModel``.


Definition of Done
---------------------

Your non-NN integration is done when:

- The chosen base class matches the real algorithm
- ``fit()`` behavior is explicit (even if it's a no-op)
- ``predict()`` returns the correct task result key
- Helper methods return the expected array
- Targeted tests cover the advertised input modes (both dict and file input)
