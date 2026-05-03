.. PyPOTS developer documentation - Data Flow and Dataset

Data Flow and Dataset Contracts
==================================

Most integration bugs in PyPOTS happen at one boundary:

.. code-block:: text

   Dataset sample list → Wrapper input dict → Core forward()

This page makes that boundary explicit.


The Normal Data Flow
-----------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────┐
   │  1. Dataset.__getitem__(idx)   → returns a sample list             │
   │  2. DataLoader                 → batches those lists               │
   │  3. Wrapper._assemble_input_*  → turns list into a dict           │
   │  4. Core.forward(inputs)       → reads the dict, does computation  │
   │  5. Wrapper / task base        → returns task-level result         │
   └─────────────────────────────────────────────────────────────────────┘

If one stage uses the wrong keys or the wrong shape, the failure usually appears
at the boundary between stages 3 and 4.


What ``BaseDataset`` Returns
------------------------------

``BaseDataset`` (in ``pypots/data/dataset/base.py``) always starts from the same base items:

.. code-block:: text

   [idx, X, missing_mask]

Then it appends optional items based on flags:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Flag
     - Extra Items
     - Why They Exist
   * - ``return_X_ori=True``
     - ``X_ori``, ``indicating_mask``
     - For models that need original targets or artificial-missing masks
   * - ``return_X_pred=True``
     - ``X_pred``, ``X_pred_missing_mask``
     - For forecasting targets
   * - ``return_y=True``
     - ``y``
     - For classification or other supervised outputs

Array input and file-backed input follow the same logical order.


What Each Stage Should Assemble
---------------------------------

The three assembly functions exist because **train, validation, and test do not always need the same tensors**.

Typical pattern:

- **Training**: include everything needed for loss computation
- **Validation**: include everything needed for metric computation
- **Testing**: keep only inference-time inputs

This is why one model may need more tensors in ``fit()`` than in ``predict()``.


SAITS: A Concrete Data Flow Example
--------------------------------------

``SAITS`` is the clearest reference for a standard NN model with extra data requirements.

Custom Dataset: ``DatasetForSAITS``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``DatasetForSAITS`` extends ``BaseDataset`` because the default dataset is not enough.
It introduces **Masked Imputation Training (MIT)** — artificially masking a portion of
observed values to create a self-supervised training signal.

.. code-block:: python

   from pypots.data.dataset.base import BaseDataset
   from pygrinder import mcar

   class DatasetForSAITS(BaseDataset):
       def __init__(self, data, return_X_ori, return_y, file_type="hdf5", rate=0.2):
           super().__init__(data, return_X_ori=return_X_ori, return_y=return_y,
                            file_type=file_type)
           self.rate = rate  # artificial masking rate for MIT

       def _fetch_data_from_array(self, idx):
           # Get original data
           X = self.X[idx]
           missing_mask = self.missing_mask[idx]
           X_ori = self.X_ori[idx]
           indicating_mask = self.indicating_mask[idx]

           # Apply additional artificial masking for MIT
           X_hat, missing_mask_hat = mcar(X, rate=self.rate)
           indicating_mask_hat = missing_mask - missing_mask_hat

           return (
               idx,
               X_hat,             # masked input fed to model
               missing_mask_hat,  # observed vs missing in X_hat
               X_ori,             # original targets for loss
               indicating_mask_hat # artificially hidden positions
           )

This shows a clean reason to add a custom dataset:
training needs artificial masking, validation needs original targets, testing should stay minimal.


Training and Validation Assembly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``SAITS._assemble_input_for_training()`` builds this dict:

.. code-block:: python

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

Inside ``_SAITS.forward()``, ``X`` and ``missing_mask`` are always used for the model pass.
When ``calc_criterion=True``, ``X_ori`` and ``indicating_mask`` are used to produce
training loss or validation metric.


Testing Assembly
^^^^^^^^^^^^^^^^^

``SAITS._assemble_input_for_testing()`` intentionally drops the extra tensors:

.. code-block:: python

   def _assemble_input_for_testing(self, data: list) -> dict:
       indices, X, missing_mask = self._send_data_to_given_device(data)

       inputs = {
           "X": X,
           "missing_mask": missing_mask,
       }
       return inputs

That is the contract to remember:

- **Training/validation** need ``X_ori`` and ``indicating_mask``
- **Testing** does **not**


The Complete Forward Flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   Training:
   DatasetForSAITS → [idx, X, mask, X_ori, indicating_mask]
     → _assemble_input_for_training() → {"X", "missing_mask", "X_ori", "indicating_mask"}
       → _SAITS.forward(inputs, calc_criterion=True)
         → returns {"imputation", "loss", ...}

   Validation:
   DatasetForSAITS → [idx, X, mask, X_ori, indicating_mask]
     → _assemble_input_for_validating() → {"X", "missing_mask", "X_ori", "indicating_mask"}
       → _SAITS.forward(inputs, calc_criterion=True)
         → returns {"imputation", "metric", ...}

   Testing (inference):
   BaseDataset → [idx, X, missing_mask]
     → _assemble_input_for_testing() → {"X", "missing_mask"}
       → _SAITS.forward(inputs, calc_criterion=False)
         → returns {"imputation", ...}


When You Need a Custom Dataset
---------------------------------

Add ``data.py`` **only** when ``BaseDataset`` cannot express your model's sample contract.

**Good reasons** to create a custom dataset:

- You need artificial masking like SAITS
- You need extra stage-dependent tensors
- File-mode loading needs special handling
- The default sample order does not cover your model

**Bad reason**:

- You only want to rename keys that the wrapper could assemble directly

Most models in PyPOTS do **not** need a custom dataset.
Check if ``BaseDataset`` can handle your requirements first.


Data Input Formats
--------------------

PyPOTS supports two input modes for all models:

**Dict input** (in-memory):

.. code-block:: python

   train_set = {
       "X": np.array(...),        # shape: [n_samples, n_steps, n_features]
       "y": np.array(...),        # shape: [n_samples] (optional)
   }
   val_set = {
       "X": np.array(...),
       "X_ori": np.array(...),    # original data for validation metric
       "y": np.array(...),
   }

**File input** (lazy-loading from HDF5):

.. code-block:: python

   from pypots.data.saving import save_dict_into_h5

   # Save data to HDF5 files
   save_dict_into_h5(train_set, "train_set.h5")
   save_dict_into_h5(val_set, "val_set.h5")

   # Use file paths instead of dicts
   model.fit("train_set.h5", "val_set.h5")
   results = model.predict("test_set.h5")

Both modes follow the same logical order. Make sure your model handles both consistently.


Fast Debugging Checklist
--------------------------

Before changing model math, check these first:

1. **Print the sample length and item order** from the dataset
2. **Print the dict keys** right before ``forward()``
3. **Print tensor shapes** for train, validation, and test separately
4. **Verify array input and file input** follow the same contract
5. **Verify masks** mean the same thing in dataset code and model code

.. code-block:: python

   # Quick debugging snippet for your _assemble_input_for_training:
   def _assemble_input_for_training(self, data: list) -> dict:
       print(f"Number of items in data: {len(data)}")
       for i, item in enumerate(data):
           if hasattr(item, 'shape'):
               print(f"  data[{i}]: shape={item.shape}, dtype={item.dtype}")
           else:
               print(f"  data[{i}]: type={type(item)}")
       # ... your actual assembly logic

In PyPOTS, many "model bugs" are really **data-contract bugs**.
