.. PyPOTS developer documentation - Testing and CI

Testing Checklist and CI Guide
================================

Run this checklist before opening a model-related PR.


Environment Setup
-------------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/WenjieDu/PyPOTS.git
   cd PyPOTS

   # Install in development mode
   pip install -e ".[dev]"

   # Generate test data (required before running any test)
   python tests/global_test_config.py


Understanding the Test Infrastructure
-----------------------------------------

Test Configuration: ``global_test_config.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file ``tests/global_test_config.py`` sets up shared test data and configuration:

.. code-block:: python

   # Key constants used across all tests:
   RANDOM_SEED = 2023
   EPOCHS = 2                  # Very few epochs for fast testing
   N_STEPS = 6
   N_PRED_STEPS = 2
   N_FEATURES = 5
   N_CLASSES = 2
   N_SAMPLES_PER_CLASS = 100
   MISSING_RATE = 0.1

   # Pre-generated data splits:
   TRAIN_SET = {"X": ..., "y": ...}
   VAL_SET = {"X": ..., "X_ori": ..., "y": ...}
   TEST_SET = {"X": ..., "X_ori": ..., "y": ...}

   # HDF5 file paths for lazy-loading tests:
   GENERAL_H5_TRAIN_SET_PATH = "..."
   GENERAL_H5_VAL_SET_PATH = "..."
   GENERAL_H5_TEST_SET_PATH = "..."

   # For forecasting tasks:
   FORECASTING_TRAIN_SET = {"X": ..., "X_pred": ...}
   FORECASTING_VAL_SET = {"X": ..., "X_pred": ...}
   FORECASTING_TEST_SET = {"X": ..., "X_pred": ...}

   # Device selection (auto-detects CUDA):
   DEVICE = None  # or cuda device if available

The test data is generated using ``benchpots.datasets.preprocess_random_walk()``
which creates a synthetic random walk dataset with configurable missingness.


Test File Structure
^^^^^^^^^^^^^^^^^^^^^

Each model has a dedicated test file under ``tests/<task>/``:

.. code-block:: text

   tests/
   ├── global_test_config.py       # Shared configuration
   ├── imputation/
   │   ├── saits.py                # SAITS test cases
   │   ├── brits.py                # BRITS test cases
   │   ├── locf.py                 # LOCF test cases
   │   ├── usgan.py                # USGAN test cases
   │   └── ...                     # One file per model
   ├── classification/
   ├── forecasting/
   ├── clustering/
   ├── anomaly_detection/
   └── representation/


Writing Tests for Your Model
-------------------------------

Use the SAITS test as a reference. Here is a complete test template:

.. code-block:: python

   # tests/imputation/your_model.py

   import os
   import unittest

   import numpy as np
   import pytest

   from pypots.imputation import YourModel
   from pypots.nn.functional import calc_mse
   from pypots.optim import Adam
   from pypots.utils.logging import logger
   from tests.global_test_config import (
       DATA,
       EPOCHS,
       DEVICE,
       TRAIN_SET,
       VAL_SET,
       TEST_SET,
       GENERAL_H5_TRAIN_SET_PATH,
       GENERAL_H5_VAL_SET_PATH,
       GENERAL_H5_TEST_SET_PATH,
       RESULT_SAVING_DIR_FOR_IMPUTATION,
       check_tb_and_model_checkpoints_existence,
   )


   class TestYourModel(unittest.TestCase):
       logger.info("Running tests for YourModel...")

       # Set paths
       saving_path = os.path.join(
           RESULT_SAVING_DIR_FOR_IMPUTATION, "YourModel"
       )
       model_save_name = "saved_your_model.pypots"

       # Initialize optimizer
       optimizer = Adam(lr=0.001, weight_decay=1e-5)

       # Initialize model with small hyperparameters for fast testing
       model = YourModel(
           DATA["n_steps"],
           DATA["n_features"],
           d_model=32,
           epochs=EPOCHS,
           saving_path=saving_path,
           optimizer=optimizer,
           device=DEVICE,
       )

       @pytest.mark.xdist_group(name="imputation-your_model")
       def test_0_fit(self):
           """Test that the model trains successfully."""
           self.model.fit(TRAIN_SET, VAL_SET)

       @pytest.mark.xdist_group(name="imputation-your_model")
       def test_1_impute(self):
           """Test that predict() returns valid imputation results."""
           results = self.model.predict(TEST_SET)
           assert not np.isnan(results["imputation"]).any(), (
               "Output still has missing values after imputation."
           )

           test_MSE = calc_mse(
               results["imputation"],
               DATA["test_X_ori"],
               DATA["test_X_indicating_mask"],
           )
           logger.info(f"YourModel test_MSE: {test_MSE}")

       @pytest.mark.xdist_group(name="imputation-your_model")
       def test_2_parameters(self):
           """Test that model parameters are properly initialized."""
           assert hasattr(self.model, "model") and self.model.model is not None
           assert hasattr(self.model, "optimizer") and self.model.optimizer is not None
           assert hasattr(self.model, "best_loss")
           self.assertNotEqual(self.model.best_loss, float("inf"))
           assert hasattr(self.model, "best_model_dict")
           assert self.model.best_model_dict is not None

       @pytest.mark.xdist_group(name="imputation-your_model")
       def test_3_saving_path(self):
           """Test model save and load functionality."""
           # Check tensorboard and checkpoint files
           assert os.path.exists(self.saving_path)
           check_tb_and_model_checkpoints_existence(self.model)

           # Test save/load round trip
           saved_model_path = os.path.join(
               self.saving_path, self.model_save_name
           )
           self.model.save(saved_model_path)
           self.model.load(saved_model_path)

       @pytest.mark.xdist_group(name="imputation-your_model")
       def test_4_lazy_loading(self):
           """Test with HDF5 file-backed input (lazy loading)."""
           self.model.fit(
               GENERAL_H5_TRAIN_SET_PATH,
               GENERAL_H5_VAL_SET_PATH
           )
           results = self.model.predict(GENERAL_H5_TEST_SET_PATH)
           assert not np.isnan(results["imputation"]).any(), (
               "Output still has missing values with lazy loading."
           )

           test_MSE = calc_mse(
               results["imputation"],
               DATA["test_X_ori"],
               DATA["test_X_indicating_mask"],
           )
           logger.info(f"Lazy-loading YourModel test_MSE: {test_MSE}")


   if __name__ == "__main__":
       unittest.main()

Key points about the test structure:

- **Test numbering**: Tests are numbered ``test_0_``, ``test_1_``, etc. to ensure execution order
- **xdist_group marker**: Required for parallel test execution with ``pytest-xdist``
- **Lazy loading test**: Tests HDF5 file input in addition to dict input
- **Save/load test**: Verifies the full checkpoint round trip
- **MSE calculation**: Uses ``calc_mse`` with the indicating mask for proper evaluation


Minimum Required Checks
--------------------------


1. Run the Targeted Model Test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Your specific model
   pytest -rA tests/imputation/your_model.py -n 1

   # Example reference models
   pytest -rA tests/imputation/saits.py -n 1
   pytest -rA tests/imputation/usgan.py -n 1
   pytest -rA tests/imputation/locf.py -n 1


2. Verify the Real Contract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At minimum, confirm all of these:

- ``fit()`` completes if the model has a training phase
- ``predict()`` returns the correct task result key
- Helper methods (e.g. ``impute()``, ``forecast()``) return the expected array shape
- Task-specific assumptions are tested


3. Verify Save/Load When State Exists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the model is stateful, verify the full round trip:

.. code-block:: python

   # 1. Train the model
   model.fit(TRAIN_SET, VAL_SET)

   # 2. Save
   model.save("checkpoint.pypots")

   # 3. Load
   model.load("checkpoint.pypots")

   # 4. Predict again — should still work
   results = model.predict(TEST_SET)
   assert not np.isnan(results["imputation"]).any()


4. Verify Every Claimed Input Mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the model claims to support file-path input, **test file-path input**.
Do not stop after dict input passes.


When to Run Broader Regression
---------------------------------

Run broader regression when you change shared modules:

- ``pypots/base.py``
- ``pypots/data/``
- ``pypots/nn/``
- ``pypots/optim/``

.. code-block:: bash

   pytest -rA -s tests/*/* -n 1 --cov=pypots --dist=loadgroup --cov-config=.coveragerc


CI and Lint
=============

This section maps real PyPOTS CI behavior to local commands.


What CI Checks
-----------------

The CI workflows currently perform these core checks:

1. ``flake8 .`` — code style linting
2. Package build — ``python -m build``
3. Full pytest with coverage — parallel execution with ``--dist=loadgroup``


Local Commands That Match CI
-------------------------------

Lint
^^^^^^

.. code-block:: bash

   flake8 .


Test Environment Setup
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python tests/global_test_config.py


Targeted Model Test
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pytest -rA tests/imputation/your_model.py -n 1


Full Regression
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pytest -rA -s tests/*/* -n 1 --cov=pypots --dist=loadgroup --cov-config=.coveragerc


Package Build
^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m build

Run this when packaging or install behavior may be affected.


Fast Triage Rules
--------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Problem
     - Action
   * - Lint failure only
     - Start with ``flake8 .``
   * - One-model failure
     - Run that model's test file directly
   * - Shared-module change
     - Run the full regression command
   * - Packaging suspicion
     - Run ``python -m build``


Review-Ready Evidence
-----------------------

A PR is **not** review-ready unless it includes:

- The **exact commands** you ran
- Whether the run was **targeted or broad**
- The **result** of those commands
- Any **remaining gap** you did not cover

Example PR evidence:

.. code-block:: text

   ## Testing Evidence

   ### Environment
   - Python 3.10, PyTorch 2.1, CUDA 12.1

   ### Commands Run
   ```
   python tests/global_test_config.py
   pytest -rA tests/imputation/my_model.py -n 1
   flake8 .
   ```

   ### Results
   - All 5 tests passed
   - No lint errors
   - Scope: targeted (only my_model changed)
