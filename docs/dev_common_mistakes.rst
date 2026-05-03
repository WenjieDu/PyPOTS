.. PyPOTS developer documentation - Common Integration Mistakes

Common Integration Mistakes
==============================

These are the failures that show up again and again during model integration.
Read this page before opening your PR.


1. Trusting Placeholders More Than Contracts
------------------------------------------------

**Symptom**: Template placeholder names used in production code without verifying the base class.

Templates are scaffolds. Task base classes define the real contract.

Example: Classification models are consumed through ``BaseNNClassifier.predict()``, which works with
``classification_proba`` and then adds ``classification``. A placeholder name inside a template
is not enough proof that the key is correct.

**Fix**:

- Read the task base class before copying a template
- Verify the exact result keys used by ``predict()`` and task helper methods

.. code-block:: python

   # Wrong: blindly copied from a template
   results = {"output": imputed_data}

   # Correct: verified against BaseNNImputer contract
   results = {"imputation": imputed_data}


2. Train/Validation/Test Input Drift
-----------------------------------------

**Symptoms**:

- Training works but validation crashes
- Dict input works but file input fails
- Testing uses a smaller input dict than training, and the core still expects training-only tensors

**Fix**:

- Print the assembled dict for all three stages
- Check sample order from the dataset
- Keep stage-specific assumptions explicit

``SAITS`` is the canonical example: training and validation need ``X_ori`` and ``indicating_mask``,
testing does **not**.

.. code-block:: python

   # Debugging: print what each stage receives
   def _assemble_input_for_training(self, data):
       result = self._do_assembly(data)
       print(f"Training keys: {result.keys()}")
       return result

   def _assemble_input_for_testing(self, data):
       result = self._do_assembly_minimal(data)
       print(f"Testing keys: {result.keys()}")
       return result


3. Choosing the Wrong Path
-----------------------------

**Symptoms**:

- A supposedly standard model keeps growing custom loop logic
- A rule-based model gets wrapped in fake optimizer code
- Review discussion keeps circling back to architecture choice

**Fix**: Refer to the decision matrix:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Situation
     - Correct Path
   * - One optimizer, one ordinary loop
     - :doc:`dev_standard_nn`
   * - Custom orchestration (multi-optimizer, pretraining, etc.)
     - :doc:`dev_complex_nn`
   * - No ``BaseNNModel``, no gradients
     - :doc:`dev_non_nn`


4. Breaking Checkpoint Semantics in Custom Training
------------------------------------------------------

**Symptoms**:

- Custom ``_train_model()`` runs, but the best model is never restored
- Early stopping stops too early or never stops
- Validation metric exists but is not used for model selection

**Fix**: When overriding ``_train_model()``, always preserve these four things:

.. code-block:: python

   # 1. Track best loss/epoch
   if mean_val_loss < self.best_loss:
       self.best_loss = mean_val_loss
       self.best_epoch = epoch
       self.best_model_dict = deepcopy(self.model.state_dict())
       patience_counter = 0
   else:
       patience_counter += 1

   # 2. Early stopping check
   if self.patience is not None and patience_counter >= self.patience:
       break

   # 3. Load best model after training loop
   self.model.load_state_dict(self.best_model_dict)

   # 4. Auto-save if configured
   self._auto_save_model_if_necessary(
       confirm_saving=self.model_saving_strategy == "best"
   )


5. Adding ``data.py`` for the Wrong Reason
---------------------------------------------

**Symptoms**:

- A custom dataset only renames fields that the wrapper could assemble
- Array mode and file mode start returning different contracts
- Dataset logic becomes harder to review than the model itself

**Fix**:

- Keep ``BaseDataset`` unless you truly need a new sample contract
- Add ``data.py`` only for new tensors, special masking, or file-mode behavior

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Good Reason
     - Bad Reason
   * - Need artificial masking (like SAITS)
     - Only want to rename dict keys
   * - Need extra stage-dependent tensors
     - Want to add convenience methods
   * - File-mode loading needs special handling
     - The default sample order is fine but you want different naming
   * - The default sample order doesn't cover your model
     - Copying another model's dataset "just in case"


6. Shipping Without the Right Test Scope
-------------------------------------------

**Symptoms**:

- The new model passes one happy-path test but fails save/load
- Only training is tested, not inference
- The model claims file input support but no file-path test exists

**Fix**:

- Run the model's targeted test file
- Verify ``fit()``, ``predict()``, and helper methods
- Verify save/load when the model is stateful
- Run broader regression if shared modules changed

.. code-block:: python

   # Minimum test coverage for a new model:

   class TestYourModel(unittest.TestCase):
       def test_0_fit(self):
           """Test that fit() runs without error."""
           self.model.fit(TRAIN_SET, VAL_SET)

       def test_1_predict(self):
           """Test that predict() returns correct results."""
           results = self.model.predict(TEST_SET)
           assert "imputation" in results
           assert not np.isnan(results["imputation"]).any()

       def test_2_impute(self):
           """Test the helper method."""
           imputed = self.model.impute(TEST_SET)
           assert imputed.shape == TEST_SET["X"].shape

       def test_3_save_load(self):
           """Test save/load round trip."""
           self.model.save("model_checkpoint.pypots")
           self.model.load("model_checkpoint.pypots")
           results = self.model.predict(TEST_SET)
           assert not np.isnan(results["imputation"]).any()

       def test_4_lazy_loading(self):
           """Test file-backed input."""
           self.model.fit(H5_TRAIN_PATH, H5_VAL_PATH)
           results = self.model.predict(H5_TEST_PATH)
           assert not np.isnan(results["imputation"]).any()


Quick Pre-PR Checklist
------------------------

Before opening a PR, verify:

.. code-block:: text

   [ ] Base class chosen correctly (standard NN / complex NN / non-NN)
   [ ] Result keys match the task base contract
   [ ] _assemble_input_* methods are consistent across train/val/test
   [ ] Custom dataset is justified (or BaseDataset is reused)
   [ ] Checkpoint semantics preserved if _train_model() is overridden
   [ ] Tests cover fit(), predict(), helper methods, save/load
   [ ] Both dict input and file input tested
   [ ] flake8 passes
   [ ] Targeted test file passes
