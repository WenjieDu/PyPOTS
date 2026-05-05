.. PyPOTS developer documentation - Complex NN Integration Path

Complex NN Integration Path
================================

Use this path when the **default** ``BaseNNModel._train_model()`` is **no longer enough**.

``USGAN`` is the clearest example.


When This Path Is Correct
----------------------------

Move to the complex path if you need one or more of these:

- **Multiple optimizers** (e.g. generator + discriminator)
- **Alternating update schedules** (e.g. train G for k steps, then D for 1 step)
- **Different training branches** for different submodules
- **Explicit pretraining** before the main training loop
- **Custom checkpoint-selection logic** that still follows task semantics

Examples in PyPOTS:

- ``USGAN`` — generator/discriminator alternation
- ``CRLI`` — multi-optimizer clustering training
- ``VaDER`` — pretraining before the main training phase


What Stays The Same
---------------------

Even when you override ``_train_model()``, these contracts should stay **stable**:

- The wrapper still exposes the same public ``fit()`` and ``predict()`` API
- The model still returns the correct inference-time task result
- Best checkpoint tracking still exists
- Early stopping still uses a meaningful metric
- The final trained wrapper still loads the best checkpoint before inference

If those invariants disappear, review becomes much harder.


``USGAN`` as the Reference Pattern
--------------------------------------

``USGAN`` still inherits ``BaseNNImputer``, but it cannot use the default training loop.

Why?

- It has ``G_optimizer`` and ``D_optimizer``
- It alternates generator and discriminator updates
- One batch may execute multiple optimizer steps

So ``USGAN`` **overrides** ``_train_model()`` in the wrapper.


What ``USGAN._train_model()`` Preserves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The useful lesson is not "copy this loop."
The useful lesson is **what it preserves** while changing the orchestration:

- Resets best-loss state at the start
- Runs explicit train and validation phases
- Tracks the metric used for best-model selection
- Updates patience for early stopping
- Saves better checkpoints when configured
- Leaves inference-time task behavior unchanged

That is the contract to preserve.


Implementation Guide
========================


Step 1: Start From the Standard Path
---------------------------------------

Begin by setting up your model as if it were a standard NN model (see :doc:`dev_standard_nn`).
Then identify which part of the training loop needs to change.

Typical reasons to override ``_train_model()``:

- You have two or more optimizers that need separate ``step()`` calls
- Different sub-models need different forward passes per batch
- You need a pretraining phase before the main loop


Step 2: Implement the Custom Training Loop
----------------------------------------------

Override ``_train_model()`` in your ``model.py``. Here is a skeleton based on ``USGAN``:

.. code-block:: python

   # In your model.py

   class YourGANModel(BaseNNImputer):
       def __init__(self, ...):
           super().__init__(...)

           # Set up dual optimizers
           self.model = _YourGANCore(...)
           self._print_model_size()
           self._send_model_to_given_device()

           # Generator optimizer
           self.G_optimizer = Adam(lr=0.001)
           self.G_optimizer.init_optimizer(self.model.generator.parameters())

           # Discriminator optimizer
           self.D_optimizer = Adam(lr=0.001)
           self.D_optimizer.init_optimizer(self.model.discriminator.parameters())

       def _train_model(self, train_dataloader, val_dataloader=None):
           # Reset tracking state
           self.best_loss = float("inf")
           self.best_epoch = 0
           patience_counter = 0

           for epoch in range(self.epochs):
               self.model.train()
               epoch_train_loss_collector = []

               for idx, data in enumerate(train_dataloader):
                   inputs = self._assemble_input_for_training(data)

                   # --- Discriminator update ---
                   for _ in range(self.D_steps):
                       self.D_optimizer.zero_grad()
                       results = self.model(inputs, training_object="discriminator")
                       results["D_loss"].backward()
                       self.D_optimizer.step()

                   # --- Generator update ---
                   for _ in range(self.G_steps):
                       self.G_optimizer.zero_grad()
                       results = self.model(inputs, training_object="generator")
                       results["G_loss"].backward()
                       self.G_optimizer.step()

                   epoch_train_loss_collector.append(
                       results["G_loss"].item()
                   )

               # --- Validation phase ---
               if val_dataloader is not None:
                   self.model.eval()
                   val_loss_collector = []
                   with torch.no_grad():
                       for idx, data in enumerate(val_dataloader):
                           inputs = self._assemble_input_for_validating(data)
                           results = self.model(inputs, calc_criterion=True)
                           val_loss_collector.append(
                               results["metric"].item()
                           )

                   mean_val_loss = np.mean(val_loss_collector)

                   # --- Best model tracking ---
                   if mean_val_loss < self.best_loss:
                       self.best_loss = mean_val_loss
                       self.best_epoch = epoch
                       self.best_model_dict = deepcopy(self.model.state_dict())
                       patience_counter = 0
                   else:
                       patience_counter += 1

                   # --- Early stopping ---
                   if (self.patience is not None
                       and patience_counter >= self.patience):
                       break

               # --- TensorBoard logging ---
               if self.summary_writer is not None:
                   mean_train_loss = np.mean(epoch_train_loss_collector)
                   self._save_log_into_tb_file(
                       epoch, "training", {"loss": mean_train_loss}
                   )
                   if val_dataloader is not None:
                       self._save_log_into_tb_file(
                           epoch, "validating", {"loss": mean_val_loss}
                       )


Step 3: Handle the Core Differently
--------------------------------------

For a GAN-like model, your ``core.py`` forward pass may need a ``training_object`` parameter
to distinguish between generator and discriminator forward passes:

.. code-block:: python

   class _YourGANCore(nn.Module):
       def __init__(self, ...):
           super().__init__()
           self.generator = ...
           self.discriminator = ...

       def forward(self, inputs, training_object="generator",
                   calc_criterion=False):
           X, missing_mask = inputs["X"], inputs["missing_mask"]

           # Generator forward pass
           imputed_data = self.generator(X, missing_mask)
           imputed_data = missing_mask * X + (1 - missing_mask) * imputed_data

           results = {"imputation": imputed_data}

           if training_object == "discriminator":
               # Discriminator-specific loss
               d_prob = self.discriminator(imputed_data.detach(), missing_mask)
               results["D_loss"] = self._d_loss(d_prob, missing_mask)

           elif training_object == "generator":
               # Generator-specific loss (including adversarial + reconstruction)
               d_prob = self.discriminator(imputed_data, missing_mask)
               results["G_loss"] = self._g_loss(
                   d_prob, imputed_data, X, missing_mask
               )

           if calc_criterion:
               # For validation metric
               X_ori = inputs["X_ori"]
               indicating_mask = inputs["indicating_mask"]
               if not self.training:
                   results["metric"] = self.validation_metric(
                       imputed_data, X_ori, indicating_mask
                   )

           return results


What You Are Allowed to Customize
------------------------------------

Override ``_train_model()`` **only** for orchestration-level reasons such as:

- Custom optimizer order
- Custom gradient flow
- Pretraining stages
- Multi-branch loss collection
- Special logging needs tied to the custom schedule

Do **not** override it just to move ordinary data assembly or forward logic around.


High-Risk Mistakes
---------------------

The complex path usually fails in four places:

1. **Best checkpoint selected from the wrong signal**
   — e.g. using generator loss instead of a proper validation metric
2. **Patience updated inconsistently**
   — e.g. forgetting to update patience in some code paths
3. **Training and validation use different input key assumptions**
   — e.g. validation assembly expects keys that aren't provided
4. **Inference still works, but the task result key changes silently**
   — e.g. returning ``"generated"`` instead of ``"imputation"``

Quick self-check: "Did I only change training orchestration, or did I accidentally change
the public model contract too?"


Definition of Done
---------------------

Your complex NN integration is done when:

- Each optimizer branch is exercised during training
- Validation still drives model selection in a clear way
- The best checkpoint is restored after training
- ``predict()`` still returns the expected task result
- Targeted tests prove both training and inference paths

If only one optimizer and one ordinary loss remain, the model probably belongs back
on the :doc:`dev_standard_nn`.
