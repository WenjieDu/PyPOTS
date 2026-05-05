Model Integration Guide
=========================

This section walks you through integrating a new model into PyPOTS.
Choose the path that matches your model before writing any code.

- **Standard NN Path** — one optimizer, default training loop (most models)
- **Complex NN Path** — multiple optimizers or custom training orchestration (e.g. GANs)
- **Non-NN Path** — rule-based, statistical, or algorithmic models (no gradients)

.. toctree::
   :maxdepth: 2

   dev_standard_nn
   dev_complex_nn
   dev_non_nn
