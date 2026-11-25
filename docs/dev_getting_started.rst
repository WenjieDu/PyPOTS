.. PyPOTS developer documentation - Getting Started

Getting Started
=================

Welcome to the PyPOTS developer documentation!
This guide helps contributors understand the codebase and integrate new models, algorithms, and features.

If you are new to PyPOTS, **do not start from a random model folder**.
Start from understanding the **contracts** — the base classes and their responsibilities.


Recommended Reading Route
--------------------------

Follow these sections in order:

1. **This page** — set up your environment, learn the key concepts
2. :doc:`dev_architecture` — understand the codebase layout, base class hierarchy, and data flow
3. :doc:`dev_integration_guide` — follow the step-by-step guide for your model type
4. :doc:`dev_quality` — avoid common mistakes and pass the testing checklist


Setting Up the Development Environment
-----------------------------------------

.. code-block:: bash

   git clone https://github.com/WenjieDu/PyPOTS.git
   cd PyPOTS
   pip install -e ".[dev]"
   pre-commit install

Or with conda:

.. code-block:: bash

   conda create -n pypots python=3.10
   conda activate pypots
   git clone https://github.com/WenjieDu/PyPOTS.git
   cd PyPOTS
   pip install -e ".[dev]"
   pre-commit install


Using Pre-commit
^^^^^^^^^^^^^^^^^^^^

We recommend enabling `pre-commit <https://pre-commit.com/>`_ before contributing.
The repository's ``.pre-commit-config.yaml`` defines the hooks, but they do not
run automatically until you install them in your local clone. If you have not
installed the development dependencies, you can install pre-commit separately:

.. code-block:: bash

   pip install pre-commit
   pre-commit install

Run the installation commands from the repository root. You only need to install
the Git hooks once per clone. The first run may take longer because pre-commit
creates and caches its own hook environments. Ruff is pinned to the same version
as CI and uses the rules and 120-character line-length setting in ``pyproject.toml``.
You do not need to install Ruff separately to use the hooks.

During a normal ``git commit``, the hooks check staged files, fix whitespace and
end-of-file issues, validate YAML and TOML, detect merge conflict markers, format
Python code, and run Ruff linting. If you prefer to check your staged changes
before committing, run:

.. code-block:: bash

   pre-commit run

To check all tracked files, including files outside your current changes, run:

.. code-block:: bash

   pre-commit run --all-files

If a hook modifies a file, the commit stops so you can review the changes. Use
``git diff`` to inspect them, stage the intended changes again with ``git add``,
and retry the commit. Fix any remaining lint or configuration errors before
retrying. Keep unrelated automatic fixes out of your pull request when running
hooks across the whole repository.

Pre-commit installation is recommended, not required. GitHub CI independently
checks Ruff linting and formatting for pull requests targeting ``main`` or ``dev``,
regardless of whether contributors enable local hooks. See :doc:`dev_testing`
for the pre-PR checks.


Key Concepts
--------------

Before diving into the code, understand these three concepts that define how PyPOTS works.

Three-Layer Model Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every model in PyPOTS follows a three-layer architecture:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - File
     - Layer
     - Responsibility
   * - ``model.py``
     - Wrapper
     - User-facing API, dataloaders, optimizers, training orchestration, input assembly
   * - ``core.py``
     - Core
     - Forward computation, result dict creation, loss and metric outputs
   * - ``data.py``
     - Dataset
     - Custom dataset class (only when ``BaseDataset`` is not enough)


Three Integration Paths
^^^^^^^^^^^^^^^^^^^^^^^^^^

Before writing any code, decide which integration path your model belongs to.
This is the most important decision — changing paths late usually means you started from the wrong contract.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Path
     - When to Use
     - Reference Model
   * - **Standard NN**
     - One optimizer, default training loop. Most models fall here.
     - ``SAITS`` (``pypots/imputation/saits/``)
   * - **Complex NN**
     - Multiple optimizers, alternating updates, or pretraining stages.
     - ``USGAN`` (``pypots/imputation/usgan/``)
   * - **Non-NN**
     - Rule-based, statistical, or algorithmic. No gradients.
     - ``LOCF`` (``pypots/imputation/locf/``)


Six Supported Tasks
^^^^^^^^^^^^^^^^^^^^^

PyPOTS organizes models by task. Each task has its own base class and result contract:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Task
     - NN Base
     - Non-NN Base
     - Result Key
   * - Imputation
     - ``BaseNNImputer``
     - ``BaseImputer``
     - ``"imputation"``
   * - Forecasting
     - ``BaseNNForecaster``
     - ``BaseForecaster``
     - ``"forecasting"``
   * - Classification
     - ``BaseNNClassifier``
     - ``BaseClassifier``
     - ``"classification"``
   * - Anomaly Detection
     - ``BaseNNDetector``
     - ``BaseDetector``
     - ``"anomaly_detection"``
   * - Clustering
     - ``BaseNNClusterer``
     - ``BaseClusterer``
     - ``"clustering"``
   * - Representation
     - ``BaseNNRepresentor``
     - ``BaseRepresentor``
     - ``"representation"``


How to Read a Reference Model
---------------------------------

When reading an example model implementation, follow this order:

1. **Task base class** — understand the contract (result keys, helper methods)
2. ``model.py`` — the public wrapper API, dataloaders, optimizers, training orchestration
3. ``core.py`` — forward computation and result dict contract
4. ``data.py`` — only if it exists; the custom dataset class
5. **The matching test file** — under ``tests/<task>/``


End-to-End Development Journey
---------------------------------

The shortest safe path from idea to merged PR.


Step 1: Define the Contract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before touching implementation code, decide:

- **The task**: ``imputation``, ``forecasting``, ``classification``, ``anomaly_detection``,
  ``clustering``, or ``representation``
- **The correct base class**: e.g. ``BaseNNImputer`` for an NN imputation model
- **The public result key**: e.g. ``"imputation"`` for imputation models
- **The integration path**: standard NN, complex NN, or non-NN


Step 2: Start From a Scaffold
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the task template as a starting folder:

.. code-block:: text

   pypots/imputation/template/
   pypots/forecasting/template/
   pypots/classification/template/
   pypots/clustering/template/

Then compare it with the matching reference model (``SAITS``, ``USGAN``, or ``LOCF``).
The template gives structure; the reference model gives the actual contract.


Step 3: Implement
^^^^^^^^^^^^^^^^^^^^

Follow the detailed guide for your chosen path:

- :doc:`dev_standard_nn` — for standard NN models
- :doc:`dev_complex_nn` — for complex NN models
- :doc:`dev_non_nn` — for non-NN models


Step 4: Wire the Package
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Export the model in the task package ``__init__.py``
- Add the matching test file under ``tests/<task>/``


Step 5: Validate Locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Generate test data
   python tests/global_test_config.py

   # Run your model's targeted test
   pytest -rA tests/imputation/your_model.py -n 1

   # Check formatting and lint
   ruff format . --check
   ruff check .


Step 6: Submit with Evidence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your PR should state:

- The chosen integration path and reason
- Exact local commands you ran and their results
- Known limitations, if any

See :doc:`dev_testing` for the full testing checklist and CI guide.
