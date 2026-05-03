.. PyPOTS developer documentation - Codebase Map

Codebase Map and Folder Intent
================================

Use this page to find the right place before reading code in depth.
Put code where another contributor would expect to find it — that rule prevents many review problems.


Repository Overview
---------------------

.. code-block:: text

   PyPOTS/
   ├── pypots/                     # Main source package
   │   ├── base.py                 # Shared model abstractions (BaseModel, BaseNNModel)
   │   ├── version.py              # Version information
   │   ├── imputation/             # Imputation task models
   │   ├── forecasting/            # Forecasting task models
   │   ├── classification/         # Classification task models
   │   ├── clustering/             # Clustering task models
   │   ├── anomaly_detection/      # Anomaly detection task models
   │   ├── representation/         # Representation learning models
   │   ├── data/                   # Dataset and IO helpers
   │   │   ├── dataset/base.py     # BaseDataset class
   │   │   ├── checking.py         # Data validation functions
   │   │   ├── saving/             # Data saving utilities (e.g. HDF5)
   │   │   └── utils.py            # Data transformation utilities
   │   ├── nn/                     # Reusable neural network modules
   │   │   ├── functional/         # Utility functions (error, classification, etc.)
   │   │   └── modules/            # PyTorch model components and backbones
   │   ├── optim/                  # Optimizer abstractions
   │   ├── cli/                    # Command-line interface
   │   └── utils/                  # Utility functions (logging, file ops, etc.)
   ├── tests/                      # Model and task tests
   │   ├── global_test_config.py   # Shared test configuration and data
   │   ├── imputation/             # Imputation model tests
   │   ├── forecasting/            # Forecasting model tests
   │   ├── classification/         # Classification model tests
   │   ├── clustering/             # Clustering model tests
   │   ├── anomaly_detection/      # Anomaly detection model tests
   │   └── representation/         # Representation learning tests
   ├── docs/                       # Sphinx documentation source
   ├── requirements/               # Dependency specifications
   └── .github/workflows/          # CI behavior used in pull requests


Core Directories Explained
----------------------------

``pypots/<task>/`` — Task Wrappers and Model Folders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each task package (e.g. ``pypots/imputation/``) contains:

- ``base.py`` — Task-specific base classes (e.g. ``BaseImputer``, ``BaseNNImputer``)
- ``template/`` — Scaffolding folder to help new contributors get started
- One subfolder per model (e.g. ``saits/``, ``brits/``, ``locf/``)

Each model subfolder typically has:

.. code-block:: text

   pypots/imputation/saits/
   ├── __init__.py    # Exports the public model class
   ├── model.py       # User-facing wrapper API
   ├── core.py        # Forward computation and result dict
   └── data.py        # Custom dataset (only if needed)


``pypots/base.py`` — Framework Contracts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This file defines the shared model abstractions:

- ``BaseModel`` — device setup, AMP switch, checkpoint IO, abstract ``fit()``/``predict()``
- ``BaseNNModel`` — training loop state, early stopping, best checkpoint tracking, TensorBoard logging

All models ultimately inherit from one of these.


``pypots/data/`` — Dataset and IO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``dataset/base.py`` — ``BaseDataset`` class that all datasets inherit from
- ``checking.py`` — Functions to validate data keys and structure
- ``saving/`` — Utilities for saving data to HDF5 files
- ``utils.py`` — Data transformation utilities


``pypots/nn/`` — Reusable Neural Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Contains 60+ reusable PyTorch modules:

- ``modules/base_model_core.py`` — ``ModelCore`` base class for all NN model cores
- ``modules/loss.py`` — Loss functions: ``Criterion``, ``MAE``, ``MSE``, ``RMSE``, ``MRE``, ``CrossEntropy``, ``NLL``
- ``modules/metric.py`` — Metric evaluation
- ``modules/<model_name>/`` — Model-specific backbone implementations (e.g. ``saits/``, ``transformer/``)
- ``functional/`` — Utility functions (``calc_mse``, ``calc_mae``, ``gather_listed_dicts``, etc.)

If you implement reusable NN components, put them here instead of inside a model folder.


``pypots/optim/`` — Optimizer Abstractions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Provides ``Optimizer`` base class and concrete implementations (e.g. ``Adam``, ``SGD``).
All PyPOTS optimizers wrap PyTorch optimizers with a consistent interface.


Template Directories
----------------------

If you are adding a new model, check the task template first:

.. code-block:: text

   pypots/imputation/template/
   pypots/forecasting/template/
   pypots/classification/template/
   pypots/clustering/template/

Treat templates as **scaffolding**. The task base class still defines the real contract.
Always verify result keys and helper method behavior against the base class.


Three Example Folders Worth Reading First
-------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Folder
     - Path Type
     - Key Lesson
   * - ``pypots/imputation/saits/``
     - Standard NN
     - One optimizer, default training loop, custom dataset
   * - ``pypots/imputation/usgan/``
     - Complex NN
     - Dual optimizer GAN, custom ``_train_model()``
   * - ``pypots/imputation/locf/``
     - Non-NN
     - No training, inherits ``BaseImputer`` directly

Reading those three folders gives you a fast mental map of the main extension styles in PyPOTS.


Dependency Direction
----------------------

Task packages depend on shared infrastructure.
Shared infrastructure should **not** depend on task packages.

In practice:

- Task-specific orchestration stays in task folders
- Reusable blocks move to ``pypots/nn/``
- Cross-task utilities should not be hidden inside one model folder


Module Boundary Rules
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - File
     - What Belongs Here
   * - ``model.py``
     - Public wrapper API, dataset/dataloader setup, optimizer creation, training orchestration, stage-specific input assembly
   * - ``core.py``
     - Forward computation, result dict creation, loss and metric outputs. Should **not** become a hidden wrapper.
   * - ``data.py``
     - Custom dataset class. **Only** add when ``BaseDataset`` cannot express your model's sample contract.


Quick Self-Check Before Commit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Can someone infer intent from the file location alone?
- Did reusable code go to shared modules instead of one model folder?
- Does the wrapper own orchestration and the core own math?
- Is a custom dataset really necessary?
