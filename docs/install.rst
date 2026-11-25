Installation
============

How to Install
""""""""""""""

PyPOTS requires Python 3.10 or later and is available on
`PyPI <https://pypi.org/project/pypots/>`__ and
`Anaconda <https://anaconda.org/conda-forge/pypots>`__.

Choose **one** of the following installation methods according to your environment.
Upgrade commands are only needed if you already have PyPOTS installed.

Option 1: pip (recommended)
***************************

.. code-block:: bash

   pip install pypots
   # To upgrade: pip install pypots --upgrade

Option 2: Conda
***************

.. code-block:: bash

   conda install conda-forge::pypots
   # To upgrade: conda update conda-forge::pypots

Option 3: Docker
****************

.. code-block:: bash

   docker run -it --name pypots wenjiedu/pypots

Press Ctrl-P followed by Ctrl-Q to detach without stopping the container.
To reconnect, run ``docker attach pypots``.

Option 4: Latest source code on main (may include unreleased changes)
*********************************************************************

.. code-block:: bash

   pip install https://github.com/WenjieDu/PyPOTS/archive/main.zip

Required Dependencies
"""""""""""""""""""""

* Python >=3.10

The runtime dependencies and version constraints are defined in
`requirements/requirements.txt <https://github.com/WenjieDu/PyPOTS/blob/main/requirements/requirements.txt>`__.
The current list is:

* click
* h5py
* numpy
* optuna
* scipy
* sympy
* einops
* pandas
* seaborn
* matplotlib
* tensorboard
* scikit-learn
* transformers<=4.57.6
* torch>=1.10.0
* tsdb>=0.8
* pygrinder>=0.7
* benchpots>=1
* ai4ts>=0.1

Optional Dependencies
"""""""""""""""""""""
* torch-geometric (optional, required for GNN models like Raindrop)
* torch-scatter (optional, required for GNN models like Raindrop)
* torch-sparse (optional, required for GNN models like Raindrop)
* torch-npu (optional, required for Ascend NPU acceleration; match it to your PyTorch and CANN versions)


Reasons of Version Limitations on Dependencies
""""""""""""""""""""""""""""""""""""""""""""""
* **Why do we require Python >=3.10?**

Python 3.9 reached end of life in October 2025, and PyPOTS now requires Python 3.10 or newer, including native support for PEP 604 union type annotations.
We retain Python 3.10 compatibility for users with existing environments while testing newer Python versions as well.
Our Linux CPU CI tests Python 3.10 and 3.14 with compatible versions of PyTorch, NumPy, pandas, and the optional PyG extensions.
Choose dependency versions that support your Python version and platform. See the `Python version status <https://devguide.python.org/versions/>`_ for upstream maintenance information.

* **Why we need PyTorch >=1.10?**

Because of pytorch_sparse, please refer to https://github.com/rusty1s/pytorch_sparse/issues/207#issuecomment-1065549338.

.. _hardware-acceleration:

Acceleration
""""""""""""
PyPOTS supports CPU, NVIDIA CUDA GPUs, and Huawei Ascend NPUs for supported models.
For neural-network models, the ``device`` argument selects the backend.
When ``device=None``, PyPOTS probes CUDA first, then Ascend NPU, and finally falls back to CPU.
An explicitly requested accelerator must be available in the environment.

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Backend
     - Single-device selection
     - Multi-process DDP backend
   * - CPU
     - ``device="cpu"``
     - Gloo
   * - NVIDIA CUDA GPU
     - ``device="cuda"`` or ``"cuda:0"``
     - NCCL
   * - Ascend NPU
     - ``device="npu"`` or ``"npu:0"``
     - HCCL (via torch-npu)

.. note::

   Ascend NPU and DDP support is available in the development branch.
   If your installed release does not include these features yet, install the development
   source with ``pip install https://github.com/WenjieDu/PyPOTS/archive/dev.zip``.
   The actual model and operator coverage depends on the device and software stack.

GPU Acceleration (CUDA)
************************
Install a CUDA-enabled PyTorch build suitable for your NVIDIA GPU.
Pass ``device="cuda"`` or a visible device index such as ``device="cuda:0"``
to enable GPU training and inference for supported neural-network models.
The existing device-list interface (e.g. ``device=["cuda:0", "cuda:1"]``) uses
PyTorch DataParallel for supported models. For multi-process training, see :doc:`distributed`.

NPU Acceleration (Ascend)
**************************
Install the Ascend driver, CANN, PyTorch, and a matching torch-npu package.
Initialize the CANN environment and import ``torch_npu`` before selecting an explicit
NPU device. Use ``device="npu"`` or ``device="npu:0"`` with the standard model API.
See :doc:`npu` for a tested software stack, setup, usage, and model/operator limitations.
CPU and CUDA users do not need to install torch-npu.

Distributed Training (DDP)
***************************
PyPOTS provides a ``run_ddp_fit()`` helper for multi-process training of compatible
neural-network models using PyTorch DistributedDataParallel (DDP).
Launch one process per accelerator with ``torchrun``, pass a single device to each
model instance, and call the helper instead of ``model.fit()``.
DDP is distinct from the existing DataParallel device-list interface and uses NCCL
for CUDA, HCCL for Ascend NPU, or Gloo for CPU.
See :doc:`distributed` for a runnable example, launch commands, and limitations.

Automatic mixed precision (AMP) can be enabled on CUDA and Ascend NPU for models
that opt in to AMP support by setting the environment variable ``ENABLE_AMP=1``
before training. Model-specific precision and operator limitations still apply.

CPU Acceleration
****************
If you're using a Mac device with Apple Silicon in
you can install the `accelerate` data-science packages to obtain faster processing speed,
because they get optimized for Apple Silicon.
``conda install numpy scipy scikit-learn numexpr "libblas=*=*accelerate"``

If you're using devices with Intel chips in, you should install the distribution of MKL, which is optimized for multi-core Intel CPUs,
``conda install numpy scipy scikit-learn numexpr "libblas=*=*mkl"``

If you're using devices with AMD chips in, you can install with the distribution of OpenBLAS,
``conda install -c conda-forge numpy scipy scikit-learn numexpr "libblas=*=*openblas"``
