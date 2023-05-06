Installation
============

----------------


How to Install
""""""""""""""
It is recommended to use **pip** or **conda** for PyPOTS installation as shown below:

.. code-block:: bash

    # by pip
    pip install pypots            # the first time installation
    pip install pypots --upgrade  # update pypots to the latest version

.. code-block:: bash

    # by conda
    conda install -c conda-forge pypots  # the first time installation
    conda update  -c conda-forge pypots  # update pypots to the latest version

Alternatively, you can install from the latest source code which may be not officially released yet:

.. code-block:: bash

   pip install https://github.com/WenjieDu/PyPOTS/archive/main.zip

Required Dependencies
"""""""""""""""""""""
* Python >=3.7, <=3.10
* numpy
* scipy
* scikit-learn
* pandas <2.0.0
* torch >=1.10.0
* tensorboard
* h5py
* tsdb
* pycorruptor


Optional Dependencies
*********************
* torch-geometric (optional, required for GNN models like Raindrop)
* torch-scatter (optional, required for GNN models like Raindrop)
* torch-sparse (optional, required for GNN models like Raindrop)


Reasons of Version Limitations on Dependencies
**********************************************
* **Why we need python >=3.7?**
PyG (torch-geometric) is available starting from python v3.7, please refer to https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-via-anaconda.
Although torch-geometric is an optional dependency, we hope things go smoothly when our users opt to install it.

* **Why we need pandas <2.0.0?**

Because v2 may cause ``ModuleNotFoundError: No module named 'pandas.core.indexes.numeric'``,
see https://stackoverflow.com/questions/75953279/modulenotfounderror-no-module-named-pandas-core-indexes-numeric-using-metaflo.

* **Why we need PyTorch >1.10?**

Because of pytorch_sparse, please refer to https://github.com/rusty1s/pytorch_sparse/issues/207#issuecomment-1065549338.


Acceleration
""""""""""""
GPU Acceleration
****************
Neural-network models in PyPOTS are implemented in PyTorch. So far we only support CUDA-enabled GPUs for GPU acceleration.
If you have a CUDA device, you can install PyTorch with GPU support to accelerate the training and inference of neural-network models.
After that, you can set the ``device`` argument to ``"cuda"`` when initializing the model to enable GPU acceleration.
If you don't specify ``device``, PyPOTS will automatically detect and use the first CUDA device (i.e. ``cuda:0``) if multiple CUDA devices are available.

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
