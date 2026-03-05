Installation
============

How to Install
""""""""""""""
It is recommended to use **pip** or **conda** for PyPOTS installation as shown below:

.. code-block:: bash

    # via pip
    pip install pypots            # the first time installation
    pip install pypots --upgrade  # update pypots to the latest version
    # install from the latest source code with the latest features but may be not officially released yet
    pip install https://github.com/WenjieDu/PyPOTS/archive/main.zip

    # via conda
    conda install conda-forge::pypots  # the first time installation
    conda update  conda-forge::pypots  # update pypots to the latest version

    # via docker
    docker run -it --name pypots wenjiedu/pypots  # docker will auto pull our built image and run a instance for you
    # after things settled, you can run python in the container to access the well-configured environment for running pypots
    # if you'd like to detach from the container, press ctrl-P + ctrl-Q
    # run `docker attach pypots` to enter the container again.


Required Dependencies
"""""""""""""""""""""
* Python >=3.8
* h5py
* numpy
* scipy
* sympy
* einops
* pandas
* seaborn
* matplotlib
* tensorboard
* scikit-learn
* transformers
* torch >=1.10.0
* tsdb >=0.7.1
* pygrinder >=0.7
* benchpots >=0.4
* ai4ts


Optional Dependencies
*********************
* torch-geometric (optional, required for GNN models like Raindrop)
* torch-scatter (optional, required for GNN models like Raindrop)
* torch-sparse (optional, required for GNN models like Raindrop)


Reasons of Version Limitations on Dependencies
**********************************************
* **Why we need python >=3.8?**

Python v3.6 and before versions have no longer been supported officially (check out `status of Python versions here <https://devguide.python.org/versions/>`_).
Besides, PyG (torch-geometric) is available for Python >= v3.7 (refer to https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-via-anaconda ).
Although torch-geometric is an optional dependency, we hope things go smoothly when our users opt to install it.
In addition, note that Python v.3.7 has also been in the end-of-life status since 2023-06-27.
Hence, we raise the minimum support Python version to v3.8.

* **Why we need PyTorch >=1.10?**

Because of pytorch_sparse, please refer to https://github.com/rusty1s/pytorch_sparse/issues/207#issuecomment-1065549338.

Acceleration
""""""""""""
GPU Acceleration
****************
Neural-network models in PyPOTS are implemented in PyTorch. So far we only support CUDA-enabled GPUs for GPU acceleration.
If you have a CUDA device, you can install PyTorch with GPU support to accelerate the training and inference of neural-network models.
After that, you can set the ``device`` argument to ``"cuda"`` when initializing the model to enable GPU acceleration.
If you don't specify ``device``, PyPOTS will automatically detect and use the default CUDA device if multiple CUDA devices are available.

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
