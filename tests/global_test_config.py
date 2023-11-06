"""
The global configurations for test cases.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os

import torch

from pypots.data.generating import gene_random_walk
from pypots.utils.logging import logger

# Generate the unified data for testing and cache it first, DATA here is a singleton
# Otherwise, file lock will cause bug if running test parallely with pytest-xdist.
DATA = gene_random_walk(
    n_steps=24,
    n_features=10,
    n_classes=2,
    n_samples_each_class=1000,
    missing_rate=0.1,
)

# The directory for saving the dataset into files for testing
DATA_SAVING_DIR = "h5data_for_tests"

# tensorboard and model files saving directory
RESULT_SAVING_DIR = "testing_results"


# set DEVICES to None if no cuda device is available, to avoid initialization failed while importing test classes
cuda_devices = [torch.device(i) for i in range(torch.cuda.device_count())]
if len(cuda_devices) > 2:
    logger.info("❗️Detected multiple cuda devices, using all of them to run testing.")
    DEVICE = cuda_devices
else:
    # if having no multiple cuda devices, leave it as None to use the default device
    DEVICE = None


def check_tb_and_model_checkpoints_existence(model):
    # check the tensorboard file existence
    saved_files = os.listdir(model.saving_path)
    if ".DS_Store" in saved_files:  # for macOS
        saved_files.remove(".DS_Store")
    assert (
        model.saving_path is not None and len(saved_files) > 0
    ), "tensorboard file does not exist"
    # check the model checkpoints existence
    saved_model_files = [i for i in saved_files if i.endswith(".pypots")]
    assert len(saved_model_files) > 0, "No model checkpoint saved."
