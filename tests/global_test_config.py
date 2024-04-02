"""
The global configurations for test cases.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os

import numpy as np
import torch

from pypots.data.generating import gene_random_walk
from pypots.data.saving import save_dict_into_h5
from pypots.utils.logging import logger
from pypots.utils.random import set_random_seed

set_random_seed(2023)

# Generate the unified data for testing and cache it first, DATA here is a singleton
# Otherwise, file lock will cause bug if running test parallely with pytest-xdist.
DATA = gene_random_walk(
    n_steps=24,
    n_features=10,
    n_classes=2,
    n_samples_each_class=1000,
    missing_rate=0.1,
)
# DATA = gene_physionet2012()

TRAIN_SET = {
    "X": DATA["train_X"],
    "y": DATA["train_y"].astype(float),
}
VAL_SET = {
    "X": DATA["val_X"],
    "X_ori": DATA["val_X_ori"],
    "y": DATA["val_y"].astype(float),
}
TEST_SET = {
    "X": DATA["test_X"],
    "X_ori": DATA["test_X_ori"],
    "y": DATA["test_y"].astype(float),
}

# tensorboard and model files saving directory
RESULT_SAVING_DIR = "testing_results"
RESULT_SAVING_DIR_FOR_IMPUTATION = os.path.join(RESULT_SAVING_DIR, "imputation")
RESULT_SAVING_DIR_FOR_CLASSIFICATION = os.path.join(RESULT_SAVING_DIR, "classification")
RESULT_SAVING_DIR_FOR_CLUSTERING = os.path.join(RESULT_SAVING_DIR, "clustering")
RESULT_SAVING_DIR_FOR_FORECASTING = os.path.join(RESULT_SAVING_DIR, "forecasting")

# set the number of epochs for all model training
EPOCHS = 2

# set DEVICES to None if no cuda device is available, to avoid initialization failed while importing test classes
n_cuda_devices = torch.cuda.device_count()
cuda_devices = [torch.device(i) for i in range(n_cuda_devices)]
if n_cuda_devices > 1:
    DEVICE = cuda_devices[np.random.randint(n_cuda_devices)]
    logger.info(
        f"❗️Detected multiple cuda devices, using one of them {DEVICE} to run testing."
    )
else:
    # if having no multiple cuda devices, leave it as None to use the default device
    DEVICE = None

# save the generated dataset into files for testing the lazy-loading strategy
DATA_SAVING_DIR = "h5data_for_tests"
H5_TRAIN_SET_PATH = f"{DATA_SAVING_DIR}/train_set.h5"
H5_VAL_SET_PATH = f"{DATA_SAVING_DIR}/val_set.h5"
H5_TEST_SET_PATH = f"{DATA_SAVING_DIR}/test_set.h5"


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


if __name__ == "__main__":
    if not os.path.exists(H5_TRAIN_SET_PATH):
        save_dict_into_h5(
            {
                "X": DATA["train_X"],
                "y": DATA["train_y"].astype(float),
            },
            H5_TRAIN_SET_PATH,
        )

    if not os.path.exists(H5_VAL_SET_PATH):
        save_dict_into_h5(
            {
                "X": DATA["val_X"],
                "X_ori": DATA["val_X_ori"],
                "y": DATA["val_y"].astype(float),
            },
            H5_VAL_SET_PATH,
        )

    if not os.path.exists(H5_TEST_SET_PATH):
        save_dict_into_h5(
            {
                "X": DATA["test_X"],
                "X_ori": DATA["test_X_ori"],
                "y": DATA["test_y"].astype(float),
            },
            H5_TEST_SET_PATH,
        )
