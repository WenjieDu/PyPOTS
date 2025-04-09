"""
The global configurations for test cases.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os

import numpy as np
import torch
from benchpots.datasets import preprocess_random_walk

from pypots.data.saving import save_dict_into_h5
from pypots.utils.logging import logger
from pypots.utils.random import set_random_seed

# set the random seed for all test cases
RANDOM_SEED = 2023
# set the number of epochs for all model training
EPOCHS = 2
# set the number of prediction steps for forecasting models
N_STEPS = 6
N_PRED_STEPS = 2
N_FEATURES = 5
N_CLASSES = 2
N_SAMPLES_PER_CLASS = 100
ANOMALY_RATE = 0.05
MISSING_RATE = 0.1
# tensorboard and model files saving directory
RESULT_SAVING_DIR = "testing_results"
MODEL_SAVING_DIR = f"{RESULT_SAVING_DIR}/models"
DATA_SAVING_DIR = f"{RESULT_SAVING_DIR}/datasets"
RESULT_SAVING_DIR_FOR_IMPUTATION = os.path.join(MODEL_SAVING_DIR, "imputation")
RESULT_SAVING_DIR_FOR_ANOMALY_DETECTION = os.path.join(MODEL_SAVING_DIR, "anomaly_detection")
RESULT_SAVING_DIR_FOR_CLASSIFICATION = os.path.join(MODEL_SAVING_DIR, "classification")
RESULT_SAVING_DIR_FOR_CLUSTERING = os.path.join(MODEL_SAVING_DIR, "clustering")
RESULT_SAVING_DIR_FOR_FORECASTING = os.path.join(MODEL_SAVING_DIR, "forecasting")
RESULT_SAVING_DIR_FOR_REPRESENTATION = os.path.join(MODEL_SAVING_DIR, "representation")
# paths to save the generated dataset into files for testing the lazy-loading strategy
GENERAL_DATA_SAVING_DIR = f"{DATA_SAVING_DIR}/general_h5dataset"
GENERAL_H5_TRAIN_SET_PATH = os.path.abspath(f"{GENERAL_DATA_SAVING_DIR}/train_set.h5")
GENERAL_H5_VAL_SET_PATH = os.path.abspath(f"{GENERAL_DATA_SAVING_DIR}/val_set.h5")
GENERAL_H5_TEST_SET_PATH = os.path.abspath(f"{GENERAL_DATA_SAVING_DIR}/test_set.h5")
# paths to save the generated dataset for testing forecasting models with the lazy-loading strategy
FORECASTING_DATA_SAVING_DIR = f"{DATA_SAVING_DIR}/forecasting_h5dataset"
FORECASTING_H5_TRAIN_SET_PATH = os.path.abspath(f"{FORECASTING_DATA_SAVING_DIR}/train_set.h5")
FORECASTING_H5_VAL_SET_PATH = os.path.abspath(f"{FORECASTING_DATA_SAVING_DIR}/val_set.h5")
FORECASTING_H5_TEST_SET_PATH = os.path.abspath(f"{FORECASTING_DATA_SAVING_DIR}/test_set.h5")


set_random_seed(RANDOM_SEED)

# Generate the unified data for testing and cache it first, DATA here is a singleton
# Otherwise, file lock will cause bug if running test parallely with pytest-xdist.
DATA = preprocess_random_walk(
    n_steps=N_STEPS + N_PRED_STEPS,  # the total sequence length
    n_features=N_FEATURES,
    n_classes=N_CLASSES,
    n_samples_each_class=N_SAMPLES_PER_CLASS,
    anomaly_rate=ANOMALY_RATE,
    missing_rate=MISSING_RATE,
)
# from benchpots.datasets import preprocess_physionet2012
# DATA = preprocess_physionet2012(subset="set-a", rate=0.1)

DATA["anomaly_rate"] = ANOMALY_RATE
DATA["test_X_indicating_mask"] = np.isnan(DATA["test_X"]) ^ np.isnan(DATA["test_X_ori"])
DATA["test_X_ori"] = np.nan_to_num(DATA["test_X_ori"])

# for imputation/classification/clustering tasks, we feed all N_STEPS + N_PRED_STEPS data into the model
TRAIN_SET = {
    "X": DATA["train_X"],
    "y": DATA["train_y"].astype(float),
    "anomaly_y": DATA["train_anomaly_y"].astype(float),
}
VAL_SET = {
    "X": DATA["val_X"],
    "X_ori": DATA["val_X_ori"],
    "y": DATA["val_y"].astype(float),
    "anomaly_y": DATA["val_anomaly_y"].astype(float),
}
TEST_SET = {
    "X": DATA["test_X"],
    "X_ori": DATA["test_X_ori"],
    "y": DATA["test_y"].astype(float),
    "anomaly_y": DATA["test_anomaly_y"].astype(float),
}
# for forecasting task, we feed only N_STEPS data into the model and let it predict the following N_PRED_STEPS
FORECASTING_TRAIN_SET = {
    "X": DATA["train_X"][:, :-N_PRED_STEPS],
    "X_pred": DATA["train_X"][:, -N_PRED_STEPS:],
}
FORECASTING_VAL_SET = {
    "X": DATA["val_X"][:, :-N_PRED_STEPS],
    "X_pred": DATA["val_X_ori"][:, -N_PRED_STEPS:],
}
FORECASTING_TEST_SET = {
    "X": DATA["test_X"][:, :-N_PRED_STEPS],
    "X_pred": DATA["test_X_ori"][:, -N_PRED_STEPS:],
}

# set DEVICES to None if no cuda device is available, to avoid initialization failed while importing test classes
n_cuda_devices = torch.cuda.device_count()
cuda_devices = [torch.device(i) for i in range(n_cuda_devices)]
if n_cuda_devices > 1:
    DEVICE = np.asarray(cuda_devices)[np.random.randint(n_cuda_devices, size=1)].tolist()
    logger.info(f"❗️Detected multiple cuda devices, using {DEVICE} to run testing.")
else:
    # if having no multiple cuda devices, leave it as None to use the default device
    DEVICE = None

# DEVICE = ["cuda:1"]
# DEVICE = ["cuda:1", "cuda:2"]
# DEVICE = "cpu"
# DEVICE = "mps"


def check_tb_and_model_checkpoints_existence(model):
    # check the tensorboard file existence
    saved_files = os.listdir(model.saving_path)
    if ".DS_Store" in saved_files:  # for macOS
        saved_files.remove(".DS_Store")
    assert model.saving_path is not None and len(saved_files) > 0, "tensorboard file does not exist"
    # check the model checkpoints existence
    saved_model_files = [i for i in saved_files if i.endswith(".pypots")]
    assert len(saved_model_files) > 0, "No model checkpoint saved."


if __name__ == "__main__":
    if not os.path.exists(GENERAL_H5_TRAIN_SET_PATH):
        save_dict_into_h5(TRAIN_SET, GENERAL_H5_TRAIN_SET_PATH)
    if not os.path.exists(GENERAL_H5_VAL_SET_PATH):
        save_dict_into_h5(VAL_SET, GENERAL_H5_VAL_SET_PATH)
    if not os.path.exists(GENERAL_H5_TEST_SET_PATH):
        save_dict_into_h5(TEST_SET, GENERAL_H5_TEST_SET_PATH)

    if not os.path.exists(FORECASTING_H5_TRAIN_SET_PATH):
        save_dict_into_h5(FORECASTING_TRAIN_SET, FORECASTING_H5_TRAIN_SET_PATH)
    if not os.path.exists(FORECASTING_H5_VAL_SET_PATH):
        save_dict_into_h5(FORECASTING_VAL_SET, FORECASTING_H5_VAL_SET_PATH)
    if not os.path.exists(FORECASTING_H5_TEST_SET_PATH):
        save_dict_into_h5(FORECASTING_TEST_SET, FORECASTING_H5_TEST_SET_PATH)

    logger.info(f"Files under GENERAL_DATA_SAVING_DIR: {os.listdir(GENERAL_DATA_SAVING_DIR)}")
    logger.info(f"Files under FORECASTING_DATA_SAVING_DIR: {os.listdir(FORECASTING_DATA_SAVING_DIR)}")
