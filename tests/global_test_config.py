"""
The global configurations for test cases.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import os

from pypots.data.generating import gene_incomplete_random_walk_dataset

# Generate the unified data for testing and cache it first, DATA here is a singleton
# Otherwise, file lock will cause bug if running test parallely with pytest-xdist.
DATA = gene_incomplete_random_walk_dataset(n_steps=60, n_features=10)

# The directory for saving the dataset into files for testing
DATA_SAVING_DIR = "h5data_for_tests"

# tensorboard and model files saving directory
RESULT_SAVING_DIR = "testing_results"


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
