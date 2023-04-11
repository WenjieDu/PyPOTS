"""
The global configurations for test cases.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


from pypots.data.generating import gene_incomplete_random_walk_dataset

# Generate the unified data for testing and cache it first, DATA here is a singleton
# Otherwise, file lock will cause bug if running test parallely with pytest-xdist.
DATA = gene_incomplete_random_walk_dataset()

# The directory for saving the dataset into files for testing
DATA_SAVING_DIR = "h5data_for_tests"

# tensorboard and model files saving directory
RESULT_SAVING_DIR = "testing_results"
