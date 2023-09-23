"""
Test configs for clustering models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import os

from tests.global_test_config import (
    DATA,
    RESULT_SAVING_DIR,
)


EPOCHS = 5

TRAIN_SET = {"X": DATA["train_X"]}
VAL_SET = {"X": DATA["val_X"]}
TEST_SET = {"X": DATA["test_X"]}

RESULT_SAVING_DIR_FOR_CLUSTERING = os.path.join(RESULT_SAVING_DIR, "clustering")
