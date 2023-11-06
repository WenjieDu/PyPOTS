"""
Test configs for classification models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os

from tests.global_test_config import (
    DATA,
    RESULT_SAVING_DIR,
)

EPOCHS = 5

TRAIN_SET = {"X": DATA["train_X"], "y": DATA["train_y"]}
VAL_SET = {"X": DATA["val_X"], "y": DATA["val_y"]}
TEST_SET = {"X": DATA["test_X"]}

RESULT_SAVING_DIR_FOR_CLASSIFICATION = os.path.join(RESULT_SAVING_DIR, "classification")
