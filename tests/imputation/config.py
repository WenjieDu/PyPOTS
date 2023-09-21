"""
Test configs for imputation models.
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
VAL_SET = {
    "X": DATA["val_X"],
    "X_intact": DATA["val_X_intact"],
    "indicating_mask": DATA["val_X_indicating_mask"],
}
TEST_SET = {"X": DATA["test_X"]}

RESULT_SAVING_DIR_FOR_IMPUTATION = os.path.join(RESULT_SAVING_DIR, "imputation")
