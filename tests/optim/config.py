"""
Test configs for optimizers.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

from tests.global_test_config import DATA

TRAIN_SET = {"X": DATA["train_X"]}
VAL_SET = {
    "X": DATA["val_X"],
    "X_intact": DATA["val_X_intact"],
    "indicating_mask": DATA["val_X_indicating_mask"],
}
TEST_SET = {"X": DATA["test_X"]}


EPOCHS = 1
