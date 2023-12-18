"""
Test configs for optimizers.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from tests.global_test_config import DATA

TRAIN_SET = {"X": DATA["train_X"]}
VAL_SET = {
    "X": DATA["val_X"],
    "X_ori": DATA["val_X_ori"],
}
TEST_SET = {"X": DATA["test_X"]}


EPOCHS = 1
