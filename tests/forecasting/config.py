"""
Test configs for forecasting models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from tests.global_test_config import DATA

EPOCHS = 5
N_PRED_STEP = 4

TRAIN_SET = {"X": DATA["train_X"]}
VAL_SET = {"X": DATA["val_X"]}
TEST_SET = {
    "X": DATA["test_X"][:, :-N_PRED_STEP],
    "X_ori": DATA["test_X_ori"],
}
