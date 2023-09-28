"""
Test configs for forecasting models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import os

from tests.global_test_config import (
    DATA,
    RESULT_SAVING_DIR,
)

EPOCHS = 5
N_PRED_STEP = 4

TRAIN_SET = {"X": DATA["train_X"]}
VAL_SET = {"X": DATA["val_X"]}
TEST_SET = {"X": DATA["test_X"][:, :-N_PRED_STEP]}
TEST_SET_INTACT = {"X": DATA["test_X_intact"]}

RESULT_SAVING_DIR_FOR_CLASSIFICATION = os.path.join(RESULT_SAVING_DIR, "forecasting")
