"""
Test cases for BRITS classification model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import unittest

import pytest

from pypots.classification import BRITS
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.utils.metrics import cal_binary_classification_metrics
from tests.classification.config import (
    EPOCHS,
    TRAIN_SET,
    VAL_SET,
    TEST_SET,
    RESULT_SAVING_DIR_FOR_CLASSIFICATION,
)
from tests.global_test_config import (
    DATA,
    DEVICE,
    check_tb_and_model_checkpoints_existence,
)


class TestBRITS(unittest.TestCase):
    logger.info("Running tests for a classification model BRITS...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_CLASSIFICATION, "BRITS")
    model_save_name = "saved_BRITS_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a BRITS model
    brits = BRITS(
        DATA["n_steps"],
        DATA["n_features"],
        n_classes=DATA["n_classes"],
        rnn_hidden_size=256,
        epochs=EPOCHS,
        saving_path=saving_path,
        model_saving_strategy="better",
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="classification-brits")
    def test_0_fit(self):
        self.brits.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="classification-brits")
    def test_1_classify(self):
        predictions = self.brits.classify(TEST_SET)
        metrics = cal_binary_classification_metrics(predictions, DATA["test_y"])
        logger.info(
            f'ROC_AUC: {metrics["roc_auc"]}, \n'
            f'PR_AUC: {metrics["pr_auc"]},\n'
            f'F1: {metrics["f1"]},\n'
            f'Precision: {metrics["precision"]},\n'
            f'Recall: {metrics["recall"]},\n'
        )
        assert metrics["roc_auc"] >= 0.5, "ROC-AUC < 0.5"

    @pytest.mark.xdist_group(name="classification-brits")
    def test_2_parameters(self):
        assert hasattr(self.brits, "model") and self.brits.model is not None

        assert hasattr(self.brits, "optimizer") and self.brits.optimizer is not None

        assert hasattr(self.brits, "best_loss")
        self.assertNotEqual(self.brits.best_loss, float("inf"))

        assert (
            hasattr(self.brits, "best_model_dict")
            and self.brits.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="classification-brits")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.brits)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.brits.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.brits.load(saved_model_path)


if __name__ == "__main__":
    unittest.main()
