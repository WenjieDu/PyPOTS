"""
Test cases for Raindrop classification model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import unittest

import pytest

from pypots.classification import Raindrop
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


class TestRaindrop(unittest.TestCase):
    logger.info("Running tests for a classification model Raindrop...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_CLASSIFICATION, "Raindrop")
    model_save_name = "saved_Raindrop_model.pypots"

    # initialize a Raindrop model
    raindrop = Raindrop(
        DATA["n_steps"],
        DATA["n_features"],
        DATA["n_classes"],
        n_layers=2,
        d_model=DATA["n_features"] * 4,
        d_inner=256,
        n_heads=2,
        dropout=0.3,
        d_static=0,
        aggregation="mean",
        sensor_wise_mask=False,
        static=False,
        epochs=EPOCHS,
        saving_path=saving_path,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="classification-raindrop")
    def test_0_fit(self):
        self.raindrop.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="classification-raindrop")
    def test_1_classify(self):
        predictions = self.raindrop.classify(TEST_SET)
        metrics = cal_binary_classification_metrics(predictions, DATA["test_y"])
        logger.info(
            f'ROC_AUC: {metrics["roc_auc"]}, \n'
            f'PR_AUC: {metrics["pr_auc"]},\n'
            f'F1: {metrics["f1"]},\n'
            f'Precision: {metrics["precision"]},\n'
            f'Recall: {metrics["recall"]},\n'
        )
        assert metrics["roc_auc"] >= 0.5, "ROC-AUC < 0.5"

    @pytest.mark.xdist_group(name="classification-raindrop")
    def test_2_parameters(self):
        assert hasattr(self.raindrop, "model") and self.raindrop.model is not None

        assert (
            hasattr(self.raindrop, "optimizer") and self.raindrop.optimizer is not None
        )

        assert hasattr(self.raindrop, "best_loss")
        self.assertNotEqual(self.raindrop.best_loss, float("inf"))

        assert (
            hasattr(self.raindrop, "best_model_dict")
            and self.raindrop.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="classification-raindrop")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.raindrop)

        # save the trained model into file, and check if the path exists
        self.raindrop.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )

        # test loading the saved model, not necessary, but need to test
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.raindrop.load_model(saved_model_path)


if __name__ == "__main__":
    unittest.main()
