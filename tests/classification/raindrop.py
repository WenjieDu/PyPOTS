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
from pypots.utils.metrics import calc_binary_classification_metrics
from tests.global_test_config import (
    DATA,
    EPOCHS,
    DEVICE,
    TRAIN_SET,
    VAL_SET,
    TEST_SET,
    H5_TRAIN_SET_PATH,
    H5_VAL_SET_PATH,
    H5_TEST_SET_PATH,
    RESULT_SAVING_DIR_FOR_CLASSIFICATION,
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
        metrics = calc_binary_classification_metrics(predictions, DATA["test_y"])
        logger.info(
            f'Lazy-loading Raindrop ROC_AUC: {metrics["roc_auc"]}, '
            f'PR_AUC: {metrics["pr_auc"]}, '
            f'F1: {metrics["f1"]}, '
            f'Precision: {metrics["precision"]}, '
            f'Recall: {metrics["recall"]}'
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
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.raindrop.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.raindrop.load(saved_model_path)

    @pytest.mark.xdist_group(name="classification-raindrop")
    def test_4_lazy_loading(self):
        self.raindrop.fit(H5_TRAIN_SET_PATH, H5_VAL_SET_PATH)
        results = self.raindrop.predict(H5_TEST_SET_PATH)
        metrics = calc_binary_classification_metrics(
            results["classification"], DATA["test_y"]
        )
        logger.info(
            f'Lazy-loading Raindrop ROC_AUC: {metrics["roc_auc"]}, '
            f'PR_AUC: {metrics["pr_auc"]}, '
            f'F1: {metrics["f1"]}, '
            f'Precision: {metrics["precision"]}, '
            f'Recall: {metrics["recall"]}'
        )
        assert metrics["roc_auc"] >= 0.5, "ROC-AUC < 0.5"


if __name__ == "__main__":
    unittest.main()
