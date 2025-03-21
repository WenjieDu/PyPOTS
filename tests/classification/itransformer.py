"""
Test cases for iTransformer classification model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import unittest

import pytest

from pypots.classification import iTransformer
from pypots.nn.functional import calc_binary_classification_metrics
from pypots.optim import Adam
from pypots.utils.logging import logger
from tests.global_test_config import (
    DATA,
    EPOCHS,
    DEVICE,
    TRAIN_SET,
    VAL_SET,
    TEST_SET,
    GENERAL_H5_TRAIN_SET_PATH,
    GENERAL_H5_VAL_SET_PATH,
    GENERAL_H5_TEST_SET_PATH,
    RESULT_SAVING_DIR_FOR_CLASSIFICATION,
    check_tb_and_model_checkpoints_existence,
)


class TestiTransformer(unittest.TestCase):
    logger.info("Running tests for a classification model iTransformer...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_CLASSIFICATION, "iTransformer")
    model_save_name = "saved_iTransformer_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a iTransformer model
    itransformer = iTransformer(
        DATA["n_steps"],
        DATA["n_features"],
        n_classes=DATA["n_classes"],
        n_layers=2,
        d_model=32,
        n_heads=2,
        d_k=16,
        d_v=16,
        d_ffn=32,
        dropout=0.1,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="classification-itransformer")
    def test_0_fit(self):
        self.itransformer.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="classification-itransformer")
    def test_1_classify(self):
        results = self.itransformer.predict(TEST_SET)
        metrics = calc_binary_classification_metrics(results["classification"], DATA["test_y"])
        logger.info(
            f'iTransformer ROC_AUC: {metrics["roc_auc"]}, '
            f'PR_AUC: {metrics["pr_auc"]}, '
            f'F1: {metrics["f1"]}, '
            f'Precision: {metrics["precision"]}, '
            f'Recall: {metrics["recall"]}'
        )
        assert metrics["roc_auc"] >= 0.5, "ROC-AUC < 0.5"

    @pytest.mark.xdist_group(name="classification-itransformer")
    def test_2_parameters(self):
        assert hasattr(self.itransformer, "model") and self.itransformer.model is not None

        assert hasattr(self.itransformer, "optimizer") and self.itransformer.optimizer is not None

        assert hasattr(self.itransformer, "best_loss")
        self.assertNotEqual(self.itransformer.best_loss, float("inf"))

        assert hasattr(self.itransformer, "best_model_dict") and self.itransformer.best_model_dict is not None

    @pytest.mark.xdist_group(name="classification-itransformer")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.itransformer)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.itransformer.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.itransformer.load(saved_model_path)

    @pytest.mark.xdist_group(name="classification-itransformer")
    def test_4_lazy_loading(self):
        self.itransformer.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        results = self.itransformer.predict(GENERAL_H5_TEST_SET_PATH)
        metrics = calc_binary_classification_metrics(results["classification"], DATA["test_y"])
        logger.info(
            f'Lazy-loading iTransformer ROC_AUC: {metrics["roc_auc"]}, '
            f'PR_AUC: {metrics["pr_auc"]}, '
            f'F1: {metrics["f1"]}, '
            f'Precision: {metrics["precision"]}, '
            f'Recall: {metrics["recall"]}'
        )
        assert metrics["roc_auc"] >= 0.5, "ROC-AUC < 0.5"


if __name__ == "__main__":
    unittest.main()
