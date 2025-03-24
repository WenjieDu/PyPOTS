"""
Test cases for TEFN classification model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import unittest

import pytest

from pypots.classification import TEFN
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


class TestTEFN(unittest.TestCase):
    logger.info("Running tests for a classification model TEFN...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_CLASSIFICATION, "TEFN")
    model_save_name = "saved_TEFN_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a TEFN model
    tefn = TEFN(
        DATA["n_steps"],
        DATA["n_features"],
        n_classes=DATA["n_classes"],
        n_fod=8,
        dropout=0.1,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="classification-tefn")
    def test_0_fit(self):
        self.tefn.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="classification-tefn")
    def test_1_classify(self):
        results = self.tefn.predict(TEST_SET)
        metrics = calc_binary_classification_metrics(results["classification_proba"], DATA["test_y"])
        logger.info(
            f'TEFN ROC_AUC: {metrics["roc_auc"]}, '
            f'PR_AUC: {metrics["pr_auc"]}, '
            f'F1: {metrics["f1"]}, '
            f'Precision: {metrics["precision"]}, '
            f'Recall: {metrics["recall"]}'
        )
        assert metrics["roc_auc"] >= 0.5, "ROC-AUC < 0.5"

    @pytest.mark.xdist_group(name="classification-tefn")
    def test_2_parameters(self):
        assert hasattr(self.tefn, "model") and self.tefn.model is not None

        assert hasattr(self.tefn, "optimizer") and self.tefn.optimizer is not None

        assert hasattr(self.tefn, "best_loss")
        self.assertNotEqual(self.tefn.best_loss, float("inf"))

        assert hasattr(self.tefn, "best_model_dict") and self.tefn.best_model_dict is not None

    @pytest.mark.xdist_group(name="classification-tefn")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.tefn)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.tefn.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.tefn.load(saved_model_path)

    @pytest.mark.xdist_group(name="classification-tefn")
    def test_4_lazy_loading(self):
        self.tefn.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        classification_proba = self.tefn.predict_proba(GENERAL_H5_TEST_SET_PATH)
        classification = self.tefn.classify(GENERAL_H5_TEST_SET_PATH)
        assert len(classification) == len(classification_proba)
        metrics = calc_binary_classification_metrics(classification_proba, DATA["test_y"])
        logger.info(
            f'Lazy-loading TEFN ROC_AUC: {metrics["roc_auc"]}, '
            f'PR_AUC: {metrics["pr_auc"]}, '
            f'F1: {metrics["f1"]}, '
            f'Precision: {metrics["precision"]}, '
            f'Recall: {metrics["recall"]}'
        )
        assert metrics["roc_auc"] >= 0.5, "ROC-AUC < 0.5"


if __name__ == "__main__":
    unittest.main()
