"""
Test cases for SeFT classification model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import unittest

import pytest

from pypots.classification import SeFT
from pypots.utils.logging import logger
from pypots.nn.functional import calc_binary_classification_metrics
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


class TestSeFT(unittest.TestCase):
    logger.info("Running tests for a classification model SeFT...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_CLASSIFICATION, "SeFT")
    model_save_name = "saved_SeFT_model.pypots"

    # initialize a SeFT model
    seft = SeFT(
        DATA["n_steps"],
        DATA["n_features"],
        DATA["n_classes"],
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ffn=64,
        n_seeds=2,
        dropout=0.1,
        epochs=EPOCHS,
        saving_path=saving_path,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="classification-seft")
    def test_0_fit(self):
        self.seft.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="classification-seft")
    def test_1_classify(self):
        results = self.seft.predict(TEST_SET)
        metrics = calc_binary_classification_metrics(results["classification_proba"], DATA["test_y"])
        logger.info(
            f'SeFT ROC_AUC: {metrics["roc_auc"]}, '
            f'PR_AUC: {metrics["pr_auc"]}, '
            f'F1: {metrics["f1"]}, '
            f'Precision: {metrics["precision"]}, '
            f'Recall: {metrics["recall"]}'
        )
        assert metrics["roc_auc"] >= 0.5, "ROC-AUC < 0.5"

    @pytest.mark.xdist_group(name="classification-seft")
    def test_2_parameters(self):
        assert hasattr(self.seft, "model") and self.seft.model is not None

        assert hasattr(self.seft, "optimizer") and self.seft.optimizer is not None

        assert hasattr(self.seft, "best_loss")
        self.assertNotEqual(self.seft.best_loss, float("inf"))

        assert hasattr(self.seft, "best_model_dict") and self.seft.best_model_dict is not None

    @pytest.mark.xdist_group(name="classification-seft")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.seft)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.seft.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.seft.load(saved_model_path)

    @pytest.mark.xdist_group(name="classification-seft")
    def test_4_lazy_loading(self):
        self.seft.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        classification_proba = self.seft.predict_proba(GENERAL_H5_TEST_SET_PATH)
        classification = self.seft.classify(GENERAL_H5_TEST_SET_PATH)
        assert len(classification) == len(classification_proba)
        metrics = calc_binary_classification_metrics(classification_proba, DATA["test_y"])
        logger.info(
            f'Lazy-loading SeFT ROC_AUC: {metrics["roc_auc"]}, '
            f'PR_AUC: {metrics["pr_auc"]}, '
            f'F1: {metrics["f1"]}, '
            f'Precision: {metrics["precision"]}, '
            f'Recall: {metrics["recall"]}'
        )
        assert metrics["roc_auc"] >= 0.5, "ROC-AUC < 0.5"


if __name__ == "__main__":
    unittest.main()
