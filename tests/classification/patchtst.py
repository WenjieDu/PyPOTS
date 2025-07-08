"""
Test cases for PatchTST classification model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import unittest

import pytest

from pypots.classification import PatchTST
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


class TestPatchTST(unittest.TestCase):
    logger.info("Running tests for a classification model PatchTST...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_CLASSIFICATION, "PatchTST")
    model_save_name = "saved_PatchTST_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a PatchTST model
    patchtst = PatchTST(
        DATA["n_steps"],
        DATA["n_features"],
        n_classes=DATA["n_classes"],
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_k=16,
        d_v=16,
        d_ffn=32,
        patch_size=DATA["n_steps"],
        patch_stride=8,
        dropout=0.1,
        attn_dropout=0,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="classification-patchtst")
    def test_0_fit(self):
        self.patchtst.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="classification-patchtst")
    def test_1_classify(self):
        results = self.patchtst.predict(TEST_SET)
        metrics = calc_binary_classification_metrics(results["classification_proba"], DATA["test_y"])
        logger.info(
            f'PatchTST ROC_AUC: {metrics["roc_auc"]}, '
            f'PR_AUC: {metrics["pr_auc"]}, '
            f'F1: {metrics["f1"]}, '
            f'Precision: {metrics["precision"]}, '
            f'Recall: {metrics["recall"]}'
        )
        assert metrics["roc_auc"] >= 0.5, "ROC-AUC < 0.5"

    @pytest.mark.xdist_group(name="classification-patchtst")
    def test_2_parameters(self):
        assert hasattr(self.patchtst, "model") and self.patchtst.model is not None

        assert hasattr(self.patchtst, "optimizer") and self.patchtst.optimizer is not None

        assert hasattr(self.patchtst, "best_loss")
        self.assertNotEqual(self.patchtst.best_loss, float("inf"))

        assert hasattr(self.patchtst, "best_model_dict") and self.patchtst.best_model_dict is not None

    @pytest.mark.xdist_group(name="classification-patchtst")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.patchtst)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.patchtst.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.patchtst.load(saved_model_path)

    @pytest.mark.xdist_group(name="classification-patchtst")
    def test_4_lazy_loading(self):
        self.patchtst.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        classification_proba = self.patchtst.predict_proba(GENERAL_H5_TEST_SET_PATH)
        classification = self.patchtst.classify(GENERAL_H5_TEST_SET_PATH)
        assert len(classification) == len(classification_proba)
        metrics = calc_binary_classification_metrics(classification_proba, DATA["test_y"])
        logger.info(
            f'Lazy-loading PatchTST ROC_AUC: {metrics["roc_auc"]}, '
            f'PR_AUC: {metrics["pr_auc"]}, '
            f'F1: {metrics["f1"]}, '
            f'Precision: {metrics["precision"]}, '
            f'Recall: {metrics["recall"]}'
        )
        assert metrics["roc_auc"] >= 0.5, "ROC-AUC < 0.5"


if __name__ == "__main__":
    unittest.main()
