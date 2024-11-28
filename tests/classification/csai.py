"""
Test cases for CSAI classification model.
"""

# Created by Linglong Qian <linglong.qian@kcl.ac.uk>
# License: BSD-3-Clause

import os
import unittest

import pytest

from pypots.classification import CSAI
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.utils.metrics import calc_binary_classification_metrics
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


class TestCSAI(unittest.TestCase):
    logger.info("Running tests for a classification model CSAI...")

    # Set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_CLASSIFICATION, "CSAI")
    model_save_name = "saved_CSAI_model.pypots"

    # Initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # Initialize the CSAI model for classification
    csai = CSAI(
        n_steps=DATA["n_steps"],
        n_features=DATA["n_features"],
        n_classes=DATA["n_classes"],
        rnn_hidden_size=64,
        imputation_weight=0.7,
        consistency_weight=0.3,
        classification_weight=1.0,
        removal_percent=0.1,
        increase_factor=0.1,
        step_channels=16,
        epochs=EPOCHS,
        dropout=0.5,
        optimizer=optimizer,
        device=DEVICE,
        saving_path=saving_path,
        model_saving_strategy="better",
        verbose=True,
    )

    @pytest.mark.xdist_group(name="classification-csai")
    def test_0_fit(self):
        # Fit the CSAI model on the training and validation datasets
        self.csai.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="classification-csai")
    def test_1_classify(self):
        # Classify test set using the trained CSAI model
        results = self.csai.classify(TEST_SET)

        # Calculate binary classification metrics
        metrics = calc_binary_classification_metrics(results, DATA["test_y"])

        logger.info(
            f'CSAI ROC_AUC: {metrics["roc_auc"]}, '
            f'PR_AUC: {metrics["pr_auc"]}, '
            f'F1: {metrics["f1"]}, '
            f'Precision: {metrics["precision"]}, '
            f'Recall: {metrics["recall"]}'
        )

        assert metrics["roc_auc"] >= 0.5, "ROC-AUC < 0.5"

    @pytest.mark.xdist_group(name="classification-csai")
    def test_2_parameters(self):
        # Ensure that CSAI model parameters are properly initialized and trained
        assert hasattr(self.csai, "model") and self.csai.model is not None

        assert hasattr(self.csai, "optimizer") and self.csai.optimizer is not None

        assert hasattr(self.csai, "best_loss")
        self.assertNotEqual(self.csai.best_loss, float("inf"))

        assert hasattr(self.csai, "best_model_dict") and self.csai.best_model_dict is not None

    @pytest.mark.xdist_group(name="classification-csai")
    def test_3_saving_path(self):
        # Ensure the root saving directory exists
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # Check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.csai)

        # Save the trained model to file, and verify the file existence
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.csai.save(saved_model_path)

        # Test loading the saved model
        self.csai.load(saved_model_path)

    @pytest.mark.xdist_group(name="classification-csai")
    def test_4_lazy_loading(self):
        # Fit the CSAI model using lazy-loading datasets from H5 files
        self.csai.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)

        # Perform classification using lazy-loaded data
        results = self.csai.classify(GENERAL_H5_TEST_SET_PATH)

        # Calculate binary classification metrics
        metrics = calc_binary_classification_metrics(results, DATA["test_y"])

        logger.info(
            f'Lazy-loading CSAI ROC_AUC: {metrics["roc_auc"]}, '
            f'PR_AUC: {metrics["pr_auc"]}, '
            f'F1: {metrics["f1"]}, '
            f'Precision: {metrics["precision"]}, '
            f'Recall: {metrics["recall"]}'
        )

        assert metrics["roc_auc"] >= 0.5, "ROC-AUC < 0.5"


if __name__ == "__main__":
    unittest.main()
