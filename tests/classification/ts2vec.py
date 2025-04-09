"""
Test cases for TS2Vec classification model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import unittest

import pytest

from pypots.classification import TS2Vec
from pypots.nn.functional import calc_binary_classification_metrics
from pypots.optim import Adam
from pypots.utils.logging import logger
from tests.global_test_config import (
    DATA,
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


class TestTS2Vec(unittest.TestCase):
    logger.info("Running tests for a classification model TS2Vec...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_CLASSIFICATION, "TS2Vec")
    model_save_name = "saved_TS2Vec_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    d_vectorization = 2

    # initialize a TS2Vec model
    ts2vec = TS2Vec(
        DATA["n_steps"],
        DATA["n_features"],
        n_classes=DATA["n_classes"],
        n_output_dims=d_vectorization,
        d_hidden=64,
        n_layers=2,
        epochs=5,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="classification-ts2vec")
    def test_0_fit(self):
        self.ts2vec.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="classification-ts2vec")
    def test_1_classify(self):
        results = self.ts2vec.predict(TEST_SET, classifier_type="svm")
        metrics = calc_binary_classification_metrics(results["classification_proba"], DATA["test_y"])
        logger.info(
            f'TS2Vec+svm ROC_AUC: {metrics["roc_auc"]}, '
            f'PR_AUC: {metrics["pr_auc"]}, '
            f'F1: {metrics["f1"]}, '
            f'Precision: {metrics["precision"]}, '
            f'Recall: {metrics["recall"]}'
        )

        results = self.ts2vec.predict(TEST_SET, classifier_type="knn")
        metrics = calc_binary_classification_metrics(results["classification_proba"], DATA["test_y"])
        logger.info(
            f'TS2Vec+knn ROC_AUC: {metrics["roc_auc"]}, '
            f'PR_AUC: {metrics["pr_auc"]}, '
            f'F1: {metrics["f1"]}, '
            f'Precision: {metrics["precision"]}, '
            f'Recall: {metrics["recall"]}'
        )

        results = self.ts2vec.predict(TEST_SET, classifier_type="linear_regression")
        metrics = calc_binary_classification_metrics(results["classification_proba"], DATA["test_y"])
        logger.info(
            f'TS2Vec+linear_regression ROC_AUC: {metrics["roc_auc"]}, '
            f'PR_AUC: {metrics["pr_auc"]}, '
            f'F1: {metrics["f1"]}, '
            f'Precision: {metrics["precision"]}, '
            f'Recall: {metrics["recall"]}'
        )

        assert metrics["roc_auc"] >= 0.5, "ROC-AUC < 0.5"

    @pytest.mark.xdist_group(name="classification-ts2vec")
    def test_2_parameters(self):
        assert hasattr(self.ts2vec, "model") and self.ts2vec.model is not None

        assert hasattr(self.ts2vec, "optimizer") and self.ts2vec.optimizer is not None

        assert hasattr(self.ts2vec, "best_loss")
        self.assertNotEqual(self.ts2vec.best_loss, float("inf"))

        assert hasattr(self.ts2vec, "best_model_dict") and self.ts2vec.best_model_dict is not None

    @pytest.mark.xdist_group(name="classification-ts2vec")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.ts2vec)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.ts2vec.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.ts2vec.load(saved_model_path)

    @pytest.mark.xdist_group(name="classification-ts2vec")
    def test_4_lazy_loading(self):
        self.ts2vec.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        classification_proba = self.ts2vec.predict_proba(GENERAL_H5_TEST_SET_PATH)
        classification = self.ts2vec.classify(GENERAL_H5_TEST_SET_PATH)
        assert len(classification) == len(classification_proba)
        metrics = calc_binary_classification_metrics(classification_proba, DATA["test_y"])
        logger.info(
            f'Lazy-loading TS2Vec ROC_AUC: {metrics["roc_auc"]}, '
            f'PR_AUC: {metrics["pr_auc"]}, '
            f'F1: {metrics["f1"]}, '
            f'Precision: {metrics["precision"]}, '
            f'Recall: {metrics["recall"]}'
        )
        assert metrics["roc_auc"] >= 0.5, "ROC-AUC < 0.5"


if __name__ == "__main__":
    unittest.main()
