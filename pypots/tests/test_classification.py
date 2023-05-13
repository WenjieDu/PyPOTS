"""
Test cases for classification models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import os
import unittest

import pytest

from pypots.classification import BRITS, GRUD, Raindrop
from pypots.tests.global_test_config import (
    DATA,
    RESULT_SAVING_DIR,
    check_tb_and_model_checkpoints_existence,
)
from pypots.utils.logging import logger
from pypots.utils.metrics import cal_binary_classification_metrics

EPOCHS = 5

TRAIN_SET = {"X": DATA["train_X"], "y": DATA["train_y"]}
VAL_SET = {"X": DATA["val_X"], "y": DATA["val_y"]}
TEST_SET = {"X": DATA["test_X"]}

RESULT_SAVING_DIR_FOR_CLASSIFICATION = os.path.join(RESULT_SAVING_DIR, "classification")


class TestBRITS(unittest.TestCase):
    logger.info("Running tests for a classification model BRITS...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_CLASSIFICATION, "BRITS")
    model_save_name = "saved_BRITS_model.pypots"

    # initialize a BRITS model
    brits = BRITS(
        DATA["n_steps"],
        DATA["n_features"],
        256,
        n_classes=DATA["n_classes"],
        epochs=EPOCHS,
        saving_path=saving_path,
        model_saving_strategy="better",
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
        self.brits.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )

        # test loading the saved model, not necessary, but need to test
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.brits.load_model(saved_model_path)


class TestGRUD(unittest.TestCase):
    logger.info("Running tests for a classification model GRUD...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_CLASSIFICATION, "GRUD")
    model_save_name = "saved_GRUD_model.pypots"

    # initialize a GRUD model
    grud = GRUD(
        DATA["n_steps"],
        DATA["n_features"],
        256,
        n_classes=DATA["n_classes"],
        epochs=EPOCHS,
        saving_path=saving_path,
    )

    @pytest.mark.xdist_group(name="classification-grud")
    def test_0_fit(self):
        self.grud.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="classification-grud")
    def test_1_classify(self):
        predictions = self.grud.classify(TEST_SET)
        metrics = cal_binary_classification_metrics(predictions, DATA["test_y"])
        logger.info(
            f'ROC_AUC: {metrics["roc_auc"]}, \n'
            f'PR_AUC: {metrics["pr_auc"]},\n'
            f'F1: {metrics["f1"]},\n'
            f'Precision: {metrics["precision"]},\n'
            f'Recall: {metrics["recall"]},\n'
        )
        assert metrics["roc_auc"] >= 0.5, "ROC-AUC < 0.5"

    @pytest.mark.xdist_group(name="classification-grud")
    def test_2_parameters(self):
        assert hasattr(self.grud, "model") and self.grud.model is not None

        assert hasattr(self.grud, "optimizer") and self.grud.optimizer is not None

        assert hasattr(self.grud, "best_loss")
        self.assertNotEqual(self.grud.best_loss, float("inf"))

        assert (
            hasattr(self.grud, "best_model_dict")
            and self.grud.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="classification-grud")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.grud)

        # save the trained model into file, and check if the path exists
        self.grud.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )

        # test loading the saved model, not necessary, but need to test
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.grud.load_model(saved_model_path)


class TestRaindrop(unittest.TestCase):
    logger.info("Running tests for a classification model Raindrop...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_CLASSIFICATION, "Raindrop")
    model_save_name = "saved_Raindrop_model.pypots"

    # initialize a Raindrop model
    raindrop = Raindrop(
        DATA["n_features"],
        2,
        DATA["n_features"] * 4,
        256,
        2,
        DATA["n_classes"],
        0.3,
        DATA["n_steps"],
        0,
        "mean",
        False,
        False,
        epochs=EPOCHS,
        saving_path=saving_path,
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
