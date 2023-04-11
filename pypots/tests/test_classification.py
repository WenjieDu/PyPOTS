"""
Test cases for classification models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import os
import unittest

import pytest

from pypots.classification import BRITS, GRUD, Raindrop
from pypots.tests.global_test_config import DATA, RESULT_SAVING_DIR
from pypots.utils.logging import logger
from pypots.utils.metrics import cal_binary_classification_metrics

EPOCHS = 5

TRAIN_SET = {"X": DATA["train_X"], "y": DATA["train_y"]}
VAL_SET = {"X": DATA["val_X"], "y": DATA["val_y"]}
TEST_SET = {"X": DATA["test_X"]}

RESULT_SAVING_DIR_FOR_CLASSIFICATION = f"{RESULT_SAVING_DIR}/classification"


class TestBRITS(unittest.TestCase):
    logger.info("Running tests for a classification model BRITS...")

    # set the log and model saving path
    saving_path = f"{RESULT_SAVING_DIR_FOR_CLASSIFICATION}/BRITS"
    model_save_name = "saved_BRITS_model.pypots"

    # initialize a BRITS model
    brits = BRITS(
        DATA["n_steps"],
        DATA["n_features"],
        256,
        n_classes=DATA["n_classes"],
        epochs=EPOCHS,
        tb_file_saving_path=saving_path,
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
        # whether the tensorboard file exists
        files = os.listdir(self.saving_path)
        assert len(files) > 0, "tensorboard dir does not exist"
        tensorboard_dir_name = files[0]
        tensorboard_dir_path = os.path.join(self.saving_path, tensorboard_dir_name)
        assert (
            tensorboard_dir_name.startswith("tensorboard")
            and len(os.listdir(tensorboard_dir_path)) > 0
        ), "tensorboard file does not exist"

        # save the trained model into file, and check if the path exists
        self.brits.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        assert os.path.exists(
            saved_model_path
        ), f"file {self.saving_path} does not exist, model not saved"


class TestGRUD(unittest.TestCase):
    logger.info("Running tests for a classification model GRUD...")

    # set the log and model saving path
    saving_path = f"{RESULT_SAVING_DIR_FOR_CLASSIFICATION}/GRUD"
    model_save_name = "saved_GRUD_model.pypots"

    # initialize a GRUD model
    grud = GRUD(
        DATA["n_steps"],
        DATA["n_features"],
        256,
        n_classes=DATA["n_classes"],
        epochs=EPOCHS,
        tb_file_saving_path=saving_path,
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
        # whether the tensorboard file exists
        files = os.listdir(self.saving_path)
        assert len(files) > 0, "tensorboard dir does not exist"
        tensorboard_dir_name = files[0]
        tensorboard_dir_path = os.path.join(self.saving_path, tensorboard_dir_name)
        assert (
            tensorboard_dir_name.startswith("tensorboard")
            and len(os.listdir(tensorboard_dir_path)) > 0
        ), "tensorboard file does not exist"

        # save the trained model into file, and check if the path exists
        self.grud.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        assert os.path.exists(
            saved_model_path
        ), f"file {self.saving_path} does not exist, model not saved"


class TestRaindrop(unittest.TestCase):
    logger.info("Running tests for a classification model Raindrop...")

    # set the log and model saving path
    saving_path = f"{RESULT_SAVING_DIR_FOR_CLASSIFICATION}/Raindrop"
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
        tb_file_saving_path=saving_path,
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
        # whether the tensorboard file exists
        files = os.listdir(self.saving_path)
        assert len(files) > 0, "tensorboard dir does not exist"
        tensorboard_dir_name = files[0]
        tensorboard_dir_path = os.path.join(self.saving_path, tensorboard_dir_name)
        assert (
            tensorboard_dir_name.startswith("tensorboard")
            and len(os.listdir(tensorboard_dir_path)) > 0
        ), "tensorboard file does not exist"

        # save the trained model into file, and check if the path exists
        self.raindrop.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        assert os.path.exists(
            saved_model_path
        ), f"file {self.saving_path} does not exist, model not saved"


if __name__ == "__main__":
    unittest.main()
