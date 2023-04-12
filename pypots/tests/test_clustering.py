"""
Test cases for clustering models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


import unittest

import numpy as np
import pytest
import os

from pypots.clustering import VaDER, CRLI
from pypots.tests.global_test_config import DATA, RESULT_SAVING_DIR
from pypots.utils.logging import logger
from pypots.utils.metrics import cal_rand_index, cal_cluster_purity

EPOCHS = 5

TRAIN_SET = {"X": DATA["train_X"]}
VAL_SET = {"X": DATA["val_X"]}
TEST_SET = {"X": DATA["test_X"]}

RESULT_SAVING_DIR_FOR_CLUSTERING = os.path.join(RESULT_SAVING_DIR, "clustering")


class TestCRLI(unittest.TestCase):
    logger.info("Running tests for a clustering model CRLI...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_CLUSTERING, "CRLI")
    model_save_name = "saved_CRLI_model.pypots"

    # initialize a CRLI model
    crli = CRLI(
        n_steps=DATA["n_steps"],
        n_features=DATA["n_features"],
        n_clusters=DATA["n_classes"],
        n_generator_layers=2,
        rnn_hidden_size=128,
        epochs=EPOCHS,
        tb_file_saving_path=saving_path,
    )

    @pytest.mark.xdist_group(name="clustering-crli")
    def test_0_fit(self):
        self.crli.fit(TRAIN_SET)

    @pytest.mark.xdist_group(name="clustering-crli")
    def test_1_parameters(self):
        assert hasattr(self.crli, "model") and self.crli.model is not None

        assert hasattr(self.crli, "G_optimizer") and self.crli.G_optimizer is not None
        assert hasattr(self.crli, "D_optimizer") and self.crli.D_optimizer is not None

        assert hasattr(self.crli, "best_loss")
        self.assertNotEqual(self.crli.best_loss, float("inf"))

        assert (
            hasattr(self.crli, "best_model_dict")
            and self.crli.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="clustering-crli")
    def test_2_cluster(self):
        clustering = self.crli.cluster(TEST_SET)
        RI = cal_rand_index(clustering, DATA["test_y"])
        CP = cal_cluster_purity(clustering, DATA["test_y"])
        logger.info(f"RI: {RI}\nCP: {CP}")

    @pytest.mark.xdist_group(name="clustering-crli")
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
        self.crli.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        assert os.path.exists(
            saved_model_path
        ), f"file {self.saving_path} does not exist, model not saved"


class TestVaDER(unittest.TestCase):
    logger.info("Running tests for a clustering model Transformer...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_CLUSTERING, "VaDER")
    model_save_name = "saved_VaDER_model.pypots"

    # initialize a VaDER model
    vader = VaDER(
        n_steps=DATA["n_steps"],
        n_features=DATA["n_features"],
        n_clusters=DATA["n_classes"],
        rnn_hidden_size=64,
        d_mu_stddev=5,
        pretrain_epochs=20,
        epochs=EPOCHS,
        tb_file_saving_path=saving_path,
    )

    @pytest.mark.xdist_group(name="clustering-vader")
    def test_0_fit(self):
        self.vader.fit(TRAIN_SET)

    @pytest.mark.xdist_group(name="clustering-vader")
    def test_1_cluster(self):
        try:
            clustering = self.vader.cluster(TEST_SET)
            RI = cal_rand_index(clustering, DATA["test_y"])
            CP = cal_cluster_purity(clustering, DATA["test_y"])
            logger.info(f"RI: {RI}\nCP: {CP}")
        except np.linalg.LinAlgError as e:
            logger.error(
                f"{e}\n"
                "Got singular matrix, please try to retrain the model to fix this"
            )

    @pytest.mark.xdist_group(name="clustering-vader")
    def test_2_parameters(self):
        assert hasattr(self.vader, "model") and self.vader.model is not None

        assert hasattr(self.vader, "optimizer") and self.vader.optimizer is not None

        assert hasattr(self.vader, "best_loss")
        self.assertNotEqual(self.vader.best_loss, float("inf"))

        assert (
            hasattr(self.vader, "best_model_dict")
            and self.vader.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="clustering-vader")
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
        self.vader.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        assert os.path.exists(
            saved_model_path
        ), f"file {self.saving_path} does not exist, model not saved"


if __name__ == "__main__":
    unittest.main()
