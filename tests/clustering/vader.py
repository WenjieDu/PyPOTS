"""
Test cases for VaDER clustering model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


import os
import unittest

import numpy as np
import pytest

from pypots.clustering import VaDER
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.utils.metrics import cal_rand_index, cal_cluster_purity
from tests.clustering.config import (
    EPOCHS,
    TRAIN_SET,
    TEST_SET,
    RESULT_SAVING_DIR_FOR_CLUSTERING,
)
from tests.global_test_config import (
    DATA,
    DEVICE,
    check_tb_and_model_checkpoints_existence,
)


class TestVaDER(unittest.TestCase):
    logger.info("Running tests for a clustering model Transformer...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_CLUSTERING, "VaDER")
    model_save_name = "saved_VaDER_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a VaDER model
    vader = VaDER(
        n_steps=DATA["n_steps"],
        n_features=DATA["n_features"],
        n_clusters=DATA["n_classes"],
        rnn_hidden_size=64,
        d_mu_stddev=5,
        pretrain_epochs=20,
        epochs=EPOCHS,
        optimizer=optimizer,
        saving_path=saving_path,
        device=DEVICE,
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

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.vader)

        # save the trained model into file, and check if the path exists
        self.vader.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )

        # test loading the saved model, not necessary, but need to test
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.vader.load_model(saved_model_path)


if __name__ == "__main__":
    unittest.main()
