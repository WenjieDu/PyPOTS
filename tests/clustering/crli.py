"""
Test cases for CRLI clustering model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


import os
import unittest

import pytest

from pypots.clustering import CRLI
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.utils.metrics import cal_rand_index, cal_cluster_purity
from pypots.utils.visualization import plot_clustering_results
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


class TestCRLI(unittest.TestCase):
    logger.info("Running tests for a clustering model CRLI...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_CLUSTERING, "CRLI")
    model_save_name = "saved_CRLI_model.pypots"

    # initialize an Adam optimizer
    G_optimizer = Adam(lr=0.001, weight_decay=1e-5)
    D_optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a CRLI model
    crli = CRLI(
        n_steps=DATA["n_steps"],
        n_features=DATA["n_features"],
        n_clusters=DATA["n_classes"],
        n_generator_layers=2,
        rnn_hidden_size=128,
        epochs=EPOCHS,
        saving_path=saving_path,
        G_optimizer=G_optimizer,
        D_optimizer=D_optimizer,
        device=DEVICE,
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
        external_metrics = cal_external_cluster_validation_metrics(
            clustering, DATA["test_y"]
        )
        logger.info(f"{external_metrics}")

    @pytest.mark.xdist_group(name="clustering-crli")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.crli)

        # save the trained model into file, and check if the path exists
        self.crli.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )

        # test loading the saved model, not necessary, but need to test
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.crli.load_model(saved_model_path)


if __name__ == "__main__":
    unittest.main()
