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
from pypots.utils.metrics import (
    cal_external_cluster_validation_metrics,
    cal_internal_cluster_validation_metrics,
)
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
    crli_gru = CRLI(
        n_steps=DATA["n_steps"],
        n_features=DATA["n_features"],
        n_clusters=DATA["n_classes"],
        n_generator_layers=2,
        rnn_hidden_size=128,
        rnn_cell_type="GRU",
        epochs=EPOCHS,
        saving_path=saving_path,
        G_optimizer=G_optimizer,
        D_optimizer=D_optimizer,
        device=DEVICE,
    )

    crli_lstm = CRLI(
        n_steps=DATA["n_steps"],
        n_features=DATA["n_features"],
        n_clusters=DATA["n_classes"],
        n_generator_layers=2,
        rnn_hidden_size=128,
        rnn_cell_type="LSTM",
        epochs=EPOCHS,
        saving_path=saving_path,
        G_optimizer=G_optimizer,
        D_optimizer=D_optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="clustering-crli")
    def test_0_fit(self):
        logger.info("Training CRLI-GRU...")
        self.crli_gru.fit(TRAIN_SET)
        logger.info("Training CRLI-LSTM...")
        self.crli_lstm.fit(TRAIN_SET)

    @pytest.mark.xdist_group(name="clustering-crli")
    def test_1_parameters(self):
        # GRU cell
        assert hasattr(self.crli_gru, "model") and self.crli_gru.model is not None

        assert (
            hasattr(self.crli_gru, "G_optimizer")
            and self.crli_gru.G_optimizer is not None
        )
        assert (
            hasattr(self.crli_gru, "D_optimizer")
            and self.crli_gru.D_optimizer is not None
        )

        assert hasattr(self.crli_gru, "best_loss")
        self.assertNotEqual(self.crli_gru.best_loss, float("inf"))

        assert (
            hasattr(self.crli_gru, "best_model_dict")
            and self.crli_gru.best_model_dict is not None
        )

        # LSTM cell
        assert hasattr(self.crli_lstm, "model") and self.crli_lstm.model is not None

        assert (
            hasattr(self.crli_lstm, "G_optimizer")
            and self.crli_lstm.G_optimizer is not None
        )
        assert (
            hasattr(self.crli_lstm, "D_optimizer")
            and self.crli_lstm.D_optimizer is not None
        )

        assert hasattr(self.crli_lstm, "best_loss")
        self.assertNotEqual(self.crli_lstm.best_loss, float("inf"))

        assert (
            hasattr(self.crli_lstm, "best_model_dict")
            and self.crli_lstm.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="clustering-crli")
    def test_2_cluster(self):
        # GRU cell
        clustering, latent_collector = self.crli_gru.cluster(
            TEST_SET, return_latent=True
        )
        external_metrics = cal_external_cluster_validation_metrics(
            clustering, DATA["test_y"]
        )
        internal_metrics = cal_internal_cluster_validation_metrics(
            latent_collector["clustering_latent"], DATA["test_y"]
        )
        logger.info(f"CRLI-GRU: {external_metrics}")
        logger.info(f"CRLI-GRU:{internal_metrics}")

        # LSTM cell
        clustering, latent_collector = self.crli_lstm.cluster(
            TEST_SET, return_latent=True
        )
        external_metrics = cal_external_cluster_validation_metrics(
            clustering, DATA["test_y"]
        )
        internal_metrics = cal_internal_cluster_validation_metrics(
            latent_collector["clustering_latent"], DATA["test_y"]
        )
        logger.info(f"CRLI-LSTM: {external_metrics}")
        logger.info(f"CRLI-LSTM: {internal_metrics}")

    @pytest.mark.xdist_group(name="clustering-crli")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.crli_gru)

        # save the trained model into file, and check if the path exists
        self.crli_gru.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )

        # test loading the saved model, not necessary, but need to test
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.crli_gru.load_model(saved_model_path)


if __name__ == "__main__":
    unittest.main()
