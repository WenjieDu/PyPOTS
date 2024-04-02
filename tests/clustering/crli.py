"""
Test cases for CRLI clustering model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os
import unittest

import pytest

from pypots.clustering import CRLI
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.utils.metrics import (
    calc_external_cluster_validation_metrics,
    calc_internal_cluster_validation_metrics,
)
from tests.global_test_config import (
    DATA,
    EPOCHS,
    DEVICE,
    TRAIN_SET,
    VAL_SET,
    TEST_SET,
    H5_TRAIN_SET_PATH,
    H5_VAL_SET_PATH,
    H5_TEST_SET_PATH,
    RESULT_SAVING_DIR_FOR_CLUSTERING,
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
        rnn_hidden_size=32,
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
        self.crli_gru.fit(TRAIN_SET, VAL_SET)
        logger.info("Training CRLI-LSTM...")
        self.crli_lstm.fit(TRAIN_SET, VAL_SET)

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
        clustering_results = self.crli_gru.predict(TEST_SET, return_latent_vars=True)
        external_metrics = calc_external_cluster_validation_metrics(
            clustering_results["clustering"], DATA["test_y"]
        )
        internal_metrics = calc_internal_cluster_validation_metrics(
            clustering_results["latent_vars"]["clustering_latent"], DATA["test_y"]
        )
        logger.info(f"CRLI-GRU external_metrics: {external_metrics}")
        logger.info(f"CRLI-GRU internal_metrics: {internal_metrics}")

        # LSTM cell
        clustering_results = self.crli_lstm.predict(TEST_SET, return_latent_vars=True)
        external_metrics = calc_external_cluster_validation_metrics(
            clustering_results["clustering"], DATA["test_y"]
        )
        internal_metrics = calc_internal_cluster_validation_metrics(
            clustering_results["latent_vars"]["clustering_latent"], DATA["test_y"]
        )
        logger.info(f"CRLI-LSTM external_metrics: {external_metrics}")
        logger.info(f"CRLI-LSTM internal_metrics: {internal_metrics}")

    @pytest.mark.xdist_group(name="clustering-crli")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.crli_gru)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.crli_gru.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.crli_gru.load(saved_model_path)

    @pytest.mark.xdist_group(name="clustering-crli")
    def test_4_lazy_loading(self):
        self.crli_gru.fit(H5_TRAIN_SET_PATH, H5_VAL_SET_PATH)
        clustering_results = self.crli_gru.predict(
            H5_TEST_SET_PATH, return_latent_vars=True
        )
        external_metrics = calc_external_cluster_validation_metrics(
            clustering_results["clustering"], DATA["test_y"]
        )
        internal_metrics = calc_internal_cluster_validation_metrics(
            clustering_results["latent_vars"]["clustering_latent"], DATA["test_y"]
        )
        logger.info(f"Lazy-loading CRLI-GRU external_metrics: {external_metrics}")
        logger.info(f"Lazy-loading CRLI-GRU internal_metrics: {internal_metrics}")


if __name__ == "__main__":
    unittest.main()
