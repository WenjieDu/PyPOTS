"""
Test cases for VaDER clustering model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os
import unittest

import numpy as np
import pytest

from pypots.clustering import VaDER
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
        self.vader.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="clustering-vader")
    def test_1_cluster(self):
        try:
            clustering_results = self.vader.predict(TEST_SET, return_latent_vars=True)
            external_metrics = calc_external_cluster_validation_metrics(
                clustering_results["clustering"], DATA["test_y"]
            )
            internal_metrics = calc_internal_cluster_validation_metrics(
                clustering_results["latent_vars"]["z"], DATA["test_y"]
            )
            logger.info(f"VaDER external_metrics: {external_metrics}")
            logger.info(f"VaDER internal_metrics: {internal_metrics}")
        except np.linalg.LinAlgError as e:
            logger.error(
                f"❌ Exception: {e}\n"
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
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.vader.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.vader.load(saved_model_path)

    @pytest.mark.xdist_group(name="clustering-vader")
    def test_4_lazy_loading(self):
        self.vader.fit(H5_TRAIN_SET_PATH, H5_VAL_SET_PATH)
        clustering_results = self.vader.predict(
            H5_TEST_SET_PATH, return_latent_vars=True
        )
        external_metrics = calc_external_cluster_validation_metrics(
            clustering_results["clustering"], DATA["test_y"]
        )
        internal_metrics = calc_internal_cluster_validation_metrics(
            clustering_results["latent_vars"]["z"], DATA["test_y"]
        )
        logger.info(f"Lazy-loading VaDER external_metrics: {external_metrics}")
        logger.info(f"Lazy-loading VaDER internal_metrics: {internal_metrics}")


if __name__ == "__main__":
    unittest.main()
