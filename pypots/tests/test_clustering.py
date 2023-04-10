"""
Test cases for clustering models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


import unittest

import numpy as np
import pytest

from pypots.clustering import VaDER, CRLI
from pypots.tests.global_test_config import DATA
from pypots.utils.logging import logger
from pypots.utils.metrics import cal_rand_index, cal_cluster_purity

EPOCHS = 5

TRAIN_SET = {"X": DATA["train_X"]}
VAL_SET = {"X": DATA["val_X"]}
TEST_SET = {"X": DATA["test_X"]}


class TestCRLI(unittest.TestCase):
    logger.info("Running tests for a clustering model CRLI...")

    # initialize a CRLI model
    crli = CRLI(
        n_steps=DATA["n_steps"],
        n_features=DATA["n_features"],
        n_clusters=DATA["n_classes"],
        n_generator_layers=2,
        rnn_hidden_size=128,
        epochs=EPOCHS,
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


class TestVaDER(unittest.TestCase):
    logger.info("Running tests for a clustering model Transformer...")

    # initialize a VaDER model
    vader = VaDER(
        n_steps=DATA["n_steps"],
        n_features=DATA["n_features"],
        n_clusters=DATA["n_classes"],
        rnn_hidden_size=64,
        d_mu_stddev=5,
        pretrain_epochs=20,
        epochs=EPOCHS,
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


if __name__ == "__main__":
    unittest.main()
