"""
Test cases for VaDER clustering model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import unittest

import numpy as np
import pytest
import torch

from pypots.clustering import VaDER
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.nn.functional import (
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
    GENERAL_H5_TRAIN_SET_PATH,
    GENERAL_H5_VAL_SET_PATH,
    GENERAL_H5_TEST_SET_PATH,
    RESULT_SAVING_DIR_FOR_CLUSTERING,
    check_tb_and_model_checkpoints_existence,
)


class _GaussianCore(torch.nn.Module):
    def forward(self, inputs):
        X = inputs["X"]
        mu_tilde = torch.stack((X[:, 0, 0], torch.zeros_like(X[:, 0, 0])), dim=1)
        return {
            "mu_tilde": mu_tilde,
            "stddev_tilde": torch.ones_like(mu_tilde),
            "mu": X.new_tensor([[-1.0, 0.0], [1.0, 0.0]]),
            "var": X.new_ones((2, 2)),
            "phi": X.new_full((2,), 0.5),
            "z": mu_tilde,
            "imputation_latent": X,
        }


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
        rnn_hidden_size=32,
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
            logger.error(f"❌ Exception: {e}\nGot singular matrix, please try to retrain the model to fix this")

    @pytest.mark.xdist_group(name="clustering-vader")
    def test_2_parameters(self):
        assert hasattr(self.vader, "model") and self.vader.model is not None

        assert hasattr(self.vader, "optimizer") and self.vader.optimizer is not None

        assert hasattr(self.vader, "best_loss")
        self.assertNotEqual(self.vader.best_loss, float("inf"))

        assert hasattr(self.vader, "best_model_dict") and self.vader.best_model_dict is not None

    @pytest.mark.xdist_group(name="clustering-vader")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.vader)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.vader.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.vader.load(saved_model_path)

    @pytest.mark.xdist_group(name="clustering-vader")
    def test_4_lazy_loading(self):
        self.vader.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        clustering_results = self.vader.predict(GENERAL_H5_TEST_SET_PATH, return_latent_vars=True)
        external_metrics = calc_external_cluster_validation_metrics(clustering_results["clustering"], DATA["test_y"])
        internal_metrics = calc_internal_cluster_validation_metrics(
            clustering_results["latent_vars"]["z"], DATA["test_y"]
        )
        logger.info(f"Lazy-loading VaDER external_metrics: {external_metrics}")
        logger.info(f"Lazy-loading VaDER internal_metrics: {internal_metrics}")


@pytest.mark.xdist_group(name="clustering-vader")
@pytest.mark.parametrize("n_samples,batch_size", [(1, 8), (8, 8), (9, 8), (17, 8), (9, 1)])
@pytest.mark.parametrize("return_latent_vars", [False, True])
def test_vader_predict_singleton_batches(n_samples, batch_size, return_latent_vars, monkeypatch):
    def forbid_fit(*args, **kwargs):
        pytest.fail("Prediction must not train the model")

    monkeypatch.setattr(VaDER, "fit", forbid_fit)
    model = VaDER(
        n_steps=4,
        n_features=2,
        n_clusters=2,
        rnn_hidden_size=4,
        d_mu_stddev=2,
        batch_size=batch_size,
        epochs=1,
        pretrain_epochs=1,
        device="cpu",
        verbose=False,
    )
    model.model = _GaussianCore()
    X = np.zeros((n_samples, 4, 2), dtype=np.float32)
    expected = np.arange(n_samples) % 2
    X[:, :, 0] = np.where(expected[:, None] == 0, -1.0, 1.0)

    result = model.predict({"X": X}, return_latent_vars=return_latent_vars)

    assert result["clustering"].shape == (n_samples,)
    np.testing.assert_array_equal(result["clustering"], expected)
    if return_latent_vars:
        for key in ("mu_tilde", "stddev_tilde", "z", "imputation_latent"):
            assert result["latent_vars"][key].shape[0] == n_samples
    else:
        assert set(result) == {"clustering"}
        np.testing.assert_array_equal(model.cluster({"X": X}), expected)


if __name__ == "__main__":
    unittest.main()
