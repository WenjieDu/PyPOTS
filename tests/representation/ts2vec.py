"""
Test cases for TS2Vec representation model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os.path
import unittest

import numpy as np
import pytest

from pypots.optim import Adam
from pypots.representation import TS2Vec
from pypots.utils.logging import logger
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
    RESULT_SAVING_DIR_FOR_REPRESENTATION,
    check_tb_and_model_checkpoints_existence,
)


class TestTS2Vec(unittest.TestCase):
    logger.info("Running tests for a representor model TS2Vec...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_REPRESENTATION, "TS2Vec")
    model_save_name = "saved_ts2vec_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    d_representation = 2

    # initialize a TS2Vec model
    ts2vec = TS2Vec(
        DATA["n_steps"],
        DATA["n_features"],
        n_output_dims=d_representation,
        d_hidden=64,
        n_layers=2,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="representor-ts2vec")
    def test_0_fit(self):
        self.ts2vec.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="representor-ts2vec")
    def test_1_represent(self):
        representor_results = self.ts2vec.predict(TEST_SET)
        assert len(representor_results["representation"].shape) == 3
        assert representor_results["representation"].shape[-1] == self.d_representation
        representation = self.ts2vec.represent(TEST_SET, encoding_window="full_series")
        assert len(representation.shape) == 2
        assert representation.shape[-1] == self.d_representation

    @pytest.mark.xdist_group(name="representor-ts2vec")
    def test_2_parameters(self):
        assert hasattr(self.ts2vec, "model") and self.ts2vec.model is not None

        assert hasattr(self.ts2vec, "optimizer") and self.ts2vec.optimizer is not None

        assert hasattr(self.ts2vec, "best_loss")
        self.assertNotEqual(self.ts2vec.best_loss, float("inf"))

        assert hasattr(self.ts2vec, "best_model_dict") and self.ts2vec.best_model_dict is not None

    @pytest.mark.xdist_group(name="representor-ts2vec")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.ts2vec)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.ts2vec.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.ts2vec.load(saved_model_path)

    @pytest.mark.xdist_group(name="representor-ts2vec")
    def test_4_lazy_loading(self):
        self.ts2vec.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        representor_results = self.ts2vec.predict(GENERAL_H5_TEST_SET_PATH)
        assert not np.isnan(
            representor_results["representation"]
        ).any(), "Output still has missing values after running impute()."


if __name__ == "__main__":
    unittest.main()
