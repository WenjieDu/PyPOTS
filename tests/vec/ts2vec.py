"""
Test cases for TS2Vec vectorizer model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os.path
import unittest

import numpy as np
import pytest

from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.vec import TS2Vec
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
    RESULT_SAVING_DIR_FOR_IMPUTATION,
    check_tb_and_model_checkpoints_existence,
)


class TestTS2Vec(unittest.TestCase):
    logger.info("Running tests for an vectorizer model TS2Vec...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "TS2Vec")
    model_save_name = "saved_ts2vec_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    d_vectorization = 2

    # initialize a TS2Vec model
    ts2vec = TS2Vec(
        DATA["n_steps"],
        DATA["n_features"],
        n_output_dims=d_vectorization,
        d_hidden=64,
        n_layers=2,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="vectorizer-ts2vec")
    def test_0_fit(self):
        self.ts2vec.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="vectorizer-ts2vec")
    def test_1_vectorize(self):
        vectorizer_results = self.ts2vec.predict(TEST_SET)
        assert len(vectorizer_results["vectorization"].shape) == 3
        assert vectorizer_results["vectorization"].shape[-1] == self.d_vectorization
        vectorization = self.ts2vec.vectorize(TEST_SET, encoding_window="full_series")
        assert len(vectorization.shape) == 2
        assert vectorization.shape[-1] == self.d_vectorization

    @pytest.mark.xdist_group(name="vectorizer-ts2vec")
    def test_2_parameters(self):
        assert hasattr(self.ts2vec, "model") and self.ts2vec.model is not None

        assert hasattr(self.ts2vec, "optimizer") and self.ts2vec.optimizer is not None

        assert hasattr(self.ts2vec, "best_loss")
        self.assertNotEqual(self.ts2vec.best_loss, float("inf"))

        assert hasattr(self.ts2vec, "best_model_dict") and self.ts2vec.best_model_dict is not None

    @pytest.mark.xdist_group(name="vectorizer-ts2vec")
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

    @pytest.mark.xdist_group(name="vectorizer-ts2vec")
    def test_4_lazy_loading(self):
        self.ts2vec.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        vectorizer_results = self.ts2vec.predict(GENERAL_H5_TEST_SET_PATH)
        assert not np.isnan(
            vectorizer_results["vectorization"]
        ).any(), "Output still has missing values after running impute()."


if __name__ == "__main__":
    unittest.main()
