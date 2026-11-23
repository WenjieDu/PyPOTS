"""
Test cases for TSICL imputation model.
"""

# Created by Etienne Le Naour <etienne.le-naour@edf.fr>, Tahar Nabil <tahar.nabil@edf.fr>,
# and Adrien Petralia <adrien.petralia@gmail.com>
# License: BSD-3-Clause

import os.path
import unittest

import numpy as np
import pytest

from pypots.imputation import TSICL
from pypots.nn.functional import calc_mse
from pypots.utils.logging import logger
from tests.global_test_config import (
    DATA,
    DEVICE,
    GENERAL_H5_TEST_SET_PATH,
    RESULT_SAVING_DIR_FOR_IMPUTATION,
    TEST_SET,
    check_tb_and_model_checkpoints_existence,
)


class TestTSICL(unittest.TestCase):
    logger.info("Running tests for an imputation model TSICL...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "TSICL")
    model_save_name = "saved_tsicl_model.pypots"

    tsicl = TSICL(device=DEVICE, saving_path=saving_path)

    @pytest.mark.xdist_group(name="imputation-tsicl")
    def test_0_fit(self):
        with pytest.warns(UserWarning):
            self.tsicl.fit(TEST_SET)

    @pytest.mark.xdist_group(name="imputation-tsicl")
    def test_1_impute(self):
        imputation_results = self.tsicl.predict(TEST_SET)
        assert not np.isnan(imputation_results["imputation"]).any(), (
            "Output still has missing values after running impute()."
        )

        # observed values must be carried through unchanged (up to TS-ICL's float32 precision)
        observed = ~np.isnan(TEST_SET["X"])
        assert np.array_equal(
            imputation_results["imputation"][observed],
            TEST_SET["X"][observed].astype(np.float32),
        )

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"TSICL test_MSE: {test_MSE}")

    @pytest.mark.xdist_group(name="imputation-tsicl")
    def test_2_fully_missing_series(self):
        # a fully-missing series (even alone in its batch) gives the model no context, it must fall back
        # to 0 instead of crashing, without affecting the other series
        X = TEST_SET["X"][:2].copy()
        X[0, :, 0] = np.nan
        tsicl = TSICL(device=DEVICE, batch_size=1)
        imputed = tsicl.predict({"X": X})["imputation"]
        assert not np.isnan(imputed).any()
        assert (imputed[0, :, 0] == 0).all()

    @pytest.mark.xdist_group(name="imputation-tsicl")
    def test_3_point_estimator_fallback(self):
        # `median` needs the 0.5 quantile, TS-ICL warns and falls back to `mean` without it
        tsicl = TSICL(device=DEVICE, point_estimator="median", quantile_levels=[0.1, 0.9])
        with pytest.warns(UserWarning, match="switch to `mean` estimator"):
            imputed = tsicl.predict({"X": TEST_SET["X"][:2]})["imputation"]
        assert not np.isnan(imputed).any()

    @pytest.mark.xdist_group(name="imputation-tsicl")
    def test_4_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.tsicl)

        # save the model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.tsicl.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.tsicl.load(saved_model_path)

    @pytest.mark.xdist_group(name="imputation-tsicl")
    def test_5_lazy_loading(self):
        imputation_results = self.tsicl.predict(GENERAL_H5_TEST_SET_PATH)
        assert not np.isnan(imputation_results["imputation"]).any(), (
            "Output still has missing values after running impute()."
        )

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"Lazy-loading TSICL test_MSE: {test_MSE}")


if __name__ == "__main__":
    unittest.main()
