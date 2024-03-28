"""
Test cases for the learning rate schedulers.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import unittest

import numpy as np
import pytest

from pypots.imputation import SAITS
from pypots.optim import Adam, AdamW, Adadelta, Adagrad, RMSprop, SGD
from pypots.optim.lr_scheduler import (
    LambdaLR,
    ConstantLR,
    ExponentialLR,
    LinearLR,
    StepLR,
    MultiStepLR,
    MultiplicativeLR,
)
from pypots.utils.logging import logger
from pypots.utils.metrics import calc_mae
from tests.global_test_config import DATA
from tests.optim.config import EPOCHS, TEST_SET, TRAIN_SET, VAL_SET


class TestLRSchedulers(unittest.TestCase):
    logger.info("Running tests for learning rate schedulers...")

    # init lambda_lrs
    lambda_lrs = LambdaLR(lr_lambda=lambda epoch: epoch // 30, verbose=True)

    # init multiplicative_lrs
    multiplicative_lrs = MultiplicativeLR(lr_lambda=lambda epoch: 0.95, verbose=True)

    # init step_lrs
    step_lrs = StepLR(step_size=30, gamma=0.1, verbose=True)

    # init multistep_lrs
    multistep_lrs = MultiStepLR(milestones=[30, 80], gamma=0.1, verbose=True)

    # init constant_lrs
    constant_lrs = ConstantLR(factor=0.5, total_iters=4, verbose=True)

    # init linear_lrs
    linear_lrs = LinearLR(start_factor=0.5, total_iters=4, verbose=True)

    # init exponential_lrs
    exponential_lrs = ExponentialLR(gamma=0.9, verbose=True)

    @pytest.mark.xdist_group(name="lrs-lambda")
    def test_0_lambda_lrs(self):
        logger.info("Running tests for Adam + LambdaLRS...")

        adam = Adam(lr=0.001, weight_decay=1e-5, lr_scheduler=self.lambda_lrs)
        saits = SAITS(
            DATA["n_steps"],
            DATA["n_features"],
            n_layers=1,
            d_model=128,
            d_ffn=64,
            n_heads=2,
            d_k=64,
            d_v=64,
            dropout=0.1,
            optimizer=adam,
            epochs=EPOCHS,
        )
        saits.fit(TRAIN_SET, VAL_SET)
        imputed_X = saits.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = calc_mae(
            imputed_X, DATA["test_X_ori"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"SAITS test_MAE: {test_MAE}")

    @pytest.mark.xdist_group(name="lrs-multiplicative")
    def test_1_multiplicative_lrs(self):
        logger.info("Running tests for Adamw + MultiplicativeLRS...")

        adamw = AdamW(lr=0.001, weight_decay=1e-5, lr_scheduler=self.multiplicative_lrs)
        saits = SAITS(
            DATA["n_steps"],
            DATA["n_features"],
            n_layers=1,
            d_model=128,
            d_ffn=64,
            n_heads=2,
            d_k=64,
            d_v=64,
            dropout=0.1,
            optimizer=adamw,
            epochs=EPOCHS,
        )
        saits.fit(TRAIN_SET, VAL_SET)
        imputed_X = saits.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = calc_mae(
            imputed_X, DATA["test_X_ori"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"SAITS test_MAE: {test_MAE}")

    @pytest.mark.xdist_group(name="lrs-step")
    def test_2_step_lrs(self):
        logger.info("Running tests for Adadelta + StepLRS...")

        adamw = Adadelta(lr=0.001, lr_scheduler=self.step_lrs)
        saits = SAITS(
            DATA["n_steps"],
            DATA["n_features"],
            n_layers=1,
            d_model=128,
            d_ffn=64,
            n_heads=2,
            d_k=64,
            d_v=64,
            dropout=0.1,
            optimizer=adamw,
            epochs=EPOCHS,
        )
        saits.fit(TRAIN_SET, VAL_SET)
        imputed_X = saits.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = calc_mae(
            imputed_X, DATA["test_X_ori"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"SAITS test_MAE: {test_MAE}")

    @pytest.mark.xdist_group(name="lrs-multistep")
    def test_3_multistep_lrs(self):
        logger.info("Running tests for Adadelta + MultiStepLRS...")

        adagrad = Adagrad(lr=0.001, lr_scheduler=self.multistep_lrs)
        saits = SAITS(
            DATA["n_steps"],
            DATA["n_features"],
            n_layers=1,
            d_model=128,
            d_ffn=64,
            n_heads=2,
            d_k=64,
            d_v=64,
            dropout=0.1,
            optimizer=adagrad,
            epochs=EPOCHS,
        )
        saits.fit(TRAIN_SET, VAL_SET)
        imputed_X = saits.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = calc_mae(
            imputed_X, DATA["test_X_ori"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"SAITS test_MAE: {test_MAE}")

    @pytest.mark.xdist_group(name="lrs-constant")
    def test_4_constant_lrs(self):
        logger.info("Running tests for RMSprop + ConstantLRS...")

        # initialize a SAITS model for testing DatasetForMIT and BaseDataset
        rmsprop = RMSprop(lr=0.001, lr_scheduler=self.constant_lrs)
        saits = SAITS(
            DATA["n_steps"],
            DATA["n_features"],
            n_layers=1,
            d_model=128,
            d_ffn=64,
            n_heads=2,
            d_k=64,
            d_v=64,
            dropout=0.1,
            optimizer=rmsprop,
            epochs=EPOCHS,
        )
        saits.fit(TRAIN_SET, VAL_SET)
        imputed_X = saits.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = calc_mae(
            imputed_X, DATA["test_X_ori"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"SAITS test_MAE: {test_MAE}")

    @pytest.mark.xdist_group(name="lrs-linear")
    def test_5_linear_lrs(self):
        logger.info("Running tests for SGD + MultiStepLRS...")

        sgd = SGD(lr=0.001, lr_scheduler=self.linear_lrs)
        saits = SAITS(
            DATA["n_steps"],
            DATA["n_features"],
            n_layers=1,
            d_model=128,
            d_ffn=64,
            n_heads=2,
            d_k=64,
            d_v=64,
            dropout=0.1,
            optimizer=sgd,
            epochs=EPOCHS,
        )
        saits.fit(TRAIN_SET, VAL_SET)
        imputed_X = saits.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = calc_mae(
            imputed_X, DATA["test_X_ori"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"SAITS test_MAE: {test_MAE}")

    @pytest.mark.xdist_group(name="lrs-exponential")
    def test_6_exponential_lrs(self):
        logger.info("Running tests for SGD + ExponentialLRS...")

        sgd = SGD(lr=0.001, lr_scheduler=self.exponential_lrs)
        saits = SAITS(
            DATA["n_steps"],
            DATA["n_features"],
            n_layers=1,
            d_model=128,
            d_ffn=64,
            n_heads=2,
            d_k=64,
            d_v=64,
            dropout=0.1,
            optimizer=sgd,
            epochs=EPOCHS,
        )
        saits.fit(TRAIN_SET, VAL_SET)
        imputed_X = saits.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = calc_mae(
            imputed_X, DATA["test_X_ori"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"SAITS test_MAE: {test_MAE}")
