"""
Test cases for imputation models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3


import unittest

import numpy as np

from pypots.data import generate_random_walk
from pypots.imputation import (
    SAITS,
    Transformer,
    BRITS,
    LOCF,
)


class TestSAITS(unittest.TestCase):
    def setUp(self) -> None:
        X = generate_random_walk()  # generate time-series data
        X[X < 0] = np.nan  # create missing values
        self.X = X
        self.saits = SAITS(n_layers=2, d_model=256, d_inner=128, n_head=4,
                           d_k=64, d_v=64, dropout=0.1, epochs=1)
        self.saits.fit(self.X)

    def test_parameters(self):
        assert (hasattr(self.saits, 'model')
                and self.saits.model is not None)

        assert (hasattr(self.saits, 'optimizer')
                and self.saits.optimizer is not None)

        assert hasattr(self.saits, 'best_loss')
        self.assertNotEqual(self.saits.best_loss, float('inf'))

        assert (hasattr(self.saits, 'best_model_dict')
                and self.saits.best_model_dict is not None)

    def test_impute(self):
        imputed_X = self.saits.impute(self.X)
        assert not np.isnan(imputed_X).any(), 'Output still has missing values after running impute().'


class TestTransformer(unittest.TestCase):
    def setUp(self) -> None:
        X = generate_random_walk()  # generate time-series data
        X[X < 0] = np.nan  # create missing values
        self.X = X
        self.transformer = Transformer(n_layers=2, d_model=256, d_inner=128, n_head=4,
                                       d_k=64, d_v=64, dropout=0.1, epochs=1)
        self.transformer.fit(self.X)

    def test_parameters(self):
        assert (hasattr(self.transformer, 'model')
                and self.transformer.model is not None)

        assert (hasattr(self.transformer, 'optimizer')
                and self.transformer.optimizer is not None)

        assert hasattr(self.transformer, 'best_loss')
        self.assertNotEqual(self.transformer.best_loss, float('inf'))

        assert (hasattr(self.transformer, 'best_model_dict')
                and self.transformer.best_model_dict is not None)

    def test_impute(self):
        imputed_X = self.transformer.impute(self.X)
        assert not np.isnan(imputed_X).any(), 'Output still has missing values after running impute().'


class TestBRITS(unittest.TestCase):
    def setUp(self) -> None:
        X = generate_random_walk()  # generate time-series data
        X[X < 0] = np.nan  # create missing values
        self.X = X
        self.brits = BRITS(256, epochs=1)
        self.brits.fit(self.X)

    def test_parameters(self):
        assert (hasattr(self.brits, 'model')
                and self.brits.model is not None)

        assert (hasattr(self.brits, 'optimizer')
                and self.brits.optimizer is not None)

        assert hasattr(self.brits, 'best_loss')
        self.assertNotEqual(self.brits.best_loss, float('inf'))

        assert (hasattr(self.brits, 'best_model_dict')
                and self.brits.best_model_dict is not None)

    def test_impute(self):
        imputed_X = self.brits.impute(self.X)
        assert not np.isnan(imputed_X).any(), 'Output still has missing values after running impute().'


class TestLOCF(unittest.TestCase):
    def setUp(self) -> None:
        X = generate_random_walk()  # generate time-series data
        X[X < 0] = np.nan  # create missing values
        self.X = X
        self.locf = LOCF(nan=0)

    def test_parameters(self):
        assert (hasattr(self.locf, 'nan')
                and self.locf.nan is not None)

    def test_impute(self):
        imputed_X = self.locf.impute(self.X)
        assert not np.isnan(imputed_X).any(), 'Output still has missing values after running impute().'


if __name__ == '__main__':
    unittest.main()
