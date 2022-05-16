"""
Test cases for forecasting models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import unittest

import numpy as np

from pypots.forecasting import BTTF
from pypots.tests.unified_data_for_test import gene_random_walk_data
from pypots.utils.metrics import cal_mae

EPOCHS = 5


class TestBTTF(unittest.TestCase):
    def setUp(self) -> None:
        DATA = gene_random_walk_data(n_steps=120, n_features=10)
        self.test_X = DATA['test_X']
        self.test_X_intact = DATA['test_X_intact']
        self.test_X_for_input = self.test_X[:, :100]
        print('Running test cases for BTTF...')
        self.bttf = BTTF(100, 10,
                         20, 2, 10,
                         np.asarray([1, 2, 3, 10, 10 + 1, 10 + 2, 20, 20 + 1, 20 + 2]),
                         5, 5)

    def test_forecasting(self):
        predictions = self.bttf.forecast(self.test_X_for_input)
        mae = cal_mae(predictions, self.test_X_intact[:, 100:])
        print(f'prediction MAE: {mae}')

    if __name__ == '__main__':
        unittest.main()
