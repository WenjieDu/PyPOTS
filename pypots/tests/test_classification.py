"""
Test cases for 
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import unittest

import numpy as np

from pypots.classification import BRITS
from pypots.data import generate_random_walk_for_classification


class TestBRITS(unittest.TestCase):
    def setUp(self) -> None:
        # generate time-series classification data
        X, y = generate_random_walk_for_classification(n_classes=3, n_samples_each_class=10)
        X[X < 0] = np.nan  # create missing values
        self.X = X
        self.y = y
        self.brits = BRITS(256, n_classes=3, epochs=1)
        self.brits.fit(self.X, self.y)

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
        predictions = self.brits.classify(self.X)


if __name__ == '__main__':
    unittest.main()
