"""
Test cases for the functions and classes in package `pypots.utils.visual`.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import unittest

import numpy as np

from pypots.imputation import LOCF
from pypots.utils.visual.data import plot_data, plot_missingness
from tests.global_test_config import TEST_SET


class TestVisual(unittest.TestCase):
    locf = LOCF()
    imputed_test_set = locf.predict(TEST_SET)
    imputed_X = imputed_test_set["imputation"]
    X_with_missingness = TEST_SET["X"]
    X_ori = TEST_SET["X_ori"]

    def test_plot_data(self):
        plot_data(self.X_with_missingness, self.X_ori, self.imputed_X, sample_idx=10)

    def test_plot_missingness(self):
        plot_missingness(self.X_with_missingness, max_step=24, sample_idx=10)
        plot_missingness(
            ~np.isnan(self.X_with_missingness[10]), max_step=24, sample_idx=10
        )


if __name__ == "__main__":
    unittest.main()
