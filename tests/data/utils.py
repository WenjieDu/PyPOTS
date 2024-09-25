"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import unittest

import numpy as np
import pytest

from pypots.data import (
    sliding_window,
    inverse_sliding_window,
)
from pypots.utils.logging import logger


class TestDataSavingAndLoading(unittest.TestCase):
    logger.info("Running tests for data utils...")

    @pytest.mark.xdist_group(name="data-utils")
    def test_0_sliding_window(self):
        data = np.random.randn(102, 5)
        window_size = 10
        stride = 10
        samples = sliding_window(data, window_size, stride)
        assert len(samples) == 10
        restored_data = inverse_sliding_window(samples, stride)
        assert len(restored_data) == 100
