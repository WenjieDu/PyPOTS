"""
Test cases for metric functions
"""

# Created by Wenjie Du <wdu@time-series.ai>
# License: BSD-3-Clause


import unittest

import pytest
import torch

from pypots.nn.functional import calc_quantile_crps


class TestMetrics(unittest.TestCase):
    @pytest.mark.xdist_group(name="metric-quantile-crps")
    def test_quantile_crps(self):
        if torch.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        P = torch.randn(2, 6, 3, 5, device=device)
        T = torch.randn(2, 3, 5, device=device)
        M = torch.ones(2, 3, 5, device=device)
        calc_quantile_crps(P, T, M)


if __name__ == "__main__":
    unittest.main()
