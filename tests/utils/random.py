"""
Test cases for the functions and classes in package `pypots.utils.random`.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import unittest

import torch

from pypots.utils.random import set_random_seed


class TestRandom(unittest.TestCase):
    def test_set_random_seed(self):
        random_state1 = torch.get_rng_state()
        torch.rand(
            1, 3
        )  # randomly generate something, the random state will be reset, so two states should be varying
        random_state2 = torch.get_rng_state()
        assert not torch.equal(
            random_state1, random_state2
        ), "The random seed hasn't set, so two random states should be different."

        set_random_seed(26)
        random_state1 = torch.get_rng_state()
        set_random_seed(26)
        random_state2 = torch.get_rng_state()
        assert torch.equal(
            random_state1, random_state2
        ), "The random seed has been set, two random states are not the same."


if __name__ == "__main__":
    unittest.main()
