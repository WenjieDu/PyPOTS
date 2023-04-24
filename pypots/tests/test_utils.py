"""
Test cases for the functions and classes in package `pypots.utils`.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

import os
import shutil
import unittest

import torch

from pypots.utils.logging import Logger
from pypots.utils.random import set_random_seed


class TestLogging(unittest.TestCase):
    logger_creator = Logger(name="PyPOTS testing log", logging_level="debug")
    logger = logger_creator.logger

    def test_different_level_logging(self):
        self.logger.debug("debug")
        self.logger.info("info")
        self.logger.warning("warning")
        self.logger.error("error")

    def test_changing_level(self):
        self.logger_creator.set_level("info")
        assert (
            self.logger.level == 20
        ), f"the level of logger is {self.logger.level}, not INFO"
        self.logger_creator.set_level("warning")
        assert (
            self.logger.level == 30
        ), f"the level of logger is {self.logger.level}, not WARNING"
        self.logger_creator.set_level("error")
        assert (
            self.logger.level == 40
        ), f"the level of logger is {self.logger.level}, not ERROR"
        self.logger_creator.set_level("debug")
        assert (
            self.logger.level == 10
        ), f"the level of logger is {self.logger.level}, not DEBUG"

    def test_saving_log_into_file(self):
        self.logger_creator.set_saving_path("test_log", "testing.log")
        assert os.path.exists("test_log/testing.log")
        shutil.rmtree("test_log", ignore_errors=True)


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
