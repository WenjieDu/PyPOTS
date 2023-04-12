"""
Test cases of logging.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

import os
import shutil
import unittest

from pypots.utils.logging import Logger


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


class TestPyPOTSCLI(unittest.TestCase):
    def test_pypots_cli(self):
        # TODO: need more test cases here
        os.system("python pypots/utils/commands/pypots_cli.py")


if __name__ == "__main__":
    unittest.main()
