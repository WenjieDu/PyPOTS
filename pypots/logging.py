"""
Configure logging here.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

import logging
import os


class Logger:
    def __init__(
            self,
            name="PyPOTS running log",
            logging_level=logging.DEBUG,
            logging_format="%(asctime)s [%(levelname)s]: %(message)s",
    ):
        self.logger = logging.getLogger(name)
        self.logging_level = logging_level
        self.formatter = logging.Formatter(logging_format, datefmt="%Y-%m-%d %H:%M:%S")

        self.stream_handler = logging.StreamHandler()
        self.file_handler = None

        self.set_level(logging_level)
        self.set_logging_format()
        self.logger.propagate = False

    def set_logging_format(self):
        self.stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.stream_handler)

    def set_saving_path(self, saving_dir, name, mode="a"):
        if saving_dir is not None:
            os.makedirs(saving_dir, exist_ok=True)
            path = os.path.join(saving_dir, name)
            self.file_handler = logging.FileHandler(path, mode=mode)
            self.file_handler.setLevel(self.logging_level)
            self.file_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.file_handler)
            self.logger.info(f'Log will be saved to {path}')
        else:
            self.file_handler = None

    def set_level(self, level):
        self.logging_level = level
        self.logger.setLevel(self.logging_level)
        if self.stream_handler is not None:
            self.stream_handler.setLevel(level)
        if self.file_handler is not None:
            self.file_handler.setLevel(level)
        self.logger.info(f'Successfully set logging level to {level}')


logger_creator = Logger()
logger = logger_creator.logger
