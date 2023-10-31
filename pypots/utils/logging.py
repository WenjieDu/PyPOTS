"""
Configure logging here.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

from tsdb.utils.logging import Logger

# initialize a logger for PyPOTS logging
logger_creator = Logger(name="PyPOTS running log")
logger = logger_creator.logger
