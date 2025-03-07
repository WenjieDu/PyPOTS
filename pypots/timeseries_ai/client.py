"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from ai4ts.client import TimeSeriesAI as TimeSeriesAI_Client

from ..utils.logging import logger


class TimeSeriesAI(TimeSeriesAI_Client):
    def __init__(self):
        logger.info(
            "AI for real-world time series is coming! 🚀\n"
            "Join our waitlist and stay tuned please! https://time-series.ai"
        )
