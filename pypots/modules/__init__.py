"""
Everything used to be in this package has been moved to pypots.nn.modules.
This package is kept for backward compatibility and will be removed in the future.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from ..utils.logging import logger

logger.warning(
    "🚨 pypots.modules package has been moved to pypots.nn.modules. "
    "Please import everything from pypots.nn.modules instead."
)
