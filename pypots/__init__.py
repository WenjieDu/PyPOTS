"""
PyPOTS package.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3


# PyPOTS version
#
# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
# Generic release markers:
# X.Y
# X.Y.Z # For bugfix releases
#
# Admissible pre-release markers:
# X.YaN # Alpha release
# X.YbN # Beta release
# X.YrcN # Release Candidate
# X.Y # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
__version__ = "0.1.1"


__all__ = [
    "imputation",
    "classification",
    "clustering",
    "forecasting",
    "optim",
    "data",
    "utils",
    "__version__",
]
