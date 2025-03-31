"""
The package of the partially-observed time-series imputation model TimeMixer++.

Refer to the paper
`Shiyu Wang, Jiawei Li, Xiaoming Shi, Zhou Ye, Baichuan Mo, Wenze Lin, Ju Shengtong, Zhixuan Chu, Ming Jin.
"TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis".
ICLR 2025.
<https://openreview.net/pdf?id=1CLzLXSFNn>`_

Notes
-----
This implementation is inspired by the official one https://anonymous.4open.science/r/TimeMixerPP

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import TimeMixerPP

__all__ = [
    "TimeMixerPP",
]
