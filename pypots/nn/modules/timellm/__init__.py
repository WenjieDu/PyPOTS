"""
The package including the modules of Time-LLM.

Refer to the paper
`Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y. Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang,
Yuan-Fang Li, Shirui Pan, and Qingsong Wen.
Time-LLM: Time Series Forecasting by Reprogramming Large Language Models.
In the 12th International Conference on Learning Representations, 2024.
<https://openreview.net/pdf?id=Unb5CVPtae>`_

Notes
-----
This implementation is inspired by the official one https://github.com/KimMeen/Time-LLM

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .backbone import BackboneTimeLLM


__all__ = [
    "BackboneTimeLLM",
]
