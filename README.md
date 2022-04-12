<a href='https://github.com/WenjieDu/PyPOTS'><img src='https://raw.githubusercontent.com/WenjieDu/PyPOTS/main/docs/figs/PyPOTS%20logo.svg?sanitize=true' width='190' align='right' /></a>

# <p align='center'> Welcome to PyPOTS </p>
### <p align='center'> A Python Toolbox for Data Mining on Partially-Observed Time Series </p>

<p align='center'>
    <!-- Python version -->
    <img src='https://img.shields.io/badge/python-v3-green'>
    <!-- License -->
    <img src='https://img.shields.io/badge/License-GPL--v3-brightgreen'>
    <!-- PyPI download number -->
    <a alt='PyPI download number' href='https://pypi.org/project/pypots'>
        <img src='https://static.pepy.tech/personalized-badge/pypots?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads'>
    </a>
    <!-- Hits number -->
    <img src='https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FPyPOTS%2FPyPOTS&count_bg=%23009A0A&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Hits&edge_flat=false'>
    <!-- Chat in Discussions -->
    <a alt='GitHub Discussions' href='https://github.com/WenjieDu/PyPOTS/discussions'>
        <img src='https://img.shields.io/badge/Chat-in_Discussions-green?logo=github&color=60A98D'>
    </a>
</p>

⦿ `Motivation`: Due to all kinds of reasons like failure of collection sensors, communication error, and unexpected malfunction, missing values are common to see in time series from the real-world environment. This makes partially-observed time series (POTS) a pervasive problem in open-world modeling and prevents advanced data analysis. Although this problem is important, the area of data mining on POTS still lacks a dedicated toolkit. PyPOTS is created to fulfill this blank, to become a handy toolbox that is going to make data mining on POTS easy rather than tedious, to help engineers and researchers focus more on the core problems in their hands rather than on how to deal with the missing parts in their data.

⦿ `Mission`: PyPOTS will keep integrating classical and the latest state-of-the-art data mining algorithms for partially-observed multivariate time series. For sure, besides various algorithms, PyPOTS is going to have unified APIs together with detailed documentation and interactive examples across algorithms as tutorials.

## ❖ Installation
Install the latest release from PyPI: 
> pip install pypots

Install with the latest code on GitHub: 
> pip install `https://github.com/WenjieDu/PyPOTS/archive/master.zip`

## ❖ Available Algorithms
| Task Type  | Model Type | Algorithm    | Year    | Reference |        
|------------|------------|--------------|---------|-----------|
| Imputation | Neural Network | SAITS: Self-Attention-based Imputation for Time Series | 2022 | [^1] |
| Imputation | Neural Network | Transformer | 2017 | [^2] [^1] |
| Imputation,<br>Classification | Neural Network | BRITS: Bidirectional Recurrent Imputation for Time Series | 2018 | [^3] |

---
‼️ PyPOTS is currently under development. If you like it and look forward to its growth, <ins>please give PyPOTS a star and watch it to keep you posted on its progress and to let me know that its development is meaningful</ins>. If you have any feedback, or want to contribute ideas/suggestions or share time-series related algorithms/papers, please join PyPOTS community and <a alt='GitHub Discussions' href='https://github.com/WenjieDu/PyPOTS/discussions'><img align='center' src='https://img.shields.io/badge/Chat-in_Discussions-green?logo=github&color=60A98D'></a>, or [drop me an email](mailto:wenjay.du@gmail.com).

Thank you all for your attention! 😃

[^1]: Du, W., Cote, D., & Liu, Y. (2022). SAITS: Self-Attention-based Imputation for Time Series. ArXiv, abs/2202.08516.
[^2]: Vaswani, A., Shazeer, N.M., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., & Polosukhin, I. (2017). Attention is All you Need. NeurIPS 2017.
[^3]: Cao, W., Wang, D., Li, J., Zhou, H., Li, L., & Li, Y. (2018). BRITS: Bidirectional Recurrent Imputation for Time Series. NeurIPS 2018.