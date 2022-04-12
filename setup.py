from setuptools import setup, find_packages

from pypots.version import __version__

with open('./README.md', encoding='utf-8') as f:
    README = f.read()

setup(
    name='pypots',
    version=__version__,
    description='A Python Toolbox for Data Mining on Partially-Observed Time Series',
    long_description=README,
    long_description_content_type='text/markdown',
    license='GPL-v3',
    author='Wenjie Du',
    author_email='wenjay.du@gmail.com',
    url='https://github.com/WenjieDu/PyPOTS',
    download_url='https://github.com/WenjieDu/PyPOTS/archive/master.zip',
    keywords=[
        'data mining', 'neural networks', 'machine learning', 'deep learning',
        'partially observed', 'time series', 'missing data', 'missing values',
    ],
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=[
        'matplotlib',
        'numpy',
        'scikit_learn',
        'scipy',
        'torch',
        'pandas',
    ],
    setup_requires=['setuptools>=38.6.0'],
)
