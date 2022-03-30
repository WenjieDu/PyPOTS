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
    license='MIT',
    author='Wenjie Du',
    author_email='wenjay.du@gmail.com',
    url='https://github.com/pypots/pypots',
    download_url='https://github.com/pypots/pypots/archive/master.zip',
    keywords=[
        'data mining', 'neural networks', 'machine learning', 'deep learning',
        'partially observed', 'time series', 'missing data', 'missing values',
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'matplotlib',
        'numpy',
        'scikit_learn',
        'scipy',
        'pytorch',
        'pandas',
    ],
    setup_requires=['setuptools>=38.6.0'],
)
