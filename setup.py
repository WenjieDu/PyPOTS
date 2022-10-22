from setuptools import setup, find_packages

from pypots.__version__ import version

with open("./README.md", encoding="utf-8") as f:
    README = f.read()

setup(
    name="pypots",
    version=version,
    description="A Python Toolbox for Data Mining on Partially-Observed Time Series",
    long_description=README,
    long_description_content_type="text/markdown",
    license="GPL-3.0",
    author="Wenjie Du",
    author_email="wenjay.du@gmail.com",
    url="https://github.com/WenjieDu/PyPOTS",
    download_url="https://github.com/WenjieDu/PyPOTS/archive/master.zip",
    keywords=[
        "data mining",
        "neural networks",
        "machine learning",
        "deep learning",
        "partially observed",
        "time series",
        "missing data",
        "missing values",
    ],
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    install_requires=[
        "matplotlib",
        "numpy",
        "scikit_learn",
        "scipy",
        "torch>=1.10",  # torch_sparse v0.6.12 requires 1.9<=torch<1.10, v0.6.13 needs torch>=1.10
        "torch_sparse==0.6.13",
        "torch_scatter",
        "torch_geometric",
        "tensorboard",
        "pandas",
        "pycorruptor",
        "tsdb",
    ],
    python_requires=">=3.7.0",
    setup_requires=["setuptools>=38.6.0"],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
