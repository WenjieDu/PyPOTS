"""
Functions to load supported open-source time-series datasets.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import tsdb

from .load_preprocessing import preprocess_physionet2012
from ..utils.logging import logger

# currently supported datasets
SUPPORTED_DATASETS = [
    "physionet_2012",
]

# preprocessing functions of the supported datasets
PREPROCESSING_FUNC = {
    "physionet_2012": preprocess_physionet2012,
}


def list_supported_datasets() -> list:
    """Return the datasets natively supported by PyPOTS so far.

    Returns
    -------
    SUPPORTED_DATASETS :
        A list including all supported datasets.

    """
    return SUPPORTED_DATASETS


def load_specific_dataset(dataset_name: str, use_cache: bool = True) -> dict:
    """Load specific datasets supported by PyPOTS.
    Different from tsdb.load_dataset(), which only produces merely raw data,
    load_specific_dataset here does some preprocessing operations,
    like truncating time series to generate samples with the same length.

    Parameters
    ----------
    dataset_name :
        The name of the dataset to be loaded, which should be supported, i.e. in SUPPORTED_DATASETS.

    use_cache :
        Whether to use cache. This is an argument of tsdb.load_dataset().

    Returns
    -------
    data :
        A dict contains the preprocessed dataset.
        Users only need to continue the preprocessing steps to generate the data they want,
        e.g. standardizing and splitting.

    """
    logger.info(
        f"Loading the dataset {dataset_name} with TSDB (https://github.com/WenjieDu/Time_Series_Data_Beans)..."
    )
    assert dataset_name in SUPPORTED_DATASETS, (
        f"Dataset {dataset_name} is not supported. "
        f"If you believe this dataset is valuable to be supported by PyPOTS,"
        f"please create an issue on GitHub "
        f"https://github.com/WenjieDu/PyPOTS/issues"
    )
    logger.info(f"Starting preprocessing {dataset_name}...")
    data = tsdb.load(dataset_name, use_cache)
    data = PREPROCESSING_FUNC[dataset_name](data)
    return data
