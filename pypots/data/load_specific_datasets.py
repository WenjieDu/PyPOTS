"""
Functions to load supported open-source time-series datasets.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import pandas as pd
import tsdb
from pypots.utils.logging import logger

SUPPORTED_DATASETS = [
    "physionet_2012",
]


def list_supported_datasets():
    """

    Returns
    -------
    SUPPORTED_DATASETS : list
        A list including all supported datasets.

    """
    return SUPPORTED_DATASETS


def preprocess_physionet2012(data):
    """
    Parameters
    ----------
    data : dict,
        A data dict from tsdb.load_dataset().

    Returns
    -------
    dict :
        A dict containing processed data.

    """
    X = data["X"].drop(data["static_features"], axis=1)

    def apply_func(df_temp):  # pad and truncate to set the max length of samples as 48
        missing = list(set(range(0, 48)).difference(set(df_temp["Time"])))
        missing_part = pd.DataFrame({"Time": missing})
        df_temp = pd.concat([df_temp, missing_part], ignore_index=False, sort=False)  # pad
        df_temp = df_temp.set_index("Time").sort_index().reset_index()
        df_temp = df_temp.iloc[:48]  # truncate
        return df_temp

    X = X.groupby("RecordID").apply(apply_func)
    X = X.drop("RecordID", axis=1)  #
    X = X.reset_index()
    X = X.drop(["level_1", "Time"], axis=1)
    return {"X": X, "y": data["y"]}


PREPROCESSING = {"physionet_2012": preprocess_physionet2012}


def load_specific_dataset(dataset_name, use_cache=True):
    """Load specific datasets supported by PyPOTS.
    Different from tsdb.load_dataset(), which only produces merely raw data,
    load_specific_dataset here does some preprocessing operations,
    like truncating time series to generate samples with the same length.

    Parameters
    ----------
    dataset_name : str,
        The name of the dataset to be loaded, which should be supported, i.e. in SUPPORTED_DATASETS.

    use_cache :
        Whether to use cache. This is an argument of tsdb.load_dataset().

    Returns
    -------
    data : dict,
        A dict contains the preprocessed dataset.
        Users only need to continue the preprocessing steps to generate the data they want,
        e.g. standardizing and splitting.

    """
    logger.info(
        f"Loading the dataset {dataset_name} with TSDB (https://github.com/WenjieDu/Time_Series_Database)..."
    )
    assert dataset_name in SUPPORTED_DATASETS, (
        f"Dataset {dataset_name} is not supported. "
        f"If you believe this dataset is valuable to be supported by PyPOTS,"
        f"please create an issue on GitHub "
        f"https://github.com/WenjieDu/PyPOTS/issues"
    )
    logger.info(f"Starting preprocessing {dataset_name}...")
    data = tsdb.load_dataset(dataset_name, use_cache)
    data = PREPROCESSING[dataset_name](data)
    return data
