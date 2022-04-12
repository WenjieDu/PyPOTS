"""
Utilities for loading specific datasets.
"""
# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

import gzip
import os
import pickle
import shutil
import tempfile
import warnings
from sys import exit
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

CACHED_DATASET_DIR = os.path.join(os.path.dirname(__file__), ".cached_datasets")

DATABASE = {
    # github.com/WenjieDu/Time_Series_Database/tree/main/datasets/PhysioNet-2012
    'physionet_2012': [
        'https://www.physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz',
        'https://www.physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz',
        'https://www.physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz',
        'https://www.physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt',
        'https://www.physionet.org/files/challenge-2012/1.0.0/Outcomes-b.txt',
        'https://www.physionet.org/files/challenge-2012/1.0.0/Outcomes-c.txt',
    ],

    # github.com/WenjieDu/Time_Series_Database/tree/main/datasets/ElectricityLoadDiagrams
    'electricity_load_diagrams':
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip',

    # github.com/WenjieDu/Time_Series_Database/tree/main/datasets/BeijingMultiSiteAirQuality
    'beijing_multisite_air_quality':
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip',

}

AVAILABLE_DATASETS = list(DATABASE.keys())


def window_truncate(feature_vectors, seq_len):
    """ Generate time series samples, truncating windows from time-series data with a given sequence length.

    Parameters
    ----------
    feature_vectors : array, shape of [total_length, feature_num]
        Time-series data.
    seq_len : int,
        Sequence length.

    Returns
    -------
    array,
        Truncated time series with given sequence length.
    """
    start_indices = np.asarray(range(feature_vectors.shape[0] // seq_len)) * seq_len
    sample_collector = []
    for idx in start_indices:
        sample_collector.append(feature_vectors[idx: idx + seq_len])

    return np.asarray(sample_collector).astype('float32')


def _download_and_extract(url, saving_path):
    """ Download dataset from the given url and extract to the given saving path.

    Parameters
    ----------
    url : str,
        URL of the dataset to be downloaded.
    saving_path : str,
        Path to save extracted dataset.

    Returns
    -------
    saving_path if successful else None
    """
    no_need_decompression_format = ['csv', 'txt']
    supported_compression_format = ["zip", "tar", "gz", "bz", "xz"]

    # truncate the file name from url
    file_name = os.path.basename(url)
    suffix = file_name.split('.')[-1]

    if suffix in no_need_decompression_format:
        raw_data_saving_path = os.path.join(saving_path, file_name)
    elif suffix in supported_compression_format:
        # create temp dir for raw data saving
        tmp_dir = tempfile.mkdtemp()
        raw_data_saving_path = os.path.join(tmp_dir, file_name)
    else:
        warnings.warn(
            "The compression format is not supported, aborting. "
            "If necessary, please create a pull request to add according supports.",
            category=RuntimeWarning
        )
        return None

    # download and save the raw dataset
    try:
        urlretrieve(url, raw_data_saving_path)
    # except Exception as e:
    except Exception as e:
        shutil.rmtree(saving_path, ignore_errors=True)
        print(f"Exception: {e}\n"
              f"Download failed. Aborting.")
        exit()
    print(f"Successfully downloaded data to {raw_data_saving_path}.")

    try:
        os.makedirs(saving_path, exist_ok=True)
        shutil.unpack_archive(raw_data_saving_path, saving_path)
        print(f"Successfully extracted data to {saving_path}")
    except gzip.BadGzipFile:
        warnings.warn("The compressed file is corrupted, aborting.", category=RuntimeWarning)
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return saving_path


def download_and_extract(dataset_name, dataset_saving_path):
    """ Wrapper of _download_and_extract.

    Parameters
    ----------
    dataset_name :
    dataset_saving_path :

    Returns
    -------

    """
    os.makedirs(dataset_saving_path)
    if isinstance(DATABASE[dataset_name], list):
        for link in DATABASE[dataset_name]:
            _download_and_extract(link, dataset_saving_path)
    else:
        _download_and_extract(DATABASE[dataset_name], dataset_saving_path)


def delete_all_cached_data():
    """ Delete CACHED_DATASET_DIR if exists.
    """
    # if CACHED_DATASET_DIR does not exist, abort
    if not os.path.exists(CACHED_DATASET_DIR):
        print('No cached data. Aborting.')
        exit()
    # if CACHED_DATASET_DIR exists, then purge
    try:
        print(f'Purging all cached data under {CACHED_DATASET_DIR}...')
        shutil.rmtree(CACHED_DATASET_DIR, ignore_errors=True)
        # check if succeed
        if not os.path.exists(CACHED_DATASET_DIR):
            print('Purge successfully.')
        else:
            raise FileExistsError(f'Deleting operation failed. {CACHED_DATASET_DIR} still exists.')
    except shutil.Error:
        raise shutil.Error('Operation failed.')


def pickle_dump(data, path):
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except pickle.PicklingError:
        print('Pickling failed. No cache will be saved.')
    return path


def pickle_load(path):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    except pickle.UnpicklingError:
        print('Cached data corrupted. Aborting...\n'
              'Please rerun func load_specific_dataset with option use_cache=False')
    return data


def load_specific_dataset(dataset_name, use_cache=True):
    """ Load dataset with given name.

    Parameters
    ----------
    dataset_name : str,
        The name of the specific dataset in DATABASE.

    use_cache : bool,
        Whether to use cache (including data downloading and processing)

    Returns
    -------
    pandas.DataFrame,
        Loaded dataset.
    """
    assert dataset_name in DATABASE.keys(), f'Input dataset name "{dataset_name}" is not in the database {DATABASE}.'
    dataset_saving_path = os.path.join(CACHED_DATASET_DIR, dataset_name)
    if not os.path.exists(dataset_saving_path):  # if the dataset is not cached, then download it
        download_and_extract(dataset_name, dataset_saving_path)
    else:
        if use_cache:
            print(f'Dataset {dataset_name} has already been downloaded. Start processing directly...')
        else:
            # if not use cache, then delete the downloaded data dir (including processing cache)
            shutil.rmtree(dataset_saving_path, ignore_errors=True)
            download_and_extract(dataset_name, dataset_saving_path)

    # if cached, then load directly
    cache_path = os.path.join(dataset_saving_path, dataset_name + '_cache.pkl')
    if os.path.exists(cache_path):
        print(f'Dataset {dataset_name} has already been cached. Loading directly...')
        result = pickle_load(cache_path)
    else:
        try:
            if dataset_name == 'physionet_2012':
                result = load_physionet2012(dataset_saving_path)
            elif dataset_name == 'electricity_load_diagrams':
                result = load_electricity(dataset_saving_path)
            elif dataset_name == 'beijing_multisite_air_quality':
                result = load_beijing_air_quality(dataset_saving_path)
            print('Loading finished.')
        except FileExistsError:
            shutil.rmtree(dataset_saving_path, ignore_errors=True)
            warnings.warn(
                'Dataset corrupted, already deleted. Please rerun load_specific_dataset() to re-download the raw data.'
            )
        pickle_dump(result, cache_path)
    return result


def load_physionet2012(local_path):
    """ Load dataset PhysioNet Challenge 2012, which is a time-series classification dataset.

    Parameters
    ----------
    local_path : str,
        The local path of dir saving the raw data of PhysioNet Challenge 2012.

    Returns
    -------
    data : dict
        A dictionary contains X and y:
            X : pandas.DataFrame,
                Time-series feature vectors.
            y : pandas.Series,
                Classification labels.

    Notes
    -----
    The preprocessing workflow is the same with the one used in paper :cite:`du2022SAITS`.
    All samples contain 48 time steps. Truncated if the sample has more than 48 steps. Padded if
    the sample has less than 48 steps. Static features such as 'Age', 'Gender', 'ICUType', 'Height',
    are removed. Column 'Time' also gets removed. Following 12 samples are dropped because of containing
    no time-series information at all: 147514, 142731, 145611, 140501, 155655, 143656, 156254, 150309,
    140936, 141264, 150649, 142998.
    """

    time_series_measurements_dir = ['set-a', 'set-b', 'set-c']
    outcome_files = ['Outcomes-a.txt', 'Outcomes-b.txt', 'Outcomes-c.txt']

    outcome_collector = []
    for o_ in outcome_files:
        outcome_file_path = os.path.join(local_path, o_)
        outcome = pd.read_csv(outcome_file_path).set_index('RecordID')['In-hospital_death']
        outcome_collector.append(outcome)
    y = pd.concat(outcome_collector)

    all_recordID = []
    df_collector = []

    # iterate over all samples
    for m_ in time_series_measurements_dir:
        raw_data_dir = os.path.join(local_path, m_)
        for filename in os.listdir(raw_data_dir):
            recordID = int(filename.split('.txt')[0])
            with open(os.path.join(raw_data_dir, filename), 'r') as f:
                df_temp = pd.read_csv(f)
            df_temp['Time'] = df_temp['Time'].apply(lambda x: int(x.split(':')[0]))
            df_temp = df_temp.pivot_table('Value', 'Time', 'Parameter')
            df_temp = df_temp.reset_index()  # take Time from index as a col
            if len(df_temp) == 1:
                print(f'Pass {recordID}, because its len==1, having no time series data')
                continue
            all_recordID.append(recordID)  # only count valid recordID

            if df_temp.shape[0] != 48:
                missing = list(set(range(0, 48)).difference(set(df_temp['Time'])))
                missing_part = pd.DataFrame({'Time': missing})
                df_temp = df_temp.append(missing_part, ignore_index=False, sort=False)
                df_temp = df_temp.set_index('Time').sort_index().reset_index()

            df_temp = df_temp.iloc[:48]  # only take 48 hours, some samples may have more records, like 49 hours
            df_temp['RecordID'] = recordID
            df_temp['Age'] = df_temp.loc[0, 'Age']
            df_temp['Height'] = df_temp.loc[0, 'Height']
            df_collector.append(df_temp)

    df = pd.concat(df_collector, sort=True)
    df = df.drop(['Age', 'Gender', 'ICUType', 'Height'], axis=1)
    df = df.reset_index(drop=True)
    X = df.drop('Time', axis=1)  # we don't need Time column
    data = {
        'X': X,
        'y': y
    }
    return data


def load_electricity(local_path):
    """ Load dataset PhysioNet Challenge 2012.

    Parameters
    ----------
    local_path : str,
        The local path of dir saving the raw data of Electricity Load Diagrams.

    Returns
    -------
    data : dict
        A dictionary contains X:
            X : pandas.DataFrame
                The time-series data of Electricity Load Diagrams.
    """
    file_path = os.path.join(local_path, 'LD2011_2014.txt')
    df = pd.read_csv(file_path, index_col=0, sep=';', decimal=',')
    df.index = pd.to_datetime(df.index)
    # feature_names = df.columns.tolist()
    # feature_num = len(feature_names)
    df['datetime'] = pd.to_datetime(df.index)
    data = {
        'X': df,
    }
    return data


def load_beijing_air_quality(local_path):
    """ Load dataset Beijing Multi-site Air Quality.

    Parameters
    ----------
    local_path : str,
        The local path of dir saving the raw data of Beijing Multi-site Air Quality.

    Returns
    -------
    data : dict
        A dictionary contains X:
            X : pandas.DataFrame
                The time-series data of Beijing Multi-site Air Quality.
    """
    dir_path = os.path.join(local_path, 'PRSA_Data_20130301-20170228')
    df_collector = []
    file_list = os.listdir(dir_path)
    for filename in file_list:
        file_path = os.path.join(dir_path, filename)
        current_df = pd.read_csv(file_path)
        df_collector.append(current_df)
        print(f'Reading {file_path}, data shape {current_df.shape}')
    df = pd.concat(df_collector, axis=0)
    data = {
        'X': df,
    }
    return data
