"""
Utilities for loading specific datasets.
"""
# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT

import gzip
import os
import shutil
import tempfile
import warnings
from urllib.request import urlretrieve

import pandas as pd

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


def download_and_extract(url, saving_path):
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
        urlretrieve(url, raw_data_saving_path)
        print(f"Successfully downloaded data to {raw_data_saving_path}.")
    elif suffix in supported_compression_format:
        # create temp dir for raw data saving
        tmp_dir = tempfile.mkdtemp()
        raw_data_saving_path = os.path.join(tmp_dir, file_name)
        # download and save the raw dataset
        urlretrieve(url, raw_data_saving_path)
        print(f"Successfully downloaded data to {raw_data_saving_path}.")
        os.makedirs(saving_path, exist_ok=True)
        try:
            shutil.unpack_archive(raw_data_saving_path, saving_path)
            print(f"Successfully extracted data to {saving_path}")
        except gzip.BadGzipFile:
            warnings.warn("The compressed file is corrupted, aborting.", category=RuntimeWarning)
            return None
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        warnings.warn(
            "The compression format is not supported, aborting. "
            "If necessary, please create a pull request to add according supports.",
            category=RuntimeWarning
        )
        return None
    return saving_path


def load_specific_dataset(dataset_name):
    """ Load dataset with given name.

    Parameters
    ----------
    dataset_name : str,
        The name of the specific dataset in DATABASE.

    Returns
    -------
    pandas.DataFrame,
        Loaded dataset.
    """
    assert dataset_name in DATABASE.keys(), f'Input dataset name "{dataset_name}" is not in the database {DATABASE}.'
    cached_dataset_dir = os.path.join(os.path.dirname(__file__), "..", ".cached_datasets")
    dataset_saving_path = os.path.join(cached_dataset_dir, dataset_name)
    if not os.path.exists(dataset_saving_path):  # if the dataset is not cached, then download it
        os.makedirs(dataset_saving_path)
        if isinstance(DATABASE[dataset_name], list):
            for link in DATABASE[dataset_name]:
                download_and_extract(link, dataset_saving_path)
        else:
            download_and_extract(DATABASE[dataset_name], dataset_saving_path)
    else:
        print(f'Dataset {dataset_name} is already cached. Directly loading...')

    # if cached, then load directly
    try:
        if dataset_name == 'physionet_2012':
            return load_physionet2012(dataset_saving_path)
    except FileExistsError:
        shutil.rmtree(dataset_saving_path, ignore_errors=True)
        warnings.warn(
            'Dataset corrupted, already deleted. Please reload it to re-download the raw data.'
        )


def load_physionet2012(local_path):
    """ Load dataset PhysioNet Challenge 2012, which is a time-series classification dataset.

    Notes
    -----


    Returns
    -------
    X : pandas.DataFrame,
        Time-series feature vectors.
    y : pandas.Series,
        Classification labels.
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

    return X, y
