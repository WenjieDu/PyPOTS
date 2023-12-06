"""
Preprocessing functions to load supported open-source time-series datasets.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import pandas as pd


def preprocess_physionet2012(data: dict) -> dict:
    """The preprocessing function for dataset PhysioNet-2012.

    Parameters
    ----------
    data :
        A data dict from tsdb.load_dataset().

    Returns
    -------
    dataset :
        A dict containing processed data, including:
            X : pandas.DataFrame,
                A dataframe contains all time series vectors from 11988 patients, distinguished by column `RecordID`.
            y : pandas.Series
                The 11988 classification labels of all patients, indicating whether they were deceased.
    """
    data["static_features"].remove("ICUType")  # keep ICUType for now
    # remove the other static features, e.g. age, gender
    X = data["X"].drop(data["static_features"], axis=1)

    def apply_func(df_temp):  # pad and truncate to set the max length of samples as 48
        missing = list(set(range(0, 48)).difference(set(df_temp["Time"])))
        missing_part = pd.DataFrame({"Time": missing})
        df_temp = pd.concat(
            [df_temp, missing_part], ignore_index=False, sort=False
        )  # pad the sample's length to 48 if it doesn't have enough time steps
        df_temp = df_temp.set_index("Time").sort_index().reset_index()
        df_temp = df_temp.iloc[:48]  # truncate
        return df_temp

    X = X.groupby("RecordID").apply(apply_func)
    X = X.drop("RecordID", axis=1)
    X = X.reset_index()
    ICUType = X[["RecordID", "ICUType"]].set_index("RecordID").dropna()
    X = X.drop(["level_1", "ICUType"], axis=1)

    dataset = {
        "X": X,
        "y": data["y"],
        "ICUType": ICUType,
    }

    return dataset
