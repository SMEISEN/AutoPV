from typing import Union

import numpy as np
import pandas as pd


def load_example_data(path_generation: str = "../data/synthetic_power_generation_181920.csv",
                      path_measurement: str = "../data/weather_rheinstetten_181920.csv",
                      train_start: Union[int, str] = "2018-01-01 00:00",
                      train_end: Union[int, str] = "2019-12-30 23:45",
                      test_start: Union[int, str] = "2019-12-31 00:00",
                      test_end: Union[int, str] = "2020-12-31 23:45",
                      drop_night_for_training: bool = True):

    # Load generation
    data_g = pd.read_csv(path_generation,
                         index_col="MESS_DATUM",
                         parse_dates=["MESS_DATUM"],
                         infer_datetime_format=True,
                         sep=",")

    # Load data measurement
    data_m = pd.read_csv(path_measurement,
                         index_col="MESS_DATUM",
                         parse_dates=["MESS_DATUM"],
                         infer_datetime_format=True,
                         sep=",")
    data_m.rename(columns={"GHI[W/m^2]": "radiation",
                           "TU[degC]": "temperature",
                           "V[m/s]": "wind_speed"}, inplace=True)

    data = pd.concat(
        [data_g, data_m[["radiation", "temperature", "wind_speed"]]], axis=1)
    data.dropna(inplace=True)
    data[f"mixed_oriented_plant"] = np.array(data[f"azmt_80_tilt_40"].values + data[f"azmt_260_tilt_40"])

    # Split data into train and pv_template data
    if all([isinstance(train_start, int), isinstance(train_end, int),
            isinstance(test_start, int), isinstance(test_end, int)]):
        train = data.iloc[train_start:train_end, :].copy()
        test = data.iloc[test_start:test_end, :].copy()
    elif all([isinstance(train_start, str), isinstance(train_end, str),
              isinstance(test_start, str), isinstance(test_end, str)]):
        train = data.loc[train_start:train_end, :].copy()
        test = data.loc[test_start:test_end, :].copy()
    else:
        raise TypeError("The parameter split is neither an integer nor a datetime!")

    train.drop(train[train["radiation"] == 0].index, inplace=True) if drop_night_for_training else None

    return data, train, test
