import pandas as pd

from autopv.autopv import AutoPV
from autopv.config import MeasurementUnit, ModelPool, AdaptionConfiguration
from data.utils import load_example_data

if __name__ == "__main__":
    # Please note that the example data includes weather measurement data only.
    # Forecasting requires providing weather forecasting data as test data.
    # Summaries will be created in the result-folder.
    _, train, test = load_example_data()

    autopv = AutoPV(target_name="mixed_oriented_plant", target_kWp=440,
                    model_pool=ModelPool.default, measurement_unit=MeasurementUnit.kW,
                    adaption_config=AdaptionConfiguration.AIB, C=pd.Timedelta("28d"), K=pd.Timedelta("28d"))

    # Default model pool in the online-mode does not require fitting any model beforehand.
    result_fit = None

    # Estimate the ensemble weights online every C days using the most recent K samples
    # and predict with the online-fitted AutoPV model.
    result_predict = autopv.predict(data=test, online_start=pd.Timestamp("2020-01-01 00:00:00"))
