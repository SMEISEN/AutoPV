import pandas as pd

from autopv.autopv import AutoPV
from autopv.config import MeasurementUnit, ModelPool, AdaptionConfiguration
from data.utils import load_example_data

if __name__ == "__main__":
    # Please note that the example data includes weather measurement data only.
    # Forecasting requires providing weather forecasting data as test data
    _, train, test = load_example_data()

    autopv = AutoPV(model_pool=ModelPool.default, measurement_unit=MeasurementUnit.kW,
                    adaption_config=AdaptionConfiguration.AIB, C=pd.Timedelta("28d"), K=pd.Timedelta("28d"))

    # Summaries will be created in the result-folder
    result_fit = autopv.fit(data=train, target_name="mixed_oriented_plant", target_kWp=440)
    result_predict = autopv.predict(data=test, target_name="mixed_oriented_plant", target_kWp=440,
                                    online_start=pd.Timestamp("2020-01-01 00:00:00"))
