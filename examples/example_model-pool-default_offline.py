from autopv.autopv import AutoPV
from autopv.config import MeasurementUnit, ModelPool, AdaptionConfiguration
from data.utils import load_example_data

if __name__ == "__main__":
    # Please note that the example data includes weather measurement data only.
    # Forecasting requires providing weather forecasting data as test data.
    # Summaries will be created in the result-folder.
    _, train, test = load_example_data()

    autopv = AutoPV(target_name="mixed_oriented_plant", target_kWp=440, latitude=48.9685, longitude=8.30704,
                    model_pool=ModelPool.default, measurement_unit=MeasurementUnit.kW,
                    adaption_config=AdaptionConfiguration.none, C=None, K=None)

    # Estimate the ensemble weights offline using the training data.
    result_fit = autopv.fit(data=train)

    # Predict with the offline-fitted AutoPV model.
    result_predict = autopv.predict(data=test)

    # Get AutoPV's ensemble weights
    weights = autopv.weights_
