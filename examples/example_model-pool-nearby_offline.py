from autopv.autopv import AutoPV
from autopv.config import MeasurementUnit, ModelPool, AdaptionConfiguration, get_plant_kwp_synthetic_selection
from data.utils import load_example_data

if __name__ == "__main__":
    # Please note that the example data includes weather measurement data only.
    # Forecasting requires providing weather measurement data as train data and weather forecasting data as test data.
    # Summaries will be created in the result-folder.
    _, train, test = load_example_data()

    nearby_plants_kWp = get_plant_kwp_synthetic_selection()

    autopv = AutoPV(target_name="mixed_oriented_plant", target_kWp=440,
                    model_pool=ModelPool.nearby, nearby_plants_kWp=nearby_plants_kWp, measurement_unit=MeasurementUnit.kW,
                    adaption_config=AdaptionConfiguration.none, C=None, K=None)

    # Fit the ensemble pool models and estimate the ensemble weights offline based on training data.
    result_fit = autopv.fit(data=train)

    # Predict with the offline-fitted AutoPV model.
    result_predict = autopv.predict(data=test)
