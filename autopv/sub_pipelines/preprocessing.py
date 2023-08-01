from pywatts.core.pipeline import Pipeline
from pywatts.modules import CustomScaler

from autopv.config import MeasurementUnit, ModelPool


def create_pipeline_preprocessing(plant_kWp_train: dict, plant_kWp_test: dict, plant_test: str, model_pool: ModelPool,
                                  modules: dict, measurement_unit: MeasurementUnit, samples_per_hour: int):
    """
    Creates the pre-processing pipeline for the PV template including scaling, normalizing, and feature extraction.
    """

    # Create preprocessing pipeline
    pipeline = Pipeline(path=f"../results/{plant_test}/preprocessing", name="pipe_preprocessing")

    # assign inputs
    input_step = pipeline[plant_test]

    # Normalize PV generation profiles
    plants_kWp = {**plant_kWp_train, **plant_kWp_test} if model_pool == ModelPool.nearby else plant_kWp_test
    for plant, peak_power in plants_kWp.items():
        if measurement_unit == MeasurementUnit.kW:
            CustomScaler(multiplier=1 / peak_power, name=f"scale_{plant}")(x=pipeline[plant])
        elif measurement_unit == MeasurementUnit.kWh:
            # samples_per_hour: convert the energy metering time series into the mean power time series
            # peak_power: divide mean power time series by peak power rating
            CustomScaler(multiplier=samples_per_hour / peak_power, name=f"scale_{plant}")(x=pipeline[plant])
        else:
            raise Exception(f"Measurement unit {measurement_unit} not known!")

    if model_pool == ModelPool.nearby:  # nearby plants model pool
        # Scale weather data
        scaled_radiation = modules["scaler_radiation"](x=pipeline["radiation"])
        scaled_temperature = modules["scaler_temperature"](x=pipeline["temperature"])

        # Polynomial features
        modules["features_polynomial"](f1=scaled_radiation, f2=scaled_temperature)

        # Calendar features
        modules["features_calendar"](x=input_step)

    pipeline[plant_test].step.last = True
    pipeline["radiation"].step.last = True
    pipeline["temperature"].step.last = True
    pipeline["wind_speed"].step.last = True

    return pipeline
