from pywatts.core.pipeline import Pipeline
from pywatts.modules import CustomScaler
from pywatts.callbacks import LinePlotCallback, CSVCallback
from pywatts.core.computation_mode import ComputationMode


from autopv.sub_pipelines.utils import assign_inputs_to_subpipeline, add_metrics
from autopv.config import MeasurementUnit


def create_pipeline_postprocessing(plants_kWp_test: dict, measurement_unit: MeasurementUnit, samples_per_hour: int,
                                   plant_test, pipeline_preprocessing: Pipeline, pipeline_ensemble: Pipeline):
    """
    Creates the post-processing pipeline for the PV template for re-scaling to individual PV peak powers.
    """

    # Create postprocessing pipeline
    pipeline = Pipeline(path=f"../results/{plant_test}/postprocessing", name="pipe_postprocessing")

    # Define which data goes into the preprocessing pipeline
    step_info_preprocessing = assign_inputs_to_subpipeline(pipeline_left=pipeline_preprocessing, pipeline_right=pipeline)
    step_info_pipeline_ensemble = assign_inputs_to_subpipeline(pipeline_left=pipeline_ensemble, pipeline_right=pipeline)

    for plant_test, peak_power in plants_kWp_test.items():
        # rescale ensemble
        scaler_name = f"rescale_ensemble_{plant_test}"
        if measurement_unit == MeasurementUnit.kW:
            scaler = CustomScaler(multiplier=1 / peak_power, name=scaler_name)
        elif measurement_unit == MeasurementUnit.kWh:
            # samples_per_hour: convert the energy metering time series into the mean power time series
            # peak_power: divide mean power time series by peak power rating
            scaler = CustomScaler(multiplier=samples_per_hour / peak_power, name=scaler_name)
        else:
            raise Exception(f"Measurement unit {measurement_unit} not known!")

        inverse_scaled = scaler(
            x=step_info_pipeline_ensemble[f"ensemble_{plant_test}"],
            computation_mode=ComputationMode.Transform,
            use_inverse_transform=True, callbacks=[LinePlotCallback("plot"), CSVCallback("csv")])

        add_metrics(y_hat=inverse_scaled, y=step_info_preprocessing[f"{plant_test}"], suffix=f"_ensemble_rescaled_{plant_test}")

    return pipeline
