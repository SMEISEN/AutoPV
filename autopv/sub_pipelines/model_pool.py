from copy import deepcopy

from pywatts.callbacks import LinePlotCallback, CSVCallback
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.modules import FunctionModule

from autopv.sub_pipelines.utils import assign_inputs_to_subpipeline, add_metrics
from autopv.config import ModelPool


def create_pvlib_model(plant: str, modules: dict, pipeline_preprocessing: Pipeline, pipeline_model_pool: Pipeline,
                       kwp: int = 220, summaries: bool = False):
    pipeline_preprocessing = assign_inputs_to_subpipeline(pipeline_left=pipeline_preprocessing,
                                                          pipeline_right=pipeline_model_pool)

    temperature = pipeline_preprocessing["temperature"]
    ghi = pipeline_preprocessing["radiation"]
    wind_speed = pipeline_preprocessing["wind_speed"]

    pv = modules["pv_lib"](temperature=temperature, ghi=ghi, wind_speed=wind_speed)
    corrected = FunctionModule(lambda x: x.clip(min=0) / kwp, name=modules["pv_lib"].name)(x=pv)

    if summaries:
        target_scaled = pipeline_preprocessing[f"scale_{plant}"]
        add_metrics(y_hat=corrected, y=target_scaled, suffix=corrected.step.name)

    return corrected


def create_individual_model(plant: str, pipeline_preprocessing: Pipeline, pipeline_model_pool: Pipeline,
                            modules: dict,
                            computation_mode: ComputationMode, pipe_model_path: str):
    """
    Creates an individual model.
    """

    pipe_model = Pipeline(path=pipe_model_path, name=f"pipe_tuning_{plant}")
    step_info = assign_inputs_to_subpipeline(pipeline_left=pipeline_preprocessing, pipeline_right=pipe_model)

    target_scaled = step_info[f"scale_{plant}"]
    inputs = {"weather_features": step_info["weather_features"], "calendar_features": step_info["calendar_features"]}
    dependency = step_info["radiation"]

    if computation_mode == ComputationMode.Default:

        # Forecasting model for day times
        forecast_day = modules["reg_day"](
            **inputs,
            target=target_scaled,
            computation_mode=computation_mode,
        )

        # Forecasting model for night times
        forecast_night = modules["reg_night"](
            **inputs,
            target=target_scaled,
            computation_mode=computation_mode,
        )

        conditioned = modules["condition_day_night"](
            dependency=dependency,
            if_true=forecast_day,
            if_false=forecast_night,
        )

        modules["correction_non_negative"](
            dependency=conditioned,
            if_true=forecast_day,
            if_false=forecast_night,
        )

        modules["tuner"].set_params(**{"estimator": deepcopy(pipe_model)})

    pipeline_preprocessing = assign_inputs_to_subpipeline(pipeline_left=pipeline_preprocessing,
                                                          pipeline_right=pipeline_model_pool)
    target_scaled = pipeline_preprocessing[f"scale_{plant}"]

    individual_model = modules["tuner"](
        target=target_scaled,
        **{key: value[1] for key, value in pipeline_model_pool.start_steps.items()},  # pass start steps
        computation_mode=computation_mode,
        config_summary=["k_best_config_"],
        callbacks=[LinePlotCallback(prefix="plot"), CSVCallback(prefix="csv")]
    )

    return individual_model


def create_pipeline_model_engineering(plants_kWp: dict, plant_test: str, model_pool: ModelPool,
                                      preprocessing: Pipeline, modules: dict):
    """
    Creates the training pipeline for the PV template including the linear regression and the dummy regression.
    """

    # Create train pipeline
    pipeline_model_pool = Pipeline(path=f"../results/{plant_test}/model_engineering", name="pipe_models")

    # Train models for ensemble
    for plant, kwp in plants_kWp.items():
        if model_pool == ModelPool.nearby:  # nearby plants model pool
            create_individual_model(plant=plant, pipeline_preprocessing=preprocessing,
                                    pipeline_model_pool=pipeline_model_pool, modules=modules[plant],
                                    computation_mode=ComputationMode.Default,
                                    pipe_model_path=f"./results/{plant_test}/model_{plant}/")
        elif model_pool == ModelPool.default:  # default model pool
            create_pvlib_model(plant=plant, modules=modules[plant], pipeline_preprocessing=preprocessing,
                               pipeline_model_pool=pipeline_model_pool, kwp=kwp)
        else:
            raise Exception(f"Model pool {model_pool} not known!")

    return pipeline_model_pool
