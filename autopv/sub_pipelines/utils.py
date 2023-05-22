import os

import cloudpickle
from typing import List

from pywatts.core.step_information import StepInformation

from pywatts.core.pipeline import Pipeline
from pywatts.core.computation_mode import ComputationMode
from pywatts.modules import SKLearnWrapper
from pywatts.summaries import RMSE, MAE

from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

from pywatts_modules.pvlib_wrapper import PVLibWrapper
from pywatts_modules.condition import Condition
from pywatts_modules.hyperparameter_optimization import RayTuneWrapper, SplitMethod
from pywatts_modules.nmae_summary import nMAE
from pywatts_modules.nrmse_summary import nRMSE

from autopv.config import ModelPool, create_search_space


def assign_inputs_to_subpipeline(pipeline_left: Pipeline, pipeline_right: Pipeline,
                                 steps_set_transform: List[str] = None):
    """
    Stacks two pipelines together.
    :param pipeline_left: The first part of the pipeline.
    :type pipeline_left: Pipeline
    :param pipeline_right: The second part of the pipeline that is executed after pipeline_left.
    :type pipeline_right: Pipeline
    :param steps_set_transform: Names of steps, whose default computation mode should be set to transform.
    :type steps_set_transform: List
    """
    if steps_set_transform is None:
        steps_set_transform = []

    kwargs = {}
    for step in pipeline_left.start_steps.keys():
        kwargs.update({f"{step}": pipeline_right[step]})

    for key, value in pipeline_left.id_to_step.items():
        if value.name in steps_set_transform:
            pipeline_left.id_to_step[key].default_run_setting.computation_mode = ComputationMode.Transform

    return pipeline_left(**kwargs)


def create_modules(plant: str, model_pool: ModelPool):
    """
    Creates an individual model.
    """

    # default model pool
    if model_pool == ModelPool.default:
        return {
            "pv_lib":
                PVLibWrapper(latitude=48.9685, longitude=8.30704, altitude=116,
                             surface_azimuth=int(plant.split("_")[1]), surface_tilt=int(plant.split("_")[3]),
                             name=f"model_{plant}")
        }

    # nearby plants model pool
    ray_tune_kwargs, ray_init_kwargs = create_search_space(plant=plant)
    return {
        "reg_day":
            SKLearnWrapper(module=LinearRegression(fit_intercept=True),
                           name=f"model_day_{plant}"),
        "reg_night":
            SKLearnWrapper(module=DummyRegressor(strategy='constant', constant=0.0),
                           name=f"model_night_{plant}"),
        "condition_day_night":
            Condition(condition=lambda x: x > 0,
                      name=f"cond_day-night_{plant}"),
        "correction_non_negative":
            Condition(condition=lambda x: x > 0,
                      name=f"cond_non-negative_{plant}"),
        "tuner": RayTuneWrapper(name=f"model_{plant}",
                                replace_weather=False,  # training: False, online: True
                                estimator=None,  # estimator is set later
                                cv=5,  # comment for old default pool
                                split_method=SplitMethod.RandomSample,  # comment for old default pool
                                ray_tune_kwargs=ray_tune_kwargs, ray_init_kwargs=ray_init_kwargs,
                                k_best=1, refit_only=True)
    }


def save_modules(pipeline_modules: dict, dry: str = "../results/individual_ensembling/models"):
    for plant, modules in pipeline_modules.items():
        path = f"{dry}/{plant}"
        if not os.path.exists(path):
            os.makedirs(path)
        for module_name, module_object in modules.items():
            cloudpickle.dump(module_object, open(f"{path}/{module_name}.pickle", 'wb'))


def load_modules(pipeline_modules: dict, dry: str = "../results/individual_ensembling/models"):
    for plant, modules in pipeline_modules.items():
        for module_name, module_object in modules.items():
            with open(f"{dry}/{plant}/{module_name}.pickle", 'rb') as file:
                pipeline_modules[plant][module_name] = cloudpickle.load(file)
    return pipeline_modules


def add_metrics(y_hat: StepInformation, y: StepInformation, suffix: str):
    RMSE(name=f"RMSE_{suffix}")(y_hat=y_hat, y=y)
    MAE(name=f"MAE_{suffix}")(y_hat=y_hat, y=y)

    nRMSE(name=f"nRMSEavg_{suffix}")(y_hat=y_hat, y=y)
    nMAE(name=f"nMAEavg_{suffix}")(y_hat=y_hat, y=y)
