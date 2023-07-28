import pandas as pd

from pywatts.callbacks import LinePlotCallback, CSVCallback
from pywatts.conditions import PeriodicCondition
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts_modules.periodic_condition_increasing_batch import PeriodicConditionIncreasingBatch
from autopv.config import ModelPool, AdaptionConfiguration

from autopv.sub_pipelines.model_pool import create_individual_model, create_pvlib_model
from autopv.sub_pipelines.utils import assign_inputs_to_subpipeline


def create_pipeline_ensemble(plant_test, pipeline_preprocessing: Pipeline,
                             modules: dict, model_pool: ModelPool,
                             K: pd.Timedelta, C: pd.Timedelta = None,
                             adaption_configuration: AdaptionConfiguration = AdaptionConfiguration.none):
    """
    Creates the ensemble pipeline with fitted forecasts.
    """

    # Create ensemble pipeline
    pipeline_ensemble = Pipeline(path=f"../results/{plant_test}/ensemble", name="pipe_ensemble", batch=C)

    step_info = assign_inputs_to_subpipeline(pipeline_left=pipeline_preprocessing, pipeline_right=pipeline_ensemble)

    if adaption_configuration == AdaptionConfiguration.none:
        computation_mode = ComputationMode.Default
    else:
        computation_mode = ComputationMode.Refit

    ensemble_model_pool = {}
    for name in modules.keys():
        if "scaler" in name or "features" in name or "ensemble" in name:
            continue

        pipeline_model_path = f"../results/{plant_test}/model_{name}/"

        if model_pool == ModelPool.nearby:  # nearby plants model pool
            if name == plant_test:
                continue  # leave test plant out of the ensemble model pool
            ensemble_model_pool[name] = create_individual_model(plant=name, pipeline_preprocessing=pipeline_preprocessing,
                                                                pipeline_model_pool=pipeline_ensemble, modules=modules[name],
                                                                computation_mode=ComputationMode.Transform,
                                                                pipe_model_path=pipeline_model_path)
        elif model_pool == ModelPool.default:  # default model pool
            ensemble_model_pool[name] = create_pvlib_model(plant=name, modules=modules[name],
                                                           pipeline_preprocessing=pipeline_preprocessing,
                                                           pipeline_model_pool=pipeline_ensemble)
        else:
            raise Exception(f"Model pool {model_pool} not known!")

    # refit conditions
    if adaption_configuration == adaption_configuration.AFB:
        refit_conditions = [PeriodicCondition(name="PeriodicEnsemble",
                                              num_steps=1,  # refits every C period
                                              refit_batch=K)]
    elif adaption_configuration == adaption_configuration.AIB:
        refit_conditions = [PeriodicConditionIncreasingBatch(name="PeriodicIncreasingEnsemble",
                                                             num_steps=1,  # refits every C period
                                                             refit_batch=pd.Timedelta(days=0),
                                                             refit_batch_append=K)]
    else:
        refit_conditions = []

    ensemble = modules["ensemble"](
        **ensemble_model_pool, target=step_info[f"scale_{plant_test}"],
        refit_conditions=refit_conditions,
        computation_mode=computation_mode,
        config_summary=["weights_"],
        callbacks=[LinePlotCallback(prefix="plot"), CSVCallback(prefix="csv")])

    # the ensemble defaults to equal weights and is then periodically fitted
    ensemble.step.module.is_fitted = True

    return pipeline_ensemble
