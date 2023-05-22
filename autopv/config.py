import logging
import multiprocessing

import numpy as np

from enum import IntEnum
from hyperopt import hp
from hyperopt.pyll import scope

from ray import tune, air
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.stopper import TimeoutStopper, ExperimentPlateauStopper, CombinedStopper

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


class AdaptionConfiguration(IntEnum):
    none = 0
    AFB = 1
    AIB = 2


class ModelPool(IntEnum):
    default = 0
    nearby = 1


class MeasurementUnit(IntEnum):
    kWh = 0
    kW = 1


def get_plant_kwp_synthetic_selection():
    azmt = [0, 90, 180, 270]
    tilt = [15, 45, 75]

    plant_kwp = {}
    for a in azmt:
        for t in tilt:
            plant_kwp.update({f"azmt_{a}_tilt_{t}": 220})
    return plant_kwp


def create_search_space(plant: str):
    ridge = Ridge(fit_intercept=True)
    mlp = MLPRegressor(solver="adam")
    gbm = GradientBoostingRegressor(min_samples_leaf=0.2)
    rf = RandomForestRegressor(min_samples_leaf=0.2)
    svr = SVR(cache_size=1920)

    # Workaround for an unresolved issue of hyperopt. Otherwise, will raise an AttributeError in the hyperopt class Literal
    rf.estimators_ = None
    gbm.estimators_ = None

    search_space = {
        f"model_day_{plant}": hp.choice(f"model_day_{plant}", [
            {
                "module": ridge,
                "alpha": hp.loguniform("alpha_ridge", np.log(0.05), np.log(1)),
            },
            {
                "module": mlp,
                "activation": hp.choice("activation", ["logistic", "tanh", "relu"]),
                "hidden_layer_sizes":
                    hp.choice(
                        "hidden_layer_sizes",
                        [
                            (
                                scope.int(hp.loguniform("neurons_layer_1_1", np.log(10), np.log(100)))
                            ),
                            (
                                scope.int(hp.loguniform("neurons_layer_2_1", np.log(10), np.log(100))),
                                scope.int(hp.loguniform("neurons_layer_2_2", np.log(10), np.log(100)))
                            ),
                            (
                                scope.int(hp.loguniform("neurons_layer_3_1", np.log(10), np.log(100))),
                                scope.int(hp.loguniform("neurons_layer_3_2", np.log(10), np.log(100))),
                                scope.int(hp.loguniform("neurons_layer_3_3", np.log(10), np.log(100)))
                            ),
                        ]),
                "batch_size": scope.int(hp.loguniform("batch_size", np.log(16), np.log(1024)))
            },
            {
                "module": gbm,
                "learning_rate": hp.loguniform("learning_rate", np.log(1e-2), np.log(1e0)),
                "n_estimators": scope.int(hp.loguniform("gbm__n_estimators", np.log(10), np.log(300))),
                "max_depth": scope.int(hp.loguniform("gbm__max_depth", np.log(1), np.log(10)))
            },
            {
                "module": rf,
                "n_estimators": scope.int(hp.loguniform("rf__n_estimators", np.log(10), np.log(300))),
                "max_depth": scope.int(hp.loguniform("rf__max_depth", np.log(1), np.log(10)))
            },
            {
                "module": svr,
                "C": hp.loguniform("C", np.log(1e-2), np.log(1e2)),
                "epsilon": hp.loguniform("epsilon", np.log(1e-3), np.log(1))
            },
        ])
    }
    ray_tune_kwargs = {
        "tune_config": tune.TuneConfig(
            search_alg=HyperOptSearch(space=search_space, metric="loss", mode="min"),
            max_concurrent_trials=multiprocessing.cpu_count(),
            num_samples=-1,  # -1 evaluates infinite configuration samples until the stopper terminates the evaluation
        ),
        "run_config": air.RunConfig(
            local_dir="../results/ray_tune",
            stop=CombinedStopper(
                TimeoutStopper(timeout=15 * 60),
                ExperimentPlateauStopper(metric="loss", mode="min", std=0.001, top=10, patience=15)
            ),
            verbose=0
        )
    }
    ray_init_kwargs = {
        # "local_mode": True,  # True for debugging
        "log_to_driver": False,
        "logging_level": logging.FATAL
    }
    return ray_tune_kwargs, ray_init_kwargs
