import json
import operator
import random
import logging
import warnings

import numpy as np
import xarray as xr
from copy import deepcopy
from enum import IntEnum
from typing import Dict, Union, Callable

import ray
from ray import tune
from ray.tune.tuner import Tuner

from pywatts.core.base import BaseEstimator, BaseTransformer
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.exceptions import WrongParameterException
from pywatts.core.pipeline import Pipeline
from pywatts.core.pipeline_step import PipelineStep
from pywatts.core.run_setting import RunSetting
from pywatts.modules.wrappers.base_wrapper import BaseWrapper
from pywatts.utils._split_kwargs import split_kwargs
from pywatts.utils._xarray_time_series_utils import _get_time_indexes

logger = logging.getLogger(__name__)


class LossMetric(IntEnum):
    """
    Enum which contains the different loss metrics of the ensemble module.
    """
    RMSE = 1
    MAE = 2
    CRPS = 3
    PINBALL = 4
    CRE = 5
    CAL = 6
    WPI = 7
    WPI_Weighted = 8


class SplitMethod(IntEnum):
    """
    Enum which contains the different Validation metrics of the ensemble module.
    """
    RandomSample = 1
    BlockedSample = 2
    RollingOriginUpdate = 3
    RollingOriginRetrain = 4
    RollingWindow = 5


class Splitter:
    """
    class that implements different validation strategies
    """

    def __init__(self, method, cv, val_split, targets, inputs):
        """
        :param method: indicates which cv strategy should be used
        :param cv: indicates the number of cv folds
        :param val_split: specifies how much of the dataset should be used for training (only used by a few of the strategies)
        :param targets: dict which contains the target values
        :param inputs:dict which contains the input values
        """
        self.method = method
        self.cv = cv
        self.val_split = val_split
        self.targets = targets
        self.inputs = inputs
        self.train_splits = []

    def get_splits(self):
        """
        function implements the different validation strategies.
        5 different strategies have been implemented.
        RandomSample; BlockedSample use self.cv parameter. this parameter specifies the number of cross validations
        all the other strategies also use the self.val_split parameter. this parameter specifies which part of the data set
        is used for training ( for all cv folds)
        the parameter self.method decides which of the strategies should be used

        the function returns a list of lists containing the different cv folds
        """

        num_samples = len(list(self.inputs.values())[0])

        if self.method == SplitMethod.RandomSample:
            self.train_splits = [i for i in range(self.cv)]
            block_length = int(num_samples / self.cv)
            list_train_test = []
            list_with_names = [i for i in range(len(list(self.inputs.values())[0]))]
            for _ in range(self.cv):
                subset_list = random.sample(list_with_names, block_length)
                list_with_names = [x for x in list_with_names if x not in subset_list]
                filter_list = []
                for i in range(len(list(self.inputs.values())[0])):
                    if i in subset_list:
                        filter_list.append(True)
                    else:
                        filter_list.append(False)
                test_inputs = {key: value[filter_list] for key, value in self.inputs.items()}
                test_targets = {key: value[filter_list] for key, value in self.targets.items()}
                train_inputs = {key: value[list(map(operator.not_, filter_list))] for key, value in self.inputs.items()}
                train_targets = {key: value[list(map(operator.not_, filter_list))] for key, value in
                                 self.targets.items()}
                list_train_test.append([train_inputs, train_targets, test_inputs, test_targets])
            return list_train_test

        elif self.method == SplitMethod.BlockedSample:
            self.train_splits = [i for i in range(self.cv)]
            block_length = int(num_samples / self.cv)
            list_train_test = []
            for i in range(self.cv):
                filter_list = [False for _ in range(len(list(self.inputs.values())[0]))]
                filter_list[i * block_length:(i + 1) * block_length] = [True for _ in range(block_length)]
                test_inputs = {key: value[filter_list] for key, value in self.inputs.items()}
                test_targets = {key: value[filter_list] for key, value in self.targets.items()}
                train_inputs = {key: value[list(map(operator.not_, filter_list))] for key, value in self.inputs.items()}
                train_targets = {key: value[list(map(operator.not_, filter_list))] for key, value in
                                 self.targets.items()}
                list_train_test.append([train_inputs, train_targets, test_inputs, test_targets])
            return list_train_test

        elif self.method == SplitMethod.RollingOriginUpdate:
            self.train_splits = [0]
            split = self._get_split_index(num_samples=num_samples)
            train_inputs, train_targets = self._get_train_set(split=split)
            # now get test sets
            test_inputs_complete, test_tagets_complete = self._get_test_set(split=split)
            num_samples_val = len(list(test_inputs_complete.values())[0])
            val_block_length = int(num_samples_val / self.cv)
            list_train_test = []
            for i in range(self.cv):
                filter_list = [False for _ in range(num_samples_val)]
                filter_list[i * val_block_length:(i + 1) * val_block_length] = [True for _ in range(val_block_length)]
                test_inputs = {key: value[filter_list] for key, value in test_inputs_complete.items()}
                test_targets = {key: value[filter_list] for key, value in test_tagets_complete.items()}
                list_train_test.append([train_inputs, train_targets, test_inputs, test_targets])
            return list_train_test

        elif self.method == SplitMethod.RollingOriginRetrain:
            self.train_splits = [i for i in range(self.cv)]
            split = self._get_split_index(num_samples=num_samples)
            test_inputs_complete, test_tagets_complete = self._get_test_set(split=split)
            num_samples_val = len(list(test_inputs_complete.values())[0])
            val_block_length = int(num_samples_val / self.cv)
            list_train_test = []
            for i in range(self.cv):
                # get test inputs / targets
                filter_list_test = [False for _ in range(num_samples_val)]
                filter_list_test[i * val_block_length:(i + 1) * val_block_length] = [True for _ in
                                                                                     range(val_block_length)]
                test_inputs = {key: value[filter_list_test] for key, value in test_inputs_complete.items()}
                test_targets = {key: value[filter_list_test] for key, value in test_tagets_complete.items()}
                # get train input/ targets
                filter_list_train = [False for _ in range(len(list(self.inputs.values())[0]))]
                filter_list_train[:(split + (i * val_block_length))] = [True for _ in
                                                                        range((split + (i * val_block_length)))]
                train_inputs = {key: value[filter_list_train] for key, value in self.inputs.items()}
                train_targets = {key: value[filter_list_train] for key, value in self.targets.items()}
                list_train_test.append([train_inputs, train_targets, test_inputs, test_targets])
            return list_train_test

        elif self.method == SplitMethod.RollingWindow:
            self.train_splits = [i for i in range(self.cv)]
            split = self._get_split_index(num_samples=num_samples)
            test_inputs_complete, test_tagets_complete = self._get_test_set(split=split)
            num_samples_val = len(list(test_inputs_complete.values())[0])
            val_block_length = int(num_samples_val / self.cv)
            list_train_test = []
            for i in range(self.cv):
                # get test inputs / targets
                filter_list_test = [False for _ in range(num_samples_val)]
                filter_list_test[i * val_block_length:(i + 1) * val_block_length] = [True for _ in
                                                                                     range(val_block_length)]
                test_inputs = {key: value[filter_list_test] for key, value in test_inputs_complete.items()}
                test_targets = {key: value[filter_list_test] for key, value in test_tagets_complete.items()}
                # get train input/ targets
                filter_list_train = [False for _ in range(len(list(self.inputs.values())[0]))]
                filter_list_train[(i * val_block_length):(split + (i * val_block_length))] = [True for _ in
                                                                                              range((split))]
                train_inputs = {key: value[filter_list_train] for key, value in self.inputs.items()}
                train_targets = {key: value[filter_list_train] for key, value in self.targets.items()}
                list_train_test.append([train_inputs, train_targets, test_inputs, test_targets])
            return list_train_test

    def _get_train_set(self, split):
        train_inputs = {key: value[:split] for key, value in self.inputs.items()}
        train_targets = {key: value[:split] for key, value in self.targets.items()}

        return train_inputs, train_targets

    def _get_test_set(self, split):
        test_inputs = {key: value[split:] for key, value in self.inputs.items()}
        test_tagets = {key: value[split:] for key, value in self.targets.items()}

        return test_inputs, test_tagets

    def _get_split_index(self, num_samples):
        return int(num_samples - (num_samples * self.val_split))


class RayTuneWrapper(BaseWrapper):
    """
     A wrappers class for ray tune. Should only be used internal by the pipeline itself.
     :param estimator: The estimator whose hyperparameters should be optimized
     :type estimator: pywatts.core.base.BaseWrapper, pywatts.core.base.BaseEstimator, pywatts.core.base.BaseTransformer,
         pywatts.core.pipeline.Pipeline
     :param ray_tune_kwargs: Keyword arguments for ray tune
     :type ray_tune_kwargs: dict
     :param ray_init_kwargs: Keyword arguments for initializing ray
     :type ray_init_kwargs: dict, optional
     :param val_split: Defines the validation split for
         SplitMethod.RollingOriginUpdate,
         SplitMethod.RollingOriginRetrain, and
         SplitMethod.RollingWindow
     :type val_split: float, optional
     :param split_method: Defines the method for splitting the data into a training an a validation set.
     :type split_method: SplitMethod, optional
     :param cv: Defines the number of cross validation splits.
     :type cv: int, optional
     :param loss_metric: Specifies the loss metric for automated optimal weight estimation. Also, a custom function
         can be defined with the inputs 'p' and 't', where 'p' is the prediction and 't' the true value, and the
         function returns a scalar.
     :type loss_metric: Callable, LossMetric, optional
     :param k_best: Defines if estimators of the k-best configurations should be fitted. If none, only an estimator
         for the best configuration is fitted.
     :type k_best: int, optional
     :param assess_fn: Defines a custom assess function, must report the score as parameter "loss" to ray tune, i.e.
         tune.report(loss=score).
     :type assess_fn: Callable, optional
     :param refit_only: During refit, the estimators are re-fitted but not re-tuned
     :type refit_only: bool, optional
     """

    def __init__(self, estimator: Union[BaseWrapper, BaseEstimator, BaseTransformer, Pipeline],
                 replace_weather: bool,
                 ray_tune_kwargs: dict, ray_init_kwargs: dict = None,
                 val_split: float = 0.0, split_method: SplitMethod = SplitMethod.RollingOriginUpdate, cv: int = 1,
                 loss_metric: Union[Callable, LossMetric] = LossMetric.RMSE,
                 k_best: int = 1, assess_fn: Callable = None, refit_only: bool = False,
                 name: str = "RayTuneWrapper"):
        super().__init__(name)

        if ray_init_kwargs is None:
            ray_init_kwargs = {}

        self.estimator = estimator
        self.ray_init_kwargs = ray_init_kwargs
        self.ray_tune_kwargs = ray_tune_kwargs
        self.val_split = val_split
        self.split_method = split_method
        self.cv = cv
        self.loss_metric = loss_metric
        self.assess_fn = assess_fn
        self.k_best = k_best
        self.refit_only = refit_only

        self.replace_weather = replace_weather

        self.analysis_ = None
        self.estimators_k_best_ = None
        self.k_best_config_ = None
        self.summary_config = None

        self.is_fitted = False

        if not ray.is_initialized:
            ray.init(**self.ray_init_kwargs)
        else:
            warnings.warn("Ray is already initialized, init_kwargs are not applied!")

    def get_params(self) -> Dict[str, object]:
        """
        Return the parameter of the ray tune module
        :return:
        """
        return {
            "estimator": self.estimator,
            "ray_tune_kwargs": self.ray_tune_kwargs,
            "ray_init_kwargs": self.ray_init_kwargs,
            "val_split": self.val_split,
            "split_method": self.split_method,
            "cv": self.cv,
            "loss_metric": self.loss_metric,
            "k_best": self.k_best,
            "assess_fn": self.assess_fn,
            "refit_only": self.refit_only,
            "replace_weather": self.replace_weather
        }

    def set_params(self, estimator: Union[BaseWrapper, BaseEstimator, BaseTransformer, Pipeline] = None,
                   ray_tune_kwargs: dict = None, ray_init_kwargs: dict = None,
                   val_split: Union[int, float] = None, split_method: SplitMethod = None, cv: int = None,
                   loss_metric: Union[Callable, LossMetric] = LossMetric.RMSE, replace_weather: bool = None,
                   k_best: int = None, assess_fn: Callable = None, refit_only: bool = None):
        """
        Set the parameter of the internal sklearn module
        :param estimator: The estimator whose hyperparameters should be optimized
        :type estimator: pywatts.core.base.BaseWrapper, pywatts.core.base.BaseEstimator, pywatts.core.base.BaseTransformer,
            pywatts.core.pipeline.Pipeline
        :param ray_tune_kwargs: Keyword arguments for ray tune
        :type ray_tune_kwargs: dict
        :param ray_init_kwargs: Keyword arguments for initializing ray
        :type ray_init_kwargs: dict, optional
        :param val_split: Defines the validation split for
            SplitMethod.RollingOriginUpdate,
            SplitMethod.RollingOriginRetrain, and
            SplitMethod.RollingWindow
        :type val_split: float, optional
        :param split_method: Defines the method for splitting the data into a training an a validation set.
        :type split_method: SplitMethod, optional
        :param cv: Defines the number of cross validation splits.
        :type cv: int, optional
        :param loss_metric: Specifies the loss metric for automated optimal weight estimation. Also, a custom function
            can be defined with the inputs 'p' and 't', where 'p' is the prediction and 't' the true value, and the
            function returns a scalar.
        :type loss_metric: Callable, LossMetric, optional
        :param k_best: Defines if estimators of the k-best configurations should be fitted. If none, only an estimator
            for the best configuration is fitted.
        :type k_best: int, optional
        :param assess_fn: Defines a custom assess function, must report the score as parameter "loss" to ray tune, i.e.
            tune.report(loss=score).
        :type assess_fn: Callable, optional
        :param refit_only: During refit, the estimators are re-fitted but not re-tuned
        :type refit_only: bool, optional
        """
        if estimator is not None:
            self.estimator = estimator
        if ray_tune_kwargs is not None:
            self.ray_tune_kwargs = ray_tune_kwargs
        if ray_init_kwargs is not None:
            self.ray_init_kwargs = ray_init_kwargs
            if ray.is_initialized:
                warnings.warn("Ray is already initialized, different init_kwargs will have no effect!")
            ray.init(**{**self.ray_init_kwargs, **{"ignore_reinit_error": True}})
        if val_split is not None:
            self.val_split = val_split
        if loss_metric is not None:
            self.loss_metric = loss_metric
        if k_best is not None:
            self.k_best = k_best
        if assess_fn is not None:
            self.assess_fn = assess_fn
        if refit_only is not None:
            self.refit_only = refit_only
        if replace_weather is not None:
            self.replace_weather = replace_weather

    def refit(self, **kwargs):
        """
        Refit the ray tune module
        :param inputs: input data
        :param targets: target data
        """
        print(f"{self.name}: start fit with refit_only={self.refit_only}")
        inputs, targets = split_kwargs(kwargs)

        if not self.refit_only:
            self.estimators_k_best_ = None
            # assess config trials on validation data to find best config
            self._assess(inputs=inputs, targets=targets)

        # fit estimator with the best config on train + val data
        self._fit_best_config(inputs, targets)

        self.is_fitted = True
        print(f"{self.name}: finish fit with refit_only={self.refit_only}")

    def fit(self, **kwargs):
        """
        Fit the ray tune module
        :param inputs: input data
        :param targets: target data
        """
        inputs, targets = split_kwargs(kwargs)

        # assess config trials on validation data to find best config
        self._assess(inputs=inputs, targets=targets)

        # fit estimator with the best config on train + val data
        self._fit_best_config(inputs, targets)

        self.is_fitted = True

    def _assess(self, inputs, targets):

        if self.assess_fn is None:
            assess_fn = self._assess_fn
        elif isinstance(self.assess_fn, Callable):
            assess_fn = self.assess_fn
        else:
            raise WrongParameterException(
                f"The given assess_fn {self.assess_fn} is not a callable.",
                "Make sure to pass a callable that reports the score as 'loss' to ray tune, "
                "i.e. tune.report(loss=score)",
                f"{self.name} of class {self.__class__.__name__}.")

        self.analysis_ = None
        estimator = deepcopy(self.estimator)
        splitter = Splitter(inputs=inputs, targets=targets, method=self.split_method, cv=self.cv,
                            val_split=self.val_split)
        ray_tune_kwargs = deepcopy(self.ray_tune_kwargs)  # otherwise, stoppers are not reset after tuning
        tuner = Tuner(tune.with_parameters(assess_fn,
                                           estimator=estimator,
                                           splitter=splitter,
                                           fit_fn=self._fit, score_fn=self._score),
                      **ray_tune_kwargs)
        self.analysis_ = tuner.fit()._experiment_analysis

        # release memory
        del tuner, estimator, splitter, ray_tune_kwargs

    def _fit_best_config(self, inputs, targets):

        k_best_config_ = [
            params[1] for params in
            sorted([(value["loss"], value["config"]) for value in self.analysis_.results.values()
                    if "loss" in list(value.keys())],
                   key=lambda x: x[0]
                   )[:self.k_best]
        ]

        # todo: parallelize
        # fit multiple estimators with the k-best hyperparameter configurations
        self.estimators_k_best_ = []
        for config in k_best_config_:  # fit 1st, 2nd, 3rd, ..., k-th best configuration
            estimator = self._fit(estimator=deepcopy(self.estimator),  # create a copy
                                  config=config,
                                  inputs_train=inputs, targets_train=targets)
            self.estimators_k_best_.append(estimator)

        for i, estimator in enumerate(self.estimators_k_best_):
            if isinstance(estimator, Pipeline):
                self._clear_buffer(estimator)

        # stringify config
        self.k_best_config_ = deepcopy(k_best_config_)
        for i, config in enumerate(self.k_best_config_):
            self.k_best_config_[i] = self._stringify_config(config)

    def _stringify_config(self, config):
        for k, v in config.items():
            if isinstance(v, dict):
                self._stringify_config(v)
            else:
                try:
                    v = json.dumps(v)
                except (TypeError, OverflowError):
                    v = str(v) if not hasattr(v, "name") else getattr(v, "name")
                config.update({k: v})
        return config

    def _recursive_search(self, step, config):
        if "module" in vars(step):
            step.refit_conditions = []
            step.callbacks = []
            step.current_run_setting.summary_formatter = None  # do not create summaries
            if "id_to_step" in vars(step.module):
                for pipeline_step in step.module.id_to_step.values():
                    pipeline_step.refit_conditions = []
                    pipeline_step.callbacks = []
                    pipeline_step.current_run_setting.summary_formatter = None  # do not create summaries
                    self._recursive_search(pipeline_step, config)
            if step.name in list(config.keys()):
                if "module" in list(config[step.name].keys()):
                    step.module.module = config[step.name]["module"]  # replace the module
                    step.module.set_params(
                        **{key: value for key, value in config[step.name].items() if key != "module"})
                else:
                    step.module.set_params(**config[step.name])

    def _fit(self, estimator, config, inputs_train, targets_train):
        if isinstance(estimator, Pipeline):
            estimator.current_run_setting.summary_formatter = None  # do not create summaries
            for step in estimator.id_to_step.values():
                self._recursive_search(step, config)
            estimator._transform(inputs_train)  # todo: implement a fit function for the Pipeline
            # clear buffers, otherwise, transform does not work after refitting
            self._clear_buffer(estimator)

            if self.replace_weather:
                # True for online mode, otherwise, online fitted models will forecast with measurement data
                # False for training otherwise, models will overfit on forecast data
                for step in estimator.id_to_step.values():
                    if "forecast_day" in step.name:
                        step.input_steps["weather_features"].buffer_element = "weather_features_forecast"
                        break

        elif isinstance(estimator, (BaseWrapper, BaseEstimator, BaseTransformer)):
            estimator.set_params(**config)
            estimator.fit(**{**inputs_train, **targets_train})
        else:
            raise WrongParameterException(
                f"The given estimator {estimator} is neither an pyWATTS pipeline nor a pyWATTS estimator ",
                "Make sure to pass one of the following estimators: "
                "BaseWrapper, BaseEstimator, BaseTransformer, Pipeline",
                f"class 'RayTuneWrapper'.")

        return estimator

    def transform(self, **kwargs) -> Union[dict, xr.DataArray]:
        """
        Transforms a dataset or predicts the result with the wrapped estimator or pipeline
        :param inputs: the input dataset
        :return: the transformed output
        """
        result = {}
        for i, estimator in enumerate(self.estimators_k_best_):
            preds = estimator.transform(**kwargs)
            if type(preds) != dict:
                preds = dict({"forecast": preds})
            result.update({f"{self.name}_rank_{i + 1}": value for value in preds.values()})
        return result

    def _score(self, p, t):
        p = p.values
        t = t.values
        if self.loss_metric == LossMetric.RMSE:
            return np.sqrt(np.mean((p - t) ** 2))
        elif self.loss_metric == LossMetric.MAE:
            return np.mean(np.abs((p - t)))
        elif isinstance(self.loss_metric, Callable):
            return self.loss_metric(p=p, t=t)
        else:
            raise WrongParameterException(
                "The specified loss metric is not implemented.",
                "Make sure to pass LossMetric.RMSE or LossMetric.MAE.",
                self.name
            )

    @staticmethod
    def _assess_fn(config, estimator, splitter, fit_fn, score_fn):
        """
        Assesses the hyperparameter configuration of a trial and reports the score to ray tune.
        :param config: The hyperparameter configuration for the trial.
        :type config: dict
        """
        scores = []

        for split_no, (inputs_train, targets_train, inputs_val, targets_val) in enumerate(splitter.get_splits()):
            if split_no in splitter.train_splits:
                estimator = fit_fn(estimator=estimator, config=config, inputs_train=inputs_train,
                                   targets_train=targets_train)

            y_hat = estimator.transform(**inputs_val)
            if not isinstance(y_hat, dict):
                y_hat = {'target': y_hat}

            time_index = _get_time_indexes(y_hat)[0]
            for (_, v_pred), (k_true, v_true) in zip(y_hat.items(), targets_val.items()):
                targets_val[k_true] = v_true.loc[v_pred[time_index][0]:v_pred[time_index][-1]]

            scores.append(np.mean([
                score_fn(pred, true) for pred, true in zip(y_hat.values(), targets_val.values())]))

        tune.report(loss=np.mean(scores))

    def _clear_buffer(self, pipeline):
        pipeline.current_run_setting.computation_mode = ComputationMode.Transform
        for (start_step, _) in pipeline.start_steps.values():
            start_step.current_buffer = {}
            start_step.result_buffer = {}
        for step in pipeline.id_to_step.values():
            step.current_buffer = {}
            step.result_buffer = {}
            step.set_run_setting(RunSetting(computation_mode=ComputationMode.Transform,
                                            summary_formatter=None,
                                            online_start=None,
                                            return_summary=False))
            for input_step in step.input_steps.values():
                input_step.current_buffer = {}
                input_step.result_buffer = {}
            if isinstance(step, PipelineStep):
                self._clear_buffer(step.module)
