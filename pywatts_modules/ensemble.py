import time
from typing import Union, List, Dict
from enum import IntEnum

import logging
import xarray as xr
import numpy as np

from scipy.optimize import least_squares

from pywatts.core.base import BaseEstimator
from pywatts.core.exceptions import WrongParameterException
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray
from pywatts.utils._split_kwargs import split_kwargs


logger = logging.getLogger(__name__)


class Ensemble(BaseEstimator):
    """
    Aggregation step to ensemble the given time series, ether by simple or weighted averaging.
    By default simple averaging is applied.
    """

    class LossMetric(IntEnum):
        """
        Enum which contains the different loss metrics of the ensemble module.
        """
        MSE = 1
        MAE = 2

    class OptimizationTimeout(object):
        def __init__(self, time_budget_s: int):
            self._time_budget_s = time_budget_s
            self.start = time.time()
            self.n_iter = 0

        def __call__(self, xk=None, convergence=None):
            elapsed_s = time.time() - self.start
            self.n_iter += 1
            if self._time_budget_s is None:
                return False
            elif elapsed_s > self._time_budget_s:
                print(f"Optimization stopping criterion reached, n_iter={self.n_iter}, elapsed_s={elapsed_s}")
                return True

    def __init__(self, weights: Union[str, list] = None, k_best: Union[str, int] = None,
                 inputs_ignored: List[str] = None, time_budget_s: int = None,
                 loss_metric: LossMetric = LossMetric.MSE, max_weight: float = 1.0, name: str = "Ensemble"):
        """ Initialize the ensemble step.
        :param weights: List of individual weights of the given forecasts for weighted averaging. Passing "autoErr"
        estimates the weights depending on the given loss values.
        :type weights: list, optional
        :param loss_metric: Specifies the loss metric for automated optimal weight estimation.
        :type loss_metric: LossMetric, optional
        :param k_best: Drop poor forecasts in the automated weight estimation. Passing "auto" drops poor forecasts based
        on the given loss values by applying the 1.5*IQR rule.
        :type k_best: str or int, optional
        :param max_weight: Defines the maximum weight of one forecast if the autoOpt strategy is chosen.
        :type max_weight: float, optional
        :param time_budget_s: Defines the time budget for the ensemble optimization if the autoOpt strategy is chosen.
        :type time_budget_s: int, optional

        example for two given forecasts
        weights = None, k_best = None              -> averaging
        weights = None, k_best = 'autoErr'         -> averaging k-best with k based on loss values
        weights = None, k_best = k                 -> averaging k-best with given k
        weights = [0.3,0.7], k_best = None         -> weighting based on given weights
        weights = [0.3,0.7], k_best = 'autoErr'    -> weighting based on given weights and k based on loss values
        weights = [0.3,0.7], k_best = k            -> weighting based on given weights and k
        weights = 'autoErr', k_best = None         -> weighting with weights based on loss values
        weights = 'autoErr', k_best = 'autoErr'    -> weighting k-best with weights and k based on loss values
        weights = 'autoErr', k_best = k            -> weighting k-best with weights based on loss values and given k
        """
        super().__init__(name)
        if inputs_ignored is None:
            inputs_ignored = []

        self.weights = weights
        self.k_best = k_best
        self.inputs_ignored = inputs_ignored
        self.max_weight = max_weight
        self.time_budget_s = time_budget_s
        self.loss_metric = loss_metric

        self.weights_ = None
        self.weights_autoOpt_ = None

        self.is_fitted = False

    def get_params(self) -> Dict[str, object]:
        """ Get parameters for the Ensemble object.
        :return: Parameters as dict object.
        :rtype: Dict[str, object]
        """
        return {
            "weights": self.weights,
            "k_best": self.k_best,
            "loss_metric": self.loss_metric,
            "max_weight": self.max_weight,
            "clustering": self.clustering,
            "inputs_ignored": self.inputs_ignored
        }

    def set_params(self, weights: Union[str, list] = None,
                   clustering: Union[bool, list] = None,
                   inputs_ignored: List[str] = None,
                   loss_metric: LossMetric = None,
                   max_weight: float = None,
                   k_best: Union[str, int] = None):
        """ Set or change Ensemble object parameters.
        :param weights: List of individual weights of the given forecasts for weighted averaging. Passing "autoErr"
        estimates the weights depending on the given loss values.
        :type weights: list, optional
        :param loss_metric: Specifies the loss metric for automated optimal weight estimation.
        :type loss_metric: LossMetric, optional
        :param k_best: Drop poor forecasts in the automated weight estimation. Passing "auto" drops poor forecasts based
        on the given loss values by applying the 1.5*IQR rule.
        :type k_best: str or int, optional
        :param max_weight: Defines the maximum weight of one forecast if the autoOpt strategy is chosen.
        :type max_weight: float, optional
        """
        if weights is not None:
            self.weights = weights
        if loss_metric is not None:
            self.loss_metric = loss_metric
        if k_best is not None:
            self.k_best = k_best
        if clustering is not None:
            self.clustering = clustering
        if max_weight is not None:
            self.max_weight = max_weight
        if inputs_ignored is not None:
            self.inputs_ignored = inputs_ignored

    def fit(self, **kwargs):
        """
        Fit the ensemble module.
        :param x: input data
        :param y: target data
        """
        if len(self.inputs_ignored) > 0:
            kwargs = {key: value for key, value in kwargs.items() if key not in self.inputs_ignored}

        forecasts, targets = split_kwargs(kwargs)

        if self.weights == 'autoErr' or self.weights == 'autoErr_after_autoOpt' or self.k_best is not None:
            # determine weights depending on in-sample loss
            loss_values = self._calculate_loss_forecasts(p_n=forecasts, t=targets)
            # drop forecasts depending on in-sample loss
            index_loss_dropped = self._drop_forecasts(loss=loss_values)

            # overwrite weights based on given loss values and set weights of dropped forecasts to zero
            if self.weights == "autoErr":  # weighted averaging depending on estimated weights
                self.weights_ = self._normalize_weights(
                    [0 if i in index_loss_dropped
                     else 1 / value for i, value in enumerate(loss_values)])
            elif self.weights == "autoErr_after_autoOpt":
                wgt = self._normalize_weights(
                    [0 if i in index_loss_dropped
                     else 1 / value for i, value in enumerate(loss_values)])
                if self.weights_autoOpt_ is not None:
                    self.weights_ = self._normalize_weights(list(np.array(self.weights_autoOpt_) * np.array(wgt)))
                else:
                    self.weights_ = wgt
            elif self.weights is None:  # averaging
                self.weights_ = self._normalize_weights(
                    [0 if i in index_loss_dropped
                     else 1 for i, value in enumerate(loss_values)])
            else:  # weighted averaging depending on given weights
                self.weights_ = self._normalize_weights(
                    [0 if i in index_loss_dropped
                     else weight for i, (value, weight) in enumerate(zip(loss_values, self.weights))])

        # eff before weights
        elif self.weights == 'autoOpt':
            # efficiency factor and weight optimization are separated
            predictions = [forecast.values for forecast in forecasts.values()]
            if self.max_weight != 1.0:
                raise UserWarning(f"Parameter max_weight={self.max_weight} is ignored for the least squares optimization!")

            bounds = [[0.0 for _ in range(len(forecasts))]] + [[1.0 for _ in range(len(forecasts))]]
            x0 = [1 / len(forecasts)] * len(forecasts)
            kwargs = {"p_n": predictions, "t": targets}
            result = least_squares(fun=self._assess_weighted_average, x0=x0, bounds=bounds, kwargs=kwargs,
                                   ftol=1e-12, xtol=1e-12, gtol=1e-12)
            self.weights_ = self._normalize_weights(list(result.x))  # repair weights, sum weights = 1
            self.weights_autoOpt_ = self.weights_

        elif self.weights is not None:
            # use given weights
            if isinstance(self.weights, list):
                if len(self.weights) is not len(forecasts):
                    raise WrongParameterException(
                        "The number of the given weights does not match the number of given forecasts.",
                        f"Make sure to pass {len(forecasts)} weights.",
                        self.name
                    )
            self.weights_ = self._normalize_weights(self.weights)

        self.is_fitted = True

    def transform(self, **kwargs) -> xr.DataArray:
        """
        Ensemble the given time series by simple or weighted averaging.
        :return: Xarray dataset aggregated by simple or weighted averaging.
        :rtype: xr.DataArray
        """

        if len(self.inputs_ignored) > 0:
            kwargs = {key: value for key, value in kwargs.items() if key not in self.inputs_ignored}

        forecasts, _ = split_kwargs(kwargs)

        list_of_series = []
        list_of_indexes = []
        for series in forecasts.values():
            list_of_indexes.append(series.indexes)
            list_of_series.append(series.data)

        if not all(all(index) == all(list_of_indexes[0]) for index in list_of_indexes):
            raise ValueError("The indexes of the given time series for averaging do not match")

        result = np.average(list_of_series, axis=0, weights=self.weights_)

        return numpy_to_xarray(result, series)

    def _assess_weighted_average(self, x, p_n: list, t: dict) -> np.float:
        """
        Assesses the weights of a trial using the weighted average and returns the score.
        :param variables: The variables to be optimized.
        :type variables: list
        :param ps: The forecasts that should be ensembled.
        :type ps: list
        :param ts: The realized values.
        :type ts: dict
        :return score
        :rtype: np.float
        """

        t = np.array([t_.values for t_ in t.values()])
        p = np.average(p_n, axis=0, weights=x)

        return self._calculate_error(p=p, t=t)

    def _calculate_error(self, p, t):
        if self.loss_metric == self.LossMetric.MSE:
            return float(np.mean((p - t) ** 2))
        elif self.loss_metric == self.LossMetric.MAE:
            return float(np.mean(np.abs((p - t))).ravel())
        else:
            raise WrongParameterException(
                "The specified loss metric is not implemented.",
                "Make sure to pass LossMetric.MSE or LossMetric.MAE.",
                self.name)

    def _calculate_loss_forecasts(self, p_n, t) -> list:
        """
        Calculates the loss of the given forecasts.
        :param ps: The predictions.
        :type ps: dict
        :param ts: The realized values.
        :type ts: dict
        :return The loss values of the given forecasts
        :rtype: list
        """

        t = np.array([t_.values for t_ in t.values()])
        loss_values = []
        for p in p_n.values():
            loss_values.append(self._calculate_error(p=p.values, t=t))

        return loss_values

    def _drop_forecasts(self, loss: list) -> list:
        """
        Drops poor performing forecasts from the pool.
        :param loss: The loss of the forecasts.
        :type loss: list
        :return The indexes of forecasts that should be dropped.
        :rtype: list
        """

        index_loss_dropped = []
        if self.k_best is not None:
            # Do not sort the loss_values! Otherwise, the weights do not match the given forecasts.
            if self.k_best == "autoErr":
                q75, q25 = np.percentile(loss, [75, 25])
                iqr = q75 - q25
                upper_bound = q75 + 1.5 * iqr  # only check for outliers with high loss
                index_loss_dropped = [i for i, value in enumerate(loss) if not (value <= upper_bound)]
            elif self.k_best > len(loss):
                raise WrongParameterException(
                    "The given k is greater than the number of the given loss values.",
                    f"Make sure to define k <= {len(loss)}.",
                    self.name
                )
            else:
                index_loss_dropped = list(np.argpartition(np.array(loss), self.k_best))[self.k_best:]

        return index_loss_dropped

    @staticmethod
    def _normalize_weights(weights: list) -> list:
        """
        Normalizes the weights in the range [0,1]
        :param weights: The weights to be normalized.
        :type weights: list
        :return The normalized weights.
        :rtype: list
        """
        return [weight / sum(weights) for weight in weights]
