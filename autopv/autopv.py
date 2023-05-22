import warnings
from typing import Union

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from pywatts.core.computation_mode import ComputationMode
from pywatts.core.summary_formatter import SummaryJSON

from autopv.sub_pipelines.ensembling import create_pipeline_ensemble
from autopv.sub_pipelines.preprocessing import create_pipeline_preprocessing
from autopv.sub_pipelines.model_pool import create_pipeline_model_engineering
from autopv.sub_pipelines.postprocessing import create_pipeline_postprocessing
from autopv.sub_pipelines.utils import create_modules, save_modules, load_modules
from autopv.config import ModelPool, MeasurementUnit, AdaptionConfiguration, get_plant_kwp_synthetic_selection
from pywatts.modules import SKLearnWrapper, CalendarExtraction, CalendarFeature


class AutoPV:
    def __init__(self, model_pool: ModelPool, measurement_unit: MeasurementUnit, samples_per_hour: int = None,
                 adaption_config: AdaptionConfiguration = AdaptionConfiguration.AIB, nearby_plants_kWp: dict = None,
                 C: Union[pd.Timedelta, None] = pd.Timedelta("28d"), K: Union[pd.Timedelta, None] = pd.Timedelta("28d")):

        if model_pool == ModelPool.nearby:
            if nearby_plants_kWp is None:
                raise SyntaxError("Using the nearby plants model pool requires defining the names and peak power ratings of "
                                  "the nearby PV plants with the argument nearby_plants_kWp. The structure is as follows\n"
                                  "{'pv_name_0': 100, 'pv_name_1': 200, 'pv_name_2': 300}")
            self.ensemble_plants_kWp = nearby_plants_kWp
        elif model_pool == ModelPool.default:
            if nearby_plants_kWp is not None:
                warnings.warn("Using the default model pool does not require the names and peak power ratings of nearby PV "
                              "plants. The argument nearby_plants_kWp will be ignored!")
            self.ensemble_plants_kWp = get_plant_kwp_synthetic_selection()

        if C is None and K is not None:
            warnings.warn("The adaption cycle C is None. The argument K will be ignored!")
        if C is not None and K is None:
            warnings.warn("The number of considered samples is None. The argument K will be ignored!")

        if measurement_unit == MeasurementUnit.kWh and samples_per_hour is None:
            raise SyntaxError("The measurement unit is kWh. Transforming the energy metering data into a mean power "
                              "generation time series requires the number of samples per hour!")

        self.model_pool = model_pool
        self.measurement_unit = measurement_unit
        self.samples_per_hour = samples_per_hour
        self.adaption_config = adaption_config
        self.C = C
        self.K = K

        self._pipeline_preprocessing = None
        self._pipeline_models = None
        self._pipeline_ensemble = None
        self._modules = {
            "scaler_radiation": SKLearnWrapper(module=MinMaxScaler(), name="scaled_radiation"),
            "scaler_temperature": SKLearnWrapper(module=MinMaxScaler(), name="scaled_temperature"),
            "features_polynomial": SKLearnWrapper(module=PolynomialFeatures(degree=3, include_bias=False),
                                                  name="weather_features"),
            "features_calendar": CalendarExtraction(continent="Europe", country="Germany", name="calendar_features",
                                                    features=[CalendarFeature.month_cos, CalendarFeature.month_sine,
                                                              CalendarFeature.minute_of_day_cos,
                                                              CalendarFeature.minute_of_day_sine])
        }

    def fit(self, data: pd.DataFrame, target_name: str, target_kWp: float,
            save_models: bool = False, model_directory: str = "./models"):

        # Create modules
        for plant in self.ensemble_plants_kWp.keys():
            self._modules[plant] = create_modules(plant=plant, model_pool=self.model_pool)

        # Create pre-processing pipeline
        self._pipeline_preprocessing = create_pipeline_preprocessing(plant_kWp_train=self.ensemble_plants_kWp,
                                                                     plant_kWp_test={target_name: target_kWp},
                                                                     plant_test=target_name,
                                                                     modules=self._modules,
                                                                     measurement_unit=self.measurement_unit,
                                                                     samples_per_hour=self.samples_per_hour)

        # Create training pipeline and couple with pre-processing pipeline
        pipeline_models = create_pipeline_model_engineering(plants_kWp=self.ensemble_plants_kWp,
                                                            plant_test=target_name,
                                                            model_pool=self.model_pool,
                                                            preprocessing=self._pipeline_preprocessing,
                                                            modules=self._modules)
        result = pipeline_models.train(data=data, summary_formatter=SummaryJSON())

        # Create ensemble pipeline
        self._pipeline_ensemble = create_pipeline_ensemble(plant_test=target_name,
                                                           pipeline_preprocessing=self._pipeline_preprocessing,
                                                           modules=self._modules, model_pool=self.model_pool, K=self.K,
                                                           C=self.C, computation_mode=ComputationMode.Refit,
                                                           adaption_configuration=self.adaption_config)
        if self.C is None:
            result = self._pipeline_ensemble.train(data=data, summary_formatter=SummaryJSON())

        if save_models:
            save_modules(pipeline_modules=self._modules, dry=model_directory)

        return result

    def predict(self, data: pd.DataFrame, target_name: str, target_kWp: float, online_start: pd.Timestamp = None,
                load_models: bool = False, model_directory: str = "./models"):

        if self.C is None and online_start is not None:
            warnings.warn("The adaption cycle C is None. The argument online_start will be ignored!")
        if self.C is not None and online_start is None:
            raise SyntaxError("The online simulation requires defining the timestamp where the online processing begins!")

        # Load modules
        if load_models:
            for plant in self.ensemble_plants_kWp.keys():
                self._modules[plant] = create_modules(plant=plant, model_pool=self.model_pool)
            self._modules = load_modules(pipeline_modules=self._modules, dry=model_directory)

            # Create ensemble pipeline
            self._pipeline_ensemble = create_pipeline_ensemble(plant_test=target_name,
                                                               pipeline_preprocessing=self._pipeline_preprocessing,
                                                               modules=self._modules, model_pool=self.model_pool, K=self.K,
                                                               C=self.C, computation_mode=ComputationMode.Refit,
                                                               adaption_configuration=self.adaption_config)

        # Create post-processing pipeline
        pipeline_postprocessing = create_pipeline_postprocessing(plants_kWp_test={target_name: target_kWp},
                                                                 plant_test=target_name,
                                                                 pipeline_preprocessing=self._pipeline_preprocessing,
                                                                 pipeline_ensemble=self._pipeline_ensemble,
                                                                 measurement_unit=self.measurement_unit,
                                                                 samples_per_hour=self.samples_per_hour)
        result = pipeline_postprocessing.test(data=data, summary_formatter=SummaryJSON(), online_start=online_start)

        return result
