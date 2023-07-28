import warnings
import pandas as pd
from typing import Union

from pywatts.core.summary_formatter import SummaryJSON

from autopv.sub_pipelines.ensembling import create_pipeline_ensemble
from autopv.sub_pipelines.preprocessing import create_pipeline_preprocessing
from autopv.sub_pipelines.model_pool import create_pipeline_model_engineering
from autopv.sub_pipelines.postprocessing import create_pipeline_postprocessing
from autopv.sub_pipelines.utils import create_modules, save_modules, load_modules
from autopv.config import ModelPool, MeasurementUnit, AdaptionConfiguration, get_plant_kwp_synthetic_selection


class AutoPV:
    def __init__(self, target_name: str, target_kWp: float,
                 model_pool: ModelPool, measurement_unit: MeasurementUnit,
                 latitude: float = None, longitude: float = None, altitude: float = 116, samples_per_hour: int = None,
                 adaption_config: AdaptionConfiguration = AdaptionConfiguration.AIB, nearby_plants_kWp: dict = None,
                 C: Union[pd.Timedelta, None] = pd.Timedelta("28d"), K: Union[pd.Timedelta, None] = pd.Timedelta("28d")):

        if model_pool == ModelPool.nearby:
            if nearby_plants_kWp is None:
                raise SyntaxError("Using the nearby plants model pool requires defining the names and peak power ratings of "
                                  "the nearby PV plants with the argument nearby_plants_kWp. The structure is as follows\n"
                                  "{'pv_name_0': 100, 'pv_name_1': 200, 'pv_name_2': 300}")
            self.ensemble_plants_kWp = nearby_plants_kWp
        elif model_pool == ModelPool.default:
            if latitude is None or longitude is None:
                raise SyntaxError("The latitude and longitude must be given!")
            if nearby_plants_kWp is not None:
                warnings.warn("Using the default model pool does not require the names and peak power ratings of nearby PV "
                              "plants. The argument nearby_plants_kWp will be ignored!")
            self.ensemble_plants_kWp = get_plant_kwp_synthetic_selection()

        if adaption_config != AdaptionConfiguration.none and (C is None or K is None):
            raise SyntaxError("The adaption cycle C and the number of considered samples K must be given!")

        if measurement_unit == MeasurementUnit.kWh and samples_per_hour is None:
            raise SyntaxError("The measurement unit is kWh. Transforming the energy metering data into a mean power "
                              "generation time series requires the number of samples per hour!")

        self.target_name = target_name
        self.target_kWp = target_kWp
        self.model_pool = model_pool
        self.measurement_unit = measurement_unit
        self.samples_per_hour = samples_per_hour
        self.adaption_config = adaption_config
        self.C = C
        self.K = K
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude

        self.weights_ = None

        # Create modules
        self._modules = create_modules(ensemble_plants_kWp=self.ensemble_plants_kWp, model_pool=self.model_pool,
                                       target_name=target_name, latitude=latitude, longitude=longitude, altitude=altitude)

        # Create pipelines
        self._pipeline_preprocessing = create_pipeline_preprocessing(plant_kWp_train=self.ensemble_plants_kWp,
                                                                     plant_kWp_test={self.target_name: self.target_kWp},
                                                                     plant_test=self.target_name,
                                                                     model_pool=self.model_pool,
                                                                     modules=self._modules,
                                                                     measurement_unit=self.measurement_unit,
                                                                     samples_per_hour=self.samples_per_hour)
        self._pipeline_models = create_pipeline_model_engineering(plants_kWp=self.ensemble_plants_kWp,
                                                                  plant_test=self.target_name,
                                                                  model_pool=self.model_pool,
                                                                  pipeline_preprocessing=self._pipeline_preprocessing,
                                                                  modules=self._modules)
        self._pipeline_ensemble = create_pipeline_ensemble(plant_test=self.target_name,
                                                           pipeline_preprocessing=self._pipeline_preprocessing,
                                                           modules=self._modules, model_pool=self.model_pool, K=self.K,
                                                           C=self.C, adaption_configuration=self.adaption_config)
        self._pipeline_postprocessing = create_pipeline_postprocessing(plants_kWp_test={self.target_name: self.target_kWp},
                                                                       plant_test=self.target_name,
                                                                       pipeline_preprocessing=self._pipeline_preprocessing,
                                                                       pipeline_ensemble=self._pipeline_ensemble,
                                                                       measurement_unit=self.measurement_unit,
                                                                       samples_per_hour=self.samples_per_hour)

    def fit(self, data: pd.DataFrame, save_models: bool = False, model_directory: str = "./models"):

        res_models = self._pipeline_models.train(data=data, summary_formatter=SummaryJSON(), summary=False)

        res_ensemble = {}
        if self.adaption_config == AdaptionConfiguration.none:
            res_ensemble = self._pipeline_ensemble.train(data=data, summary_formatter=SummaryJSON(), summary=False)
            self.weights_ = self._modules["ensemble"].weights_

        if save_models:
            save_modules(pipeline_modules=self._modules, dry=model_directory)

        return {**res_models, **res_ensemble}

    def predict(self, data: pd.DataFrame, online_start: pd.Timestamp = None,
                load_models: bool = False, model_directory: str = "./models"):

        if self.C is None and online_start is not None:
            warnings.warn("The adaption cycle C is None. The argument online_start will be ignored!")
        if self.C is not None and online_start is None:
            raise SyntaxError("The online simulation requires defining the timestamp where the online processing begins!")

        # Load modules
        if load_models:
            self._modules = load_modules(pipeline_modules=self._modules, dry=model_directory)
            self._pipeline_ensemble = create_pipeline_ensemble(plant_test=self.target_name,
                                                               pipeline_preprocessing=self._pipeline_preprocessing,
                                                               modules=self._modules, model_pool=self.model_pool, K=self.K,
                                                               C=self.C, adaption_configuration=self.adaption_config)
        result = self._pipeline_postprocessing.test(data=data, summary_formatter=SummaryJSON(), summary=False,
                                                    online_start=online_start)
        self.weights_ = self._modules["ensemble"].weights_

        return result
