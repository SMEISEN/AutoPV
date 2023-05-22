from typing import Dict

import pvlib
import xarray as xr
from pvlib import tools
from pvlib.irradiance import disc
from pywatts.core.exceptions.invalid_input_exception import InvalidInputException
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray

from pywatts.core.base import BaseTransformer


class PVLibWrapper(BaseTransformer):
    def __init__(self, *, latitude, longitude, surface_azimuth, surface_tilt,
                 altitude=None,
                 pv_module=None,
                 inverter=None,
                 temperature_model_params=None,
                 name: str = "PVLibWrapper"):
        super().__init__(name)
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.surface_azimuth = surface_azimuth
        self.surface_tilt = surface_tilt
        if pv_module is None:
            sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
            self.pv_module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
        else:
            self.pv_module = pv_module
        if inverter is None:
            sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
            self.inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
        else:
            self.inverter = inverter
        if temperature_model_params is None:
            self.temperature_model_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm'][
                'open_rack_glass_glass']
        else:
            self.temperature_model_params = temperature_model_params

    def get_params(self) -> Dict[str, object]:
        """ Get parameters for the Ensemble object.
        :return: Parameters as dict object.
        :rtype: Dict[str, object]
        """
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "surface_azimuth": self.surface_azimuth,
            "surface_tilt": self.surface_tilt,
            "altitude": self.altitude,
            "pv_module": self.pv_module,
            "inverter": self.inverter,
            "temperature_model_params": self.temperature_model_params,
        }

    def set_params(self, latitude=None, longitude=None, surface_azimuth=None, surface_tilt=None, altitude=None,
                   pv_module=None, inverter=None, temperature_model_params=None,):
        """ Set or change Ensemble object parameters.
        """
        if latitude is not None:
            self.latitude = latitude
        if longitude is not None:
            self.longitude = longitude
        if surface_azimuth is not None:
            self.surface_azimuth = surface_azimuth
        if surface_tilt is not None:
            self.surface_tilt = surface_tilt
        if altitude is not None:
            self.altitude = altitude
        if pv_module is not None:
            self.pv_module = pv_module
        if inverter is not None:
            self.inverter = inverter
        if temperature_model_params is not None:
            self.temperature_model_params = temperature_model_params

    def transform(self, **kwargs) -> xr.DataArray:

        arguments = ["ghi", "temperature", "wind_speed"]
        for argument in arguments:
            if argument not in kwargs.keys():
                raise InvalidInputException(
                    f"The module {self.name} miss {argument} as input. The module needs {arguments} as input. "
                    f"{kwargs} are given as input."
                    f"Add {argument}=<desired_input> when adding {self.name} to the pipeline."
                )
        ghi = kwargs.get("ghi")
        dni = kwargs.get("dni")
        dhi = kwargs.get("dhi")
        temperature = kwargs.get("temperature")
        wind_speed = kwargs.get("wind_speed")
        pressure = kwargs.get("pressure")

        system = {
            'module': self.pv_module,
            'inverter': self.inverter,
            'surface_azimuth': self.surface_azimuth,
            "surface_tilt": self.surface_tilt
        }

        ghi_ = ghi.to_series()
        temperature_ = temperature.to_series()
        wind_speed_ = wind_speed.to_series()
        pressure_ = pressure.to_series() if pressure is not None else pressure

        solpos = pvlib.solarposition.get_solarposition(
            time=ghi_.index,  # required
            latitude=self.latitude,  # required
            longitude=self.longitude,  # required
            altitude=self.altitude,  # default None
            temperature=temperature_,  # default 12
            pressure=pressure_,  # default None
        )

        if dni is None and dhi is None:
            dni_ = disc(
                ghi=ghi_,  # required
                solar_zenith=solpos['apparent_zenith'],  # required
                datetime_or_doy=ghi_.index,  # required
                pressure=pressure_ if pressure is not None else 101325  # default 101325
            )["dni"]
            dhi_ = ghi_ - dni_ * tools.cosd(solpos['apparent_zenith'])
            dhi_[dhi_ < 0] = float('nan')  # cutoff negative values
        elif dni is None and dhi is not None:
            dhi_ = dhi.to_series()
            dni_ = (ghi_ - dhi_) / tools.cosd(solpos['apparent_zenith'])
            dni_[dni_ < 0] = float('nan')  # cutoff negative values
        elif dhi is None and dni is not None:
            dni_ = dni.to_series()
            dhi_ = ghi_ - dni_ * tools.cosd(solpos['apparent_zenith'])
            dni_[dni_ < 0] = float('nan')  # cutoff negative values
        else:
            dni_ = dni.to_series()
            dhi_ = dhi.to_series()

        dni_extra = pvlib.irradiance.get_extra_radiation(
            datetime_or_doy=ghi_.index
        )
        airmass_rel = pvlib.atmosphere.get_relative_airmass(
            zenith=solpos['apparent_zenith']
        )
        airmass_abs = pvlib.atmosphere.get_absolute_airmass(
            airmass_relative=airmass_rel,  # required
            pressure=pressure_ if pressure is not None else 101325  # default 101325
        )
        aoi = pvlib.irradiance.aoi(
            surface_tilt=system['surface_tilt'],  # required
            surface_azimuth=system['surface_azimuth'],  # required
            solar_zenith=solpos["apparent_zenith"],  # required
            solar_azimuth=solpos["azimuth"],  # required
        )

        total_irradiance = pvlib.irradiance.get_total_irradiance(
            surface_tilt=system['surface_tilt'],  # required
            surface_azimuth=system['surface_azimuth'],  # required
            solar_zenith=solpos['apparent_zenith'],  # required
            solar_azimuth=solpos['azimuth'],  # required
            dni=dni_,  # required
            ghi=ghi_,  # required
            dhi=dhi_,  # required
            dni_extra=dni_extra,  # default None
            airmass=airmass_rel,  # default None
            model='haydavies',
        )
        cell_temperature = pvlib.temperature.sapm_cell(
            poa_global=total_irradiance['poa_global'],  # required
            temp_air=temperature_,  # required
            wind_speed=wind_speed_,  # required
            **self.temperature_model_params,  # required
        )
        effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
            poa_direct=total_irradiance['poa_direct'],  # required
            poa_diffuse=total_irradiance['poa_diffuse'],  # required
            airmass_absolute=airmass_abs,  # required
            aoi=aoi,  # required
            module=self.pv_module,  # required
        )
        dc = pvlib.pvsystem.sapm(
            effective_irradiance=effective_irradiance,  # required
            temp_cell=cell_temperature,  # required
            module=self.pv_module  # required
        )
        ac = pvlib.inverter.sandia(
            v_dc=dc['v_mp'],  # required
            p_dc=dc['p_mp'],  # required
            inverter=self.inverter  # required
        )

        return numpy_to_xarray(ac, ghi)
