import matplotlib.pyplot as plt
import pandas as pd
from pywatts.core.pipeline import Pipeline
from pywatts.modules import FunctionModule

from pywatts_modules.pvlib_wrapper import PVLibWrapper

if __name__ == "__main__":

    data = pd.read_csv("./data/weather_rheinstetten_181920.csv",
                       parse_dates=["MESS_DATUM"], index_col="MESS_DATUM")

    for surface_azimuth in range(0, 316, 45):
        for surface_tilt in range(0, 91, 15):
            pipeline = Pipeline()
            pv = PVLibWrapper(latitude=48.9685, longitude=8.30704, altitude=116,
                              surface_azimuth=surface_azimuth,
                              surface_tilt=surface_tilt)(temperature=pipeline["TU[degC]"],
                                                         pressure=pipeline["P[hPa]"],
                                                         ghi=pipeline["GHI[W/m^2]"],
                                                         dhi=pipeline["DHI[W/m^2]"],
                                                         wind_speed=pipeline["V[m/s]"])
            FunctionModule(lambda x: x.clip(min=0), name="Generation")(x=pv)

            result = pipeline.test(data, summary=False)
            result["Generation"].to_dataframe("PV Generation").to_csv(f"./data/"
                                                                      f"pv_generation_orient_{surface_azimuth}_"
                                                                      f"tilt_{surface_tilt}.csv")
            result["Generation"].to_dataframe("PV Generation").plot()
            plt.title(f"{surface_azimuth, surface_tilt}")
            plt.show()
            print(f"Finished {surface_azimuth, surface_tilt}")
