import time
import numpy as np
import pandas as pd

from autopv.autopv import AutoPV
from autopv.config import MeasurementUnit, ModelPool, AdaptionConfiguration, get_plant_kwp_synthetic_selection
from data.utils import load_example_data

if __name__ == "__main__":
    start = time.time()

    # Data specification
    _, train, test = load_example_data()

    measurement_unit = MeasurementUnit.kW

    # Experiment specification
    n_runs = 1  # number of runs

    model_pool = ModelPool.default  # default, nearby

    overwrite_plant_loop = False  # enables overwriting the loop of the leave-one-out-evaluation
    plant_test_overwrite = "pv0"  # pv0, pv1, ..., pv10

    adaption_config = AdaptionConfiguration.AIB  # none, AFB, AIB

    C = pd.Timedelta("28d")  # adaption cycle
    K = pd.Timedelta("28d")  # period of considered samples for adaption, AIB considers l * K samples
    online_start = pd.Timestamp("2020-01-01 00:00:00")

    # Plant specification
    plants_kWp_test = get_plant_kwp_synthetic_selection()
    # specify the peak power rating of the PV plants to be tested, the structure is as follows:
    # {'pv_name_0': 100, 'pv_name_1': 200, 'pv_name_2': 300}
    if model_pool == ModelPool.nearby:
        nearby_plants_kWp = get_plant_kwp_synthetic_selection()
        # specify the peak power rating of the nearby PV plants, the structure is as follows:
        # {'pv_name_0': 100, 'pv_name_1': 200, 'pv_name_2': 300}
    else:
        nearby_plants_kWp = None

    for number_run in range(0, n_runs):
        model_directory = f"results/models/pool_{number_run}"

        for target_name, target_kWp in plants_kWp_test.items():

            if overwrite_plant_loop and plant_test_overwrite != target_name:
                continue

            rd = np.random.RandomState(314 + int("".join(filter(str.isdigit, target_name))))

            autopv = AutoPV(target_name=target_name, target_kWp=target_kWp,
                            model_pool=model_pool, measurement_unit=measurement_unit,
                            adaption_config=adaption_config, C=C, K=K, nearby_plants_kWp=nearby_plants_kWp)

            # Summaries will be created in the result-folder
            autopv.fit(data=train)
            autopv.predict(data=test, online_start=online_start)

    print(f"Finished! Experiment took {(time.time() - start) / 60} minutes!")
