# AutoPV: Automated photovoltaic forecasts with limited information using an ensemble of pre-trained models
This repository contains a Python Implementation of AutoPV for automated photovoltaic forecasts with limited information using an ensemble of pre-trained models.

## Methodology

The underlying idea of AutoPV is to describe the arbitrary mounting configuration of a new PV plant as a convex linear combination of outputs from a sufficiently diverse ensemble pool of PV models of the same region. AutoPV incorporates three steps: i) create the ensemble model pool, ii) form the ensemble output by an optimally weighted sum of the scaled model outputs in the pool, and iii) rescale the ensemble output with the new PV plant’s peak power rating.

![pipeline](https://github.com/SMEISEN/AutoPV/assets/33990691/56363d4b-5418-427b-b723-bf14255804ce)


## Installation

To install this project, perform the following steps.
1) Clone the project
2) Open a terminal of the virtual environment where you want to use the project
3) cd AuroPV
4) pip install . or pip install -e . if you want to install the project editable.

## How to use

Exemplary evaluations using AutoPV are given in the examples folder.

### Model pools
- The default model pool is based on physical-inspired modeling and uses 12 models (tilt: 15°, 45°, 75°, azimuth: 0°, 90°, 180°, 270°). The default model pool is suitable for situations where no data of nearby PV plants are available.
- The nearby plants model pool uses machine learning-based modeling to create the models using data from nearby PV plants. The nearby plants model pool can represent shading if it is present in the nearby PV plants.

### Evaluation types
- The offline evaluation optimizes the ensemble weights using the entire training data and does not adapt the weights over time.
- The online evaluation cyclically adapts the weights based on the testing data (cold-start) and does not require training data.

## Citation

If you use this method please cite the corresponding paper:
> Stefan Meisenbacher, Benedikt Heidrich, Tim Martin, Ralf Mikut, and Veit Hagenmeyer. 2023. AutoPV: Automated photovoltaic forecasts with limited information using an ensemble of pre-trained models. In Proceedings of the Fourteenth ACM International Conference on Future Energy Systems (e-Energy ’23). Association for Computing Machinery, New York, NY, USA, 386–414. https://doi.org/10.1145/3575813.3597348

## Funding
This project is funded by the Helmholtz Association under the Program “Energy System Design” and the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI.

## References

The example data includes weather measurements from DWD:
> Deutscher Wetterdienst. 2023. Historical 10-minute station observations of solar incoming radiation, longwave downward radiation, pressure, air temperature, and mean wind speed for Germany. https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes
