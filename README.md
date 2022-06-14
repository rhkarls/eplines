# eplines
[![pypi_shield](https://img.shields.io/pypi/v/eplines.svg)](https://pypi.org/project/eplines/)
[![pypi_license](https://badgen.net/pypi/license/eplines/)](https://pypi.org/project/eplines/)
![tests_workflow](https://github.com/rhkarls/eplines/actions/workflows/run_flake8_pytest.yml/badge.svg)

Heatmaps of empirical and exceedance probability of many (time-)series.

This package is at alpha stage and experimental.

## Requirements

    numpy
    matplotlib

## Installation

`pip install eplines`

## Examples

See \examples folder for example applications of `eplines`

Temperature time series ECDF `\examples\example_temperature_timeseries.py`

![example_ecdf_airtemp](https://github.com/rhkarls/eplines/blob/main/examples/temperature_ecdf_example.png)

Discharge time series exceedance probability (aka flow-duration curve) `\examples\example_usgs_discharge.py`

![example_exceedance_df](https://github.com/rhkarls/eplines/blob/main/examples/discharge_exceedance_example.png)



