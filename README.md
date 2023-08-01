# eplines
[![pypi_shield](https://img.shields.io/pypi/v/eplines.svg)](https://pypi.org/project/eplines/)
[![pypi_license](https://badgen.net/pypi/license/eplines/)](https://pypi.org/project/eplines/)
![tests_workflow](https://github.com/rhkarls/eplines/actions/workflows/run_flake8_pytest.yml/badge.svg)

Heatmaps of empirical and exceedance probability of many (time-)series.

Aggregate empirical and exceedance frequency/probability of many and long time-series, for example
summarizing meteorological and hydrological time-series. Create cycle plots to illustrate for example annual cycles for longer time-series, see some examples of this usage below.

Also see the [`kdlines`](https://github.com/rhkarls/kdlines) package for similar functionality using kernel density estimation. 

This package was inspired by [DenseLines](https://dig.cmu.edu/publications/2018-million-time-series.html) 
by Moritz & Fisher.

Please note that this package is at alpha stage and experimental.
## Requirements

    numpy
    matplotlib

## Installation

`pip install eplines`

## Examples

See \examples folder for example applications of `eplines`

Temperature time-series ECDF `\examples\example_temperature_timeseries.py`

![example_ecdf_airtemp](https://github.com/rhkarls/eplines/blob/main/examples/temperature_ecdf_example.png)

Discharge time-series exceedance probability (aka flow-duration curve for each individual day) `\examples\example_usgs_discharge.py`

![example_exceedance_df](https://github.com/rhkarls/eplines/blob/main/examples/discharge_exceedance_example.png)



