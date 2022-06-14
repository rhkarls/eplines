# -*- coding: utf-8 -*-
"""
pytest functions for eplines

Expected results are stored in numpy npz file
https://numpy.org/doc/stable/reference/generated/numpy.savez.html#numpy.savez

np.savez('expected_grids.npz',
         grid_expected_ecdf=grid_expected_ecdf,
         grid_expected_exceedance=grid_expected_exceedance,
         grid_expected_deceedance=grid_expected_deceedance)

# expected grids
expected_grids = np.load("tests/expected_grids.npz")
grid_expected_ecdf = expected_grids["grid_expected_ecdf"]
grid_expected_exceedance = expected_grids["grid_expected_exceedance"]
grid_expected_deceedance = expected_grids["grid_expected_deceedance"]

expected grids test for regressions

grids are generated based on data generated below also seen in "example_generated_data.py"
"""

import numpy as np
import pytest
from eplines import ECDFLines

x_n = 100
y_n = 1000

y_res = 100
x = np.linspace(0, x_n, x_n)
ys = np.empty((y_n, x_n))
randg = np.random.default_rng(seed=645)
for i in range(y_n):
    ys[i] = np.sin(-x / 3) + 2 * randg.standard_normal(1) + x * 0.1 + 11

# expected grids
expected_grids = np.load("tests/expected_grids.npz")
grid_expected_ecdf = expected_grids["grid_expected_ecdf"]
grid_expected_exceedance = expected_grids["grid_expected_exceedance"]
grid_expected_deceedance = expected_grids["grid_expected_deceedance"]


def test_ecdf():
    ecdf_regular = ECDFLines(y_res=y_res)
    ecdf_regular.ecdf(y_lines=ys, x=x)

    grid_result = ecdf_regular.ecdfs
    grid_expected = grid_expected_ecdf

    assert np.allclose(grid_result, grid_expected)

def test_ecdf_exceedance():
    ecdf_ex = ECDFLines(y_res=y_res, mode='exceedance')
    ecdf_ex.ecdf(y_lines=ys, x=x)

    grid_result = ecdf_ex.ecdfs
    grid_expected = grid_expected_exceedance

    assert np.allclose(grid_result, grid_expected)

def test_ecdf_deceedance():
    ecdf_de = ECDFLines(y_res=y_res, mode='deceedance')
    ecdf_de.ecdf(y_lines=ys, x=x)

    grid_result = ecdf_de.ecdfs
    grid_expected = grid_expected_deceedance

    assert np.allclose(grid_result, grid_expected)