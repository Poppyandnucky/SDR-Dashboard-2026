"""Tests of the model changes the calibration depends on (global_func.py, mortality.py)."""

import math

import numpy as np

from global_func import NO_CAPACITY_LIMIT, facility_capacity_limit, odds_prob, reset_flags
from mortality import maternal_facility_weights

def test_facility_capacity_limit_is_off_by_default():
    track = {"Facility_Capacity_Track": np.array([[1340.0]])}
    flags = reset_flags()
    assert facility_capacity_limit(track, 0, flags) == NO_CAPACITY_LIMIT
    flags["flag_facility_capacity_limit"] = 1
    assert facility_capacity_limit(track, 0, flags) == 1340.0


def test_preterm_by_anc_responds_smoothly_to_pt_scale():
    rounded = odds_prob(0.48, 0.10, 0.65)
    exact = odds_prob(0.48, 0.10, 0.65, decimals=None)
    assert math.isclose(rounded[0], round(exact[0], 2))
    assert math.isclose(0.65 * exact[0] + 0.35 * exact[1], 0.10, rel_tol=1e-6)
    assert odds_prob(0.48, 0.101, 0.65, decimals=None)[0] > exact[0]


def test_facility_weights_fall_back_to_shared_value():
    weights = maternal_facility_weights({"weight_facility_mat": 2.5})

    np.testing.assert_allclose(weights, [0.0, 2.5, 2.5, 2.5])


def test_l23_relative_risk_modifies_only_l23():
    weights = maternal_facility_weights({
        "weight_facility_mat": 2.5,
        "RR_MM_l23_vs_l4": 0.8,
    })

    np.testing.assert_allclose(weights, [0.0, 2.0, 2.5, 2.5])
