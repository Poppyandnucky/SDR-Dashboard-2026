"""Tests of the model changes the calibration depends on (global_func.py, mortality.py)
and of the PULSE and PROMPTS interventions."""

import math

import numpy as np
import pandas as pd

from Calibration.calibration_common import simulate

from global_func import NO_CAPACITY_LIMIT, facility_capacity_limit, odds_prob, reset_flags
from intrapartum import apply_fqa_effect
from LB_effect import odds_update, pulse_effect
from mortality import maternal_facility_weights

PULSE_PARAM = {
    "baseline_l23_preference": 0.4,
    "pulse_implementation_index": 0.5,
    "pulse_or_prompts_registration": 1.47,
    "pulse_performance_improve": 0.062,
    "fqa_implementation_index": 0.5,
    "fqa_knowledge_improve": 0.043,
}


def flags_with(**on):
    flags = reset_flags()
    flags.update(on)
    return flags


def knowledge_gain(flags, param):
    """Rise in L2/3, L4, and L5 knowledge from apply_fqa_effect, starting at 0.6."""
    P = apply_fqa_effect({"knowledge": np.array([0.0, 0.6, 0.6, 0.6])}, flags, param)
    np.testing.assert_allclose(P["knowledge"][1:4] - 0.6, P["knowledge"][1] - 0.6)
    return float(P["knowledge"][1] - 0.6)


def test_fqa_and_pulse_knowledge_gains_skip_home():
    flags = flags_with(flag_pulse=1, flag_fqa=1)
    P = apply_fqa_effect({"knowledge": np.array([0.0, 0.6, 0.6, 0.6])}, flags, PULSE_PARAM)
    np.testing.assert_allclose(P["knowledge"], [0.0, 0.631, 0.631, 0.631])


def test_facility_capacity_limit_is_off_by_default():
    track = {"Facility_Capacity_Track": np.array([[1340.0]])}
    flags = reset_flags()
    recalibrated = {"baseline_l23_preference": 0.4}
    assert facility_capacity_limit(track, 0, flags, recalibrated) == NO_CAPACITY_LIMIT
    flags["flag_facility_capacity_limit"] = 1
    assert facility_capacity_limit(track, 0, flags, recalibrated) == 1340.0


def test_original_model_counties_keep_the_capacity_limit():
    track = {"Facility_Capacity_Track": np.array([[1340.0]])}
    assert facility_capacity_limit(track, 0, reset_flags(), {}) == 1340.0


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


def test_pulse_raises_prompts_registration_by_scaled_odds_ratio():
    # OR_effective = exp(0.5 x ln 1.47); Kisii's current registration 0.151 -> 0.177.
    pulse = pulse_effect(flags_with(flag_pulse=1), PULSE_PARAM)
    assert math.isclose(pulse["prompts_registration_or"], math.exp(0.5 * math.log(1.47)))
    assert math.isclose(pulse["knowledge_gain"], 0.5 * 0.062)
    assert math.isclose(odds_update(0.151, pulse["prompts_registration_or"]), 0.177, abs_tol=5e-4)
    assert pulse_effect(flags_with(flag_pulse=0), PULSE_PARAM) is None


def test_fqa_and_pulse_knowledge_gains_take_the_larger():
    pulse_gain = 0.5 * 0.062
    fqa_gain = 0.5 * 0.043
    assert math.isclose(knowledge_gain(flags_with(flag_pulse=1, flag_fqa=1), PULSE_PARAM), pulse_gain)
    assert math.isclose(knowledge_gain(flags_with(flag_pulse=1), PULSE_PARAM), pulse_gain)
    assert math.isclose(knowledge_gain(flags_with(flag_fqa=1), PULSE_PARAM), fqa_gain)


def test_original_model_fqa_still_reaches_home():
    original = {key: value for key, value in PULSE_PARAM.items() if key != "baseline_l23_preference"}
    P = apply_fqa_effect({"knowledge": np.array([0.0, 0.6, 0.6, 0.6])}, flags_with(flag_fqa=1), original)
    np.testing.assert_allclose(P["knowledge"], [0.0215, 0.6215, 0.6215, 0.6215])


def test_pulse_applies_in_original_model_counties_too():
    original = {key: value for key, value in PULSE_PARAM.items() if key != "baseline_l23_preference"}
    flags = flags_with(flag_pulse=1, flag_fqa=1)
    assert pulse_effect(flags, original) == pulse_effect(flags, PULSE_PARAM)
    # PULSE's gain (0.031) beats FQA's (0.0215) at facilities; FQA's still reaches home.
    P = apply_fqa_effect({"knowledge": np.array([0.0, 0.6, 0.6, 0.6])}, flags, original)
    np.testing.assert_allclose(P["knowledge"], [0.0215, 0.631, 0.631, 0.631])


def kakamega_prompts_run(prompts_values, **flags_on):
    values = {(name, None): value for name, value in prompts_values.items()}
    return simulate("kakamega", values, 1, 2, 20261201, flags_with(**flags_on))[0]


def test_prompts_registration_and_screening_for_each_mother():
    # Half the mothers registered; registered mothers' 4+ ANC probability x 1.18.
    people = kakamega_prompts_run({"prompts_implementation_index": 0.5, "prompts_rr_anc4p": 1.18}, flag_PROMPTS=1)
    registered = people["i_prompts_registered"] == 1
    assert abs(registered.mean() - 0.5) < 0.02
    anc_ratio = people.loc[registered, "i_ANC"].mean() / people.loc[~registered, "i_ANC"].mean()
    assert abs(anc_ratio - 1.18) < 0.06
    np.testing.assert_array_equal(people["i_prompts_screened"], people["i_prompts_registered"] * people["i_ANC"])


def test_prompts_without_effects_uses_the_same_random_numbers():
    # With RR 1 and screening held at baseline, PROMPTS changes nobody's outcome.
    baseline = kakamega_prompts_run({})
    no_effect = {"prompts_implementation_index": 0.5, "prompts_rr_anc4p": 1.0,
                 "sen_risk_trad_target": 0.7641, "spec_risk_trad_target": 0.6306}
    prompts = kakamega_prompts_run(no_effect, flag_PROMPTS=1)
    assert baseline["i_prompts_registered"].sum() == 0
    pd.testing.assert_frame_equal(baseline.drop(columns=["i_prompts_registered", "i_prompts_screened"]),
                                  prompts.drop(columns=["i_prompts_registered", "i_prompts_screened"]))


def test_fqa_pulse_feedback_is_off_unless_switched_on():
    param = {**PULSE_PARAM, "fqa_pulse_modifier": 0.2}
    without = pulse_effect(flags_with(flag_pulse=1, flag_fqa=1), param)
    assert without == pulse_effect(flags_with(flag_pulse=1), param)
    # Switched on, FQA (index 0.5, modifier 0.2) multiplies both PULSE effects by 1.1.
    feedback = pulse_effect(flags_with(flag_pulse=1, flag_fqa=1, flag_fqa_pulse_feedback=1), param)
    assert math.isclose(feedback["prompts_registration_or"], math.exp(1.1 * 0.5 * math.log(1.47)))
    assert math.isclose(feedback["knowledge_gain"], 1.1 * 0.5 * 0.062)
    assert pulse_effect(flags_with(flag_pulse=1, flag_fqa_pulse_feedback=1), param) == without
