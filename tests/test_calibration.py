"""Tests of the calibration: stage equations, validation roles, and the L5 sensitivity."""

import math

import numpy as np
import pandas as pd

from Calibration.calibration_common import summarize
from Calibration.run_external_l5_sensitivity import (
    apply_odds_ratio, l5_mortality_reporting_adjustment, l5_out_referral_adjustment,
)
from Calibration.stages import stage1_preterm as preterm, stage2_routing as routing
from Calibration.stages import stage4_planned_cs as planned_cs, stage5_complications as complications
from Calibration.stages import stage6_mortality as mortality
from Calibration.validate_baseline_calibration import fit_status, target_role

def test_preterm_update_is_damped():
    # Modeled 0.10 against target 0.134: a full step would multiply by 1.34, the damped one by sqrt(1.34).
    assert math.isclose(preterm.update(0.7, 0.134, 0.10, 0.5), 0.7 * math.sqrt(1.34))
    assert math.isclose(preterm.update(0.7, 0.134, 0.134), 0.7)


def test_recognized_l45_probability():
    assert math.isclose(routing.recognized_l45_probability(0.4, 2.0), 0.75)


def test_routing_solution_reproduces_final_targets():
    tau = 0.08
    solved = routing.solve_county(
        anc=0.7, home_noanc=0.1, home_anc=0.04, final_l23=0.3,
        r=0.6, theta=2.0, tau=tau,
    )
    assert math.isclose(solved["home_lowrisk"], 0.04 / 0.4)
    assert math.isclose(solved["initial_home"], 0.3 * 0.1 + 0.7 * 0.04)
    assert math.isclose(solved["final_l23"], 0.3)
    assert math.isclose(solved["p_lb_l23_45"], tau * 0.3 / (1 - tau))
    overall_final = (
        0.3 * solved["final_l23_given_noanc"] + 0.7 * solved["final_l23_given_anc"]
    )
    assert math.isclose(overall_final, 0.3, rel_tol=1e-9)


def test_recognition_rate_combines_screening_and_dating():
    param = {
        "p_highrisk": 0.25, "sen_risk_trad": 0.8, "spec_risk_trad": 0.6,
        "GA_sequence": [36, 40, 43], "GA_anc": [0.1, 0.8, 0.1],
        "E_Preterm_LMP": [0.5, 0.1], "E_Postterm_LMP": [0.0, 0.1],
    }
    screen = 0.25 * 0.8 + 0.75 * 0.4
    # Preterm-status flags: 1 - (1 - 0.5)(1 - 0); at term 1 - 0.9 * 0.9; post-term 1 - 0.6 * 0.1.
    dating = 0.1 * 0.5 + 0.8 * 0.19 + 0.1 * 0.94
    assert math.isclose(routing.recognition_rate(param), 1 - (1 - screen) * (1 - dating))


def test_transfer_residuals_use_original_scales():
    from Calibration.stages import stage3_transfers as transfers

    modeled = {"preterm_l23": 0.05, "mcr_ratio_l23_l4": 0.30}
    targets = {"preterm_l23": 0.03, "mcr_ratio_l23_l4": 0.20}
    residuals = transfers.residuals(modeled, targets)
    assert math.isclose(residuals[0], 1.0)
    assert math.isclose(residuals[1], 1.0)


def test_transfer_counting_equations():
    from Calibration.stages import stage3_transfers as transfers

    counts = {
        "E_pt": 100, "E_ns": 200,
        "preterm_share_ns": 0.1, "events_per_ns": 1.0,
        "N_C": 650, "preterm_C": 30, "events_C": 20,
        "N_L4_other": 2000, "events_L4_other": 400, "s4": 0.4,
    }
    # Half of each eligible group transfers; 40% of transfers are received at L4.
    result = transfers.predicted(counts, 0.5, 0.5)
    n_l23 = 650 + 50 + 100
    n_l4 = 2000 + 0.4 * 150
    assert math.isclose(result["preterm_l23"], (30 + 50 + 100 * 0.1) / n_l23)
    assert math.isclose(result["mcr_ratio_l23_l4"], ((20 + 100) / n_l23) / ((400 + 0.4 * 100) / n_l4))
    # Solving for targets produced at (0.3, 0.6) recovers those probabilities.
    targets = transfers.predicted(counts, 0.3, 0.6)
    p_pt, p_ns, _ = transfers.solve(counts, targets)
    assert math.isclose(p_pt, 0.3, abs_tol=1e-4) and math.isclose(p_ns, 0.6, abs_tol=1e-4)


def test_planned_cs_shared_preterm_solution():
    coeffs = {
        "x": {"a": 0.30, "b": 0.09, "c": 0.10, "d": 0.40, "n_preterm": 100},
        "y": {"a": 0.35, "b": 0.10, "c": 0.12, "d": 0.45, "n_preterm": 300},
    }
    targets = {"x": 0.08, "y": 0.11}
    q_high, q_pre = planned_cs.solve_shared_preterm(coeffs, targets, preterm_target=0.07)
    for county, item in coeffs.items():
        assert math.isclose(item["a"] * q_high[county] + item["b"] * q_pre, targets[county])
    pooled = (100 * (0.10 * q_high["x"] + 0.40 * q_pre) + 300 * (0.12 * q_high["y"] + 0.45 * q_pre)) / 400
    assert math.isclose(pooled, 0.07)


def test_complication_proportional_updates():
    values = {name: 1.0 for name in complications.INCIDENCE}
    values.update({"p_cs_capacity_l4": 0.40, "p_cs_capacity_l23": 0.05})
    metrics = {metric: 0.02 for metric in complications.INCIDENCE.values()}
    metrics["cs_l23"] = 0.02
    targets = {metric: 0.03 for metric in complications.INCIDENCE.values()}
    targets.update({"cs_l4": 0.30, "cs_l23": 0.04})
    components = {"planned": 0.12, "transferred_in": 0.10, "starters": 0.16}
    new = complications.updated(values, metrics, targets, components)
    assert math.isclose(new["p_aph"], 1.5)
    # Starters' emergency CS must fall from 0.16 to 0.30 - 0.12 - 0.10 = 0.08.
    assert math.isclose(new["p_cs_capacity_l4"], 0.20)
    assert math.isclose(new["p_cs_capacity_l23"], 0.10)


def test_mortality_update_hits_proportional_targets():
    baseline, rr = mortality.update(
        facility_baseline=0.002, l23_rr=1.0,
        modeled_l4=50.0, modeled_l23=20.0,
        target_l4=100.0, target_l23=40.0,
    )
    assert math.isclose(baseline, 0.004)
    assert math.isclose(rr, 1.0)


def test_validation_mcr_matches_khis_event_rate():
    people = pd.DataFrame({
        "i_loc": [2, 2],
        "i_loc_new_v2": [2, 2],
        "i_mod": ["SVD", "SVD"],
        "i_preterm": [0, 0],
        "i_eclampsia_new": [1, 0],
        "i_OL_final": [1, 0],
        "i_pph_new": [0, 0],
        "i_mat_sepsis_new": [0, 0],
        "i_aph": [0, 0],
        "i_ruptured_uterus": [0, 0],
        "i_mat_death": [0, 0],
        "p_mat_death": [0.0, 0.0],
        "i_severe_new": [1, 0],
        "i_ANC": [1, 0],
    })

    # Two complication events over two deliveries equals an MCR of one.
    assert summarize(people)["mcr_l4"] == 1.0
    assert summarize(people)["smo_fac"] == 0.5
    assert summarize(people)["final_l45_given_anc"] == 1.0


def test_target_roles_and_statuses_are_explicit():
    assert target_role("final_l45") == "fit_target"
    assert target_role("smo_fac") == "validation_only"
    assert target_role("final_l45_given_anc") == "validation_only"
    assert target_role("planned_cs_preterm") == "validation_only"
    assert fit_status(0.09) == "accepted_within_10pct"
    assert fit_status(-0.15) == "review_10_to_20pct"
    assert fit_status(0.21) == "outside_20pct"



def test_predicted_transfer_matches_labour_emergency_share():
    from Calibration.stages import stage3_transfers as transfers

    rescue = {"labour_emergency": 60, "maternal_complication": 25, "preterm": 15, "normal_labour": 40}
    share = transfers.known_reason_mix(rescue)["labour_emergency"]
    assert math.isclose(share, 0.6)
    # 60 predicted transfers at 0.3 imply 200 eligible; other transfers 20 + 100*0.2 + 200*0.1 = 60,
    # so predicted transfers must be 1.5 x 60 = 90, i.e. probability 0.45.
    counts = {"predicted_transfers": 60, "severe_transfers": 20, "E_pt": 100, "E_ns": 200}
    assert math.isclose(transfers.updated_predicted(0.3, counts, 0.2, 0.1, share), 0.45)


def test_tau_mix_rule_reproduces_tau_and_mix():
    from Calibration.stages import stage3_transfers as transfers

    counts = {"l23_starters": 1000, "predicted_transfers": 80, "severe_transfers": 10, "E_pt": 100, "E_ns": 200}
    shares = {"labour_emergency": 0.6, "maternal_complication": 0.25, "preterm": 0.15}
    (t_pt, t_ns, t_pred), at_bound = transfers.solve_tau_mix(0.2, counts, 0.15, shares)
    pred = t_pred * 80 / 0.2
    pt, ns = t_pt * 100, t_ns * 200
    total = pred + pt + ns + 10
    assert not at_bound
    assert abs(total / 1000 - 0.15) < 1e-12
    assert abs(pred / total - 0.6) < 1e-12 and abs(pt / total - 0.15) < 1e-12


def test_pooled_cause_mix_keeps_county_totals():
    from Calibration.calibration_common import CALIBRATED_COUNTIES, CAUSES, county_targets

    targets = {county: county_targets(county) for county in CALIBRATED_COUNTIES}
    for county, target in targets.items():
        assert math.isclose(sum(target[f"{cause}_l4"] for cause in CAUSES), target["mcr_l4"])
    kisii, mombasa = targets["kisii"], targets["mombasa"]
    for cause in CAUSES:
        assert math.isclose(kisii[f"{cause}_l4"] / kisii["mcr_l4"], mombasa[f"{cause}_l4"] / mombasa["mcr_l4"])


def source_population():
    return pd.DataFrame({
        "i_loc_new_v2": [3, 3, 3, 2],
        "i_transfer_pred": [1, 1, 0, 1],
        "i_transfer_actual": [0, 0, 0, 0],
        "i_pph_new": [1, 0, 0, 1],
        "i_mat_sepsis_new": [0, 1, 0, 0],
        "i_eclampsia_new": [0, 0, 0, 0],
        "i_OL_final": [0, 0, 0, 0],
        "i_aph": [0, 0, 0, 0],
        "i_ruptured_uterus": [0, 0, 0, 0],
        "p_mat_death": [0.10, 0.01, 0.001, 0.10],
    })


def test_apply_odds_ratio():
    result = apply_odds_ratio(np.array([0.0, 0.1, 1.0]), 2.0)
    np.testing.assert_allclose(result, [0.0, 2 * 0.1 / 1.1, 1.0])


def test_reporting_adjustment_does_not_modify_population():
    source = source_population()
    original = source.copy(deep=True)
    result = l5_mortality_reporting_adjustment(source, external_share=0.25, late_referral_rr=2.0)

    pd.testing.assert_frame_equal(source, original)
    assert np.isclose(result["internal_l5_mmr"], (0.10 + 0.01 + 0.001) / 3 * 100_000)
    # Proxy: the two complication-bearing transfers received at L5, odds doubled.
    assert np.isclose(result["proxy_external_mmr"], 1e5 * np.mean(apply_odds_ratio(np.array([0.10, 0.01]), 2.0)))
    expected = 0.75 * result["internal_l5_mmr"] + 0.25 * result["proxy_external_mmr"]
    assert np.isclose(result["reported_l5_mmr"], expected)


def test_out_referral_removes_modeled_deaths():
    # Four L5 women (risks 0.10, 0.01, 0.001, 0.05); the proxy group is the first two and the last.
    source = pd.concat([source_population(), source_population().iloc[[0]].assign(p_mat_death=0.05)],
                       ignore_index=True)
    source.loc[3, "i_loc_new_v2"] = 2
    source.loc[4, "i_loc_new_v2"] = 3
    result = l5_out_referral_adjustment(source, 0.25)  # one of four women leaves
    assert np.isclose(result["highest risk first"], 1e5 * (0.10 + 0.01 + 0.001 + 0.05 - 0.10) / 3)
    assert np.isclose(result["proxy average"], 1e5 * (0.161 - (0.10 + 0.01 + 0.05) / 3) / 3)
    assert np.isnan(l5_out_referral_adjustment(source, 0.9)["proxy average"])
