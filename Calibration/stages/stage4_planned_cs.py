"""Stage 4: analytical planned-CS equations.

Planned CS is assigned to ANC mothers who start labour at L4/5, in two groups:
    H  predicted high risk, not predicted preterm   -> p_elec_CS|highrisk
    P  predicted preterm                             -> p_elec_CS|preterm

Targets:
    planned CS at final L4 = PLANNED_CS_SHARE x KHIS L4 CS rate (elective_cs_l4)
    planned CS among preterm women starting at L4/5 = PRETERM_PLANNED_CS (planned_cs_preterm)
(both in calibration_data.py)

p_elec_CS|highrisk is county-specific; p_elec_CS|preterm is shared and solved
from the preterm target pooled over the calibrated counties.
"""

from __future__ import annotations


def coefficients(people):
    """Eligibility shares from one simulation (see module docstring)."""
    start_l45 = people["i_loc"].ge(2)
    final_l4 = people["i_loc_new_v2"].eq(2)
    preterm_start = start_l45 & people["i_preterm"].eq(1)
    anc = people["i_ANC"].eq(1)
    group_h = start_l45 & anc & people["i_risk_pred"].eq(1) & people["i_preterm_pred"].eq(0)
    group_p = start_l45 & anc & people["i_preterm_pred"].eq(1)
    return {
        "a": float(group_h[final_l4].mean()),
        "b": float(group_p[final_l4].mean()),
        "c": float(group_h[preterm_start].mean()),
        "d": float(group_p[preterm_start].mean()),
        "n_preterm": int(preterm_start.sum()),
    }


def solve_shared_preterm(coeffs, planned_targets, preterm_target):
    """County p_elec_CS|highrisk and shared p_elec_CS|preterm, in closed form.

    For each county: a·q_H,c + b·q_P = planned_targets[c], so
    q_H,c = (planned_targets[c] - b·q_P) / a. Substituting into the pooled
    preterm equation sum_c w_c (c·q_H,c + d·q_P) = preterm_target, with w_c the
    county's share of preterm women starting at L4/5, is linear in q_P.
    """
    total = sum(item["n_preterm"] for item in coeffs.values())
    constant = slope = 0.0
    for county, item in coeffs.items():
        weight = item["n_preterm"] / total
        constant += weight * item["c"] * planned_targets[county] / item["a"]
        slope += weight * (item["d"] - item["c"] * item["b"] / item["a"])
    q_preterm = (preterm_target - constant) / slope
    q_highrisk = {
        county: (planned_targets[county] - item["b"] * q_preterm) / item["a"]
        for county, item in coeffs.items()
    }
    values = [q_preterm, *q_highrisk.values()]
    if not all(0 <= value <= 1 for value in values):
        raise ValueError(f"Planned-CS solution lies outside [0, 1]: {q_highrisk}, {q_preterm}")
    return q_highrisk, q_preterm
