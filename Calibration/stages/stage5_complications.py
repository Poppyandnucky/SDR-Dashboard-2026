"""Stage 5: complication incidence and CS capacity.

Each parameter mainly drives one internal-L4 (or L2/3) target, and the model
target responds in proportion to it, so each update is one equation applied
inside the joint calibration loop:

    p_aph, p_ruptured_uterus, p_eclampsia, p_OL_scale,
    pph_incidence_multiplier, sepsis_incidence_multiplier:
        x <- x * target / modeled L4 rate
    p_cs_capacity[L4]:
        x <- x * (cs_l4_target - planned CS - transferred-in emergency CS)
                 / emergency CS among women who started at L4/5   (all shares of final L4 births)
    p_cs_capacity[L2/3]:
        x <- x * p_cs_l23 / modeled L2/3 CS

Targets: KHIS internal-L4 rates x scale_comp (4.48) for the six complications;
raw KHIS CS rates (momish_khis_targets.py).
"""

from __future__ import annotations


# Workbook parameter -> validator metric (whose target is the matching *_l4_target).
INCIDENCE = {
    "p_aph": "aph_l4",
    "p_ruptured_uterus": "rupture_l4",
    "p_eclampsia": "eclampsia_l4",
    "p_OL_scale": "ol_l4",
    "pph_incidence_multiplier": "pph_l4",
    "sepsis_incidence_multiplier": "sepsis_l4",
}
# Indices of p_cs_capacity: [home, L2/3, L4, L5].
CS_CAPACITY_L23, CS_CAPACITY_L4 = 1, 2


def cs_l4_components(people):
    """CS at final L4 split by origin, each as a share of final L4 births."""
    l4 = people["i_loc_new_v2"].eq(2)
    births = int(l4.sum())
    emergency = l4 & people["i_mod"].eq("EmCS")
    return {
        "planned": float((l4 & people["i_mod"].eq("ELCS")).sum() / births),
        "transferred_in": float((emergency & people["i_loc"].le(1)).sum() / births),
        "starters": float((emergency & people["i_loc"].ge(2)).sum() / births),
    }


def updated(values, metrics, targets, components):
    """One proportional update of the Stage 5 parameters.

    ``values`` holds the current parameters, with ``p_cs_capacity_l23`` and
    ``p_cs_capacity_l4`` for the two fitted capacity entries.
    """
    new = {name: values[name] * targets[metric] / metrics[metric] for name, metric in INCIDENCE.items()}
    needed = targets["cs_l4"] - components["planned"] - components["transferred_in"]
    if needed <= 0:
        raise ValueError("Planned and transferred-in CS alone exceed cs_l4_target")
    new["p_cs_capacity_l4"] = min(1.0, values["p_cs_capacity_l4"] * needed / components["starters"])
    new["p_cs_capacity_l23"] = min(1.0, values["p_cs_capacity_l23"] * targets["cs_l23"] / metrics["cs_l23"])
    return new
