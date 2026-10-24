"""Stage 3: L2/3 emergency-transfer probabilities.

Fitted per county:
    t_l23_l45_preterm    to the KHIS L2/3 preterm share
    t_l23_l45_notsevere  to the KHIS L2/3:L4 complication-event ratio
                         (internal L4 only; L5 receives external referrals)
    t_l23_l45_predicted  so labour-emergency transfers are the Rescue share of
                         L2/3 transfers with a known reason. No KHIS target
                         identifies it: the L2/3 CS decision precedes this transfer.
Held at original value:
    t_l23_l45_severe (90).

Counting solution. One simulation splits women who start labour at L2/3 and
are not moved by the earlier pathways (predicted, severe) into:
    A  preterm without complication  (eligible for t_l23_l45_preterm),   E_pt women
    B  non-severe complication       (eligible for t_l23_l45_notsevere), E_ns women
    C  everyone else                 (stays at L2/3)
With expected stays E_pt(1 - t_l23_l45_preterm) and E_ns(1 - t_l23_l45_notsevere),
and a share s4 of transferred women received at L4, the final L2/3 preterm share
and the L2/3:L4 event ratio are explicit functions of the two probabilities and
are solved directly. t_l23_l45_predicted is then rescaled so

    predicted transfers = w / (1 - w) x (severe + preterm + non-severe transfers),

where w is the Rescue labour-emergency share. Re-simulating updates the counts;
a few passes suffice. Probabilities here are fractions; the workbook stores
percentages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


FITTED = {
    "preterm": "t_l23_l45_preterm",
    "notsevere": "t_l23_l45_notsevere",
    "predicted": "t_l23_l45_predicted",
}
TARGETS = ("preterm_l23", "mcr_ratio_l23_l4")
# Residual scales from the original transfer objective. With two equations in two
# unknowns the solution is exact, so they matter only if a probability is at 0 or 1.
TARGET_SCALES = {"preterm_l23": 0.02, "mcr_ratio_l23_l4": 0.10}
REASONS = ("labour_emergency", "maternal_complication", "preterm")
EVENT_COLUMNS = (
    "i_eclampsia_new", "i_OL_final", "i_pph_new", "i_mat_sepsis_new", "i_aph", "i_ruptured_uterus",
)

def residuals(modeled, targets):
    return np.asarray([
        (modeled[name] - targets[name]) / TARGET_SCALES[name] for name in TARGETS
    ])

def pools(people):
    """Counts from one simulation that determine L2/3 and L4/5 composition
    (symbols as in the module docstring and the methods document)."""
    events = sum((people[column] == 1).astype(int) for column in EVENT_COLUMNS)
    preterm = people["i_preterm"] == 1
    group = people["i_transfer_group"]
    # Pathway that transferred an L2/3 starter: 1 predicted labour emergency; otherwise
    # her pre-transfer group (2 severe, 4 preterm, 5 non-severe), since those pathways
    # run in that order over disjoint groups.
    moved_later = people["i_loc"].eq(1) & people["i_transfer_pred"].eq(0) & people["i_loc_new_v2"].ne(1)
    reason = pd.Series(np.where(people["i_transfer_pred"].eq(1), 1, np.where(moved_later, group, 0)), index=people.index)
    remaining = people["i_loc"].eq(1) & ~reason.isin([1, 2])
    a = remaining & group.eq(4)
    b = remaining & group.eq(5)
    c = remaining & ~group.isin([4, 5])
    other_l4 = people["i_loc_new_v2"].eq(2) & ~reason.isin([4, 5])
    moved = people["i_loc"].eq(1) & reason.isin([4, 5])
    return {
        "l23_starters": int(people["i_loc"].eq(1).sum()),
        "predicted_transfers": int((people["i_loc"].eq(1) & reason.eq(1)).sum()),
        "severe_transfers": int((people["i_loc"].eq(1) & reason.eq(2)).sum()),
        "E_pt": int(a.sum()),                                  # E_pt: group A (preterm, no complication)
        "E_ns": int(b.sum()),                                  # E_ns: group B (non-severe complication)
        "preterm_share_ns": float(preterm[b].mean()),          # pi_B: preterm share of B
        "events_per_ns": float(events[b].mean()),              # e_B: complication events per woman in B
        "N_C": int(c.sum()),                                   # N_C: group C (everyone else)
        "preterm_C": int(preterm[c].sum()),                    # PT_C
        "events_C": int(events[c].sum()),                      # ev_C
        "N_L4_other": int(other_l4.sum()),                     # N_L4: final-L4 births not from A or B
        "events_L4_other": int(events[other_l4].sum()),        # ev_L4
        "s4": float(people.loc[moved, "i_loc_new_v2"].eq(2).mean()),  # s4: share of A/B transfers received at L4
    }


def predicted(counts, p_pt, p_ns):
    """L2/3 preterm share and L2/3:L4 event ratio at (p_pt, p_ns)."""
    stay_pt = counts["E_pt"] * (1 - p_pt)
    stay_ns = counts["E_ns"] * (1 - p_ns)
    moved_pt, moved_ns = counts["E_pt"] - stay_pt, counts["E_ns"] - stay_ns
    n_l23 = counts["N_C"] + stay_pt + stay_ns
    preterm_l23 = (counts["preterm_C"] + stay_pt + stay_ns * counts["preterm_share_ns"]) / n_l23
    events_l23 = counts["events_C"] + stay_ns * counts["events_per_ns"]
    n_l4 = counts["N_L4_other"] + counts["s4"] * (moved_pt + moved_ns)
    events_l4 = counts["events_L4_other"] + counts["s4"] * moved_ns * counts["events_per_ns"]
    ratio = (events_l23 / n_l23) / (events_l4 / n_l4)
    return {"preterm_l23": preterm_l23, "mcr_ratio_l23_l4": ratio}


def known_reason_mix(rescue):
    """Rescue shares of labour emergency, maternal complication, and preterm among
    L2/3 transfers with a model pathway (they sum to 1); normal-labour and
    other/unknown transfers have no model pathway and are excluded."""
    known = sum(rescue[reason] for reason in REASONS)
    return {reason: rescue[reason] / known for reason in REASONS}


def solve_tau_mix(current_predicted, counts, tau, shares):
    """Sensitivity rule: probabilities that give transfer share ``tau`` with the
    Rescue reason mix. With N L2/3 starters,

        t_l23_l45_predicted = tau x w_labour       x N / women eligible for predicted transfer
        t_l23_l45_preterm   = tau x w_preterm      x N / E_pt
        t_l23_l45_notsevere = (tau x w_complication x N - severe transfers) / E_ns

    KHIS L2/3 preterm and the L2/3:L4 complication ratio then become checks.
    """
    n = counts["l23_starters"]
    eligible = counts["predicted_transfers"] / current_predicted
    raw = (
        tau * shares["preterm"] * n / counts["E_pt"],
        (tau * shares["maternal_complication"] * n - counts["severe_transfers"]) / counts["E_ns"],
        tau * shares["labour_emergency"] * n / eligible,
    )
    clipped = tuple(float(np.clip(value, 0.0, 1.0)) for value in raw)
    return clipped, clipped != raw


def updated_predicted(current, counts, t_preterm, t_notsevere, share):
    """Rescale t_l23_l45_predicted so its transfers are ``share`` of all L2/3 transfers."""
    if current <= 0 or counts["predicted_transfers"] <= 0:
        raise ValueError("Cannot rescale t_l23_l45_predicted from a zero probability or count")
    others = counts["severe_transfers"] + counts["E_pt"] * t_preterm + counts["E_ns"] * t_notsevere
    target = share / (1 - share) * others
    eligible = counts["predicted_transfers"] / current
    return min(1.0, target / eligible)


def solve(counts, targets, start=(0.5, 0.5)):
    """Probabilities (p_pt, p_ns) in [0, 1] that best match the two targets."""
    fit = least_squares(
        lambda x: residuals(predicted(counts, *x), targets), start, bounds=(0.0, 1.0),
    )
    at_bound = bool(np.any((fit.x <= 1e-6) | (fit.x >= 1 - 1e-6)))
    return float(fit.x[0]), float(fit.x[1]), at_bound
