"""Stage 2: closed-form ANC recognition and initial delivery routing.

Targets are final (post-transfer) locations. Initial L2/3 is recovered from
final L2/3 with the transfer share tau = P(transfer | initial L2/3); home
deliveries are unaffected by transfer.

Symbols (as in the methods document) and workbook names:
    anc         4+ ANC coverage, a_c                          p_ANC_base
    r           P(recognized high risk | 4+ ANC), r_c          (computed: recognition_rate)
    b           L2/3 preference among facility births, b_c    baseline_l23_preference
    theta       odds ratio for L4/5 if recognized             OR_recognized_L45
    home_noanc  P(home | no 4+ ANC), h_0                      home_noANC
    home_anc    final P(home | 4+ ANC), F_H|ANC (DHS)
    home_unrecognized  P(home | 4+ ANC, not recognized), h_U  home_lowrisk
    tau         P(transfer | start at L2/3), tau_c            (from Stage 3)

Initial location (recognized women never deliver at home):
    no ANC:              home h_0,  L2/3 (1 - h_0) b
    ANC, unrecognized:   home h_U,  L2/3 (1 - h_U) b
    ANC, recognized:     home 0,    L2/3 b / (b + theta (1 - b))
The remainder starts at L4/5.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def recognized_risk_probability(prevalence, sensitivity, specificity):
    return prevalence * sensitivity + (1.0 - prevalence) * (1.0 - specificity)


def recognized_l45_probability(baseline_l23_preference, recognized_l45_or):
    b = baseline_l23_preference
    theta = recognized_l45_or
    return theta * (1.0 - b) / (b + theta * (1.0 - b))


def recognition_rate(param):
    """P(recognized risk | 4+ ANC) under baseline LMP dating and risk screening.

    Mirrors f_ANC_LB_effect_vectorized: risk screening is independent of
    gestational age, and preterm and post-term predictions use separate draws.
    ``param`` must already include derived ``GA_anc`` (after PT_scale).
    """
    screen = recognized_risk_probability(
        param["p_highrisk"], param["sen_risk_trad"], param["spec_risk_trad"]
    )
    ga = np.asarray(param["GA_sequence"])
    term_status = np.select([ga < 37, ga >= 43], [0, 2], default=1)
    preterm = np.append(param["E_Preterm_LMP"][:2], 1.0 - sum(param["E_Preterm_LMP"][:2]))
    postterm = np.append(param["E_Postterm_LMP"][:2], 1.0 - sum(param["E_Postterm_LMP"][:2]))
    flagged = 1.0 - (1.0 - preterm[term_status]) * (1.0 - postterm[term_status])
    gestational = float(np.sum(np.asarray(param["GA_anc"]) * flagged))
    return 1.0 - (1.0 - screen) * (1.0 - gestational)


def anc_initial_l23(b, theta, r, home_unrecognized):
    """Initial P(L2/3 | 4+ ANC)."""
    return (1.0 - r) * (1.0 - home_unrecognized) * b + r * (1.0 - recognized_l45_probability(b, theta))


def overall_initial_l23(b, theta, r, anc, home_noanc, home_unrecognized):
    return (
        (1.0 - anc) * (1.0 - home_noanc) * b
        + anc * anc_initial_l23(b, theta, r, home_unrecognized)
    )


def solve_county(anc, home_noanc, home_anc, final_l23, r, theta, tau):
    """Solve county routing so final home by ANC and final overall L2/3 are exact.

    ``final_l23`` is the share of all births delivered at L2/3 after transfer.
    Returns the workbook parameters and the implied initial and final shares.
    """
    home_unrecognized = home_anc / (1.0 - r)
    if not 0.0 <= home_unrecognized <= 1.0:
        raise ValueError(f"home_lowrisk = {home_unrecognized:.4f} lies outside [0, 1]")
    home = (1.0 - anc) * home_noanc + anc * home_anc
    initial_l23 = final_l23 / (1.0 - tau)
    if initial_l23 >= 1.0 - home:
        raise ValueError("Initial L2/3 implied by tau exceeds facility deliveries")

    b = brentq(
        lambda value: overall_initial_l23(value, theta, r, anc, home_noanc, home_unrecognized)
        - initial_l23,
        1e-12, 1.0 - 1e-12,
    )
    anc_l23 = anc_initial_l23(b, theta, r, home_unrecognized)
    noanc_l23 = (1.0 - home_noanc) * b
    return {
        "home_noANC": home_noanc,
        "home_lowrisk": home_unrecognized,
        "baseline_l23_preference": b,
        "OR_recognized_L45": theta,
        "recognition_rate": r,
        "initial_home": home,
        "initial_l23": initial_l23,
        "initial_l45": 1.0 - home - initial_l23,
        "final_l23": final_l23,
        "final_l45": 1.0 - home - final_l23,
        "p_lb_l23_45": tau * initial_l23,
        "final_l23_given_anc": (1.0 - tau) * anc_l23,
        "final_l23_given_noanc": (1.0 - tau) * noanc_l23,
    }
