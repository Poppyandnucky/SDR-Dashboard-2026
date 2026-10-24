"""Shared pieces of the calibration: settings, calibrated values, targets, and the
model values compared with them (summarize).

Calibrated values are held as {county: {(parameter, index): value}}; index is None
for scalars and the array position otherwise (e.g. p_cs_capacity, base_LB). They are
applied to each parameter draw in memory; the workbook is only read.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

import parameter_loader
from Calibration.calibration_data import (
    DHS, DIRECT_CAUSE_SHARE, KHIS, KHIS_2022_DELIVERIES, PLANNED_CS_SHARE, PRETERM_FAC, PRETERM_PLANNED_CS,
    SCALE_COMP, SMO_REFERENCE,
)
from global_func import reset_E, reset_HSS, reset_S, reset_flags
from model_run import run_model_dash
from parameter_loader import calculate_derived_parameters, get_parameters, get_slider_params

REPO_ROOT = Path(__file__).resolve().parents[1]
# The model reads SDR Parameters.xlsx in the repository root or data/ (another
# workbook: set SDR_PARAMS_PATH).
_WORKBOOK = next((path for path in (REPO_ROOT / "SDR Parameters.xlsx", REPO_ROOT / "data" / "SDR Parameters.xlsx")
                  if path.exists()), REPO_ROOT / "SDR Parameters.xlsx")
parameter_loader.WORKBOOK_PATH = Path(os.environ.get("SDR_PARAMS_PATH", _WORKBOOK))

# Kakamega keeps its existing values.
CALIBRATED_COUNTIES = ("kisii", "makueni", "mombasa")

# Run r uses parameter draw seed parameter_seed + r, as scenario runs do.
DEFAULTS = {
    "parameter_seed": 42,
    "simulation_seed": 20261201,
    "loop_runs": 10,
    "loop_months": 24,
    "validation_runs": 10,
    "validation_months": 60,
}

# Workbook sheet of each calibrated parameter (all others are in county_calibrated).
SHEETS = {"p_ANC_base": "county_demographics", "base_LB": "county_supply"}
# Complication -> its indicator column in the model output.
CAUSE_COLUMNS = {
    "aph": "i_aph", "pph": "i_pph_new", "eclampsia": "i_eclampsia_new",
    "ol": "i_OL_final", "rupture": "i_ruptured_uterus", "sepsis": "i_mat_sepsis_new",
}
CAUSES = tuple(CAUSE_COLUMNS)


# ---- Calibrated values ------------------------------------------------------

def parameter_draw(county, run=0):
    """The workbook parameters for one run (seed parameter_seed + run)."""
    return get_parameters(county=county, rng=np.random.default_rng(DEFAULTS["parameter_seed"] + run))


def apply_values(param, values):
    """Set calibrated values on one parameter draw, before calculate_derived_parameters."""
    for (name, index), value in values.items():
        if index is None:
            param[name] = value
        else:
            param[name] = np.array(param[name], dtype=float)
            param[name][index] = value
    # Quantities parameter_loader derives from base_LB when the workbook is read.
    base_lb = np.asarray(param["base_LB"], dtype=float)
    param["base_p_45"] = float((base_lb[2] + base_lb[3]) / base_lb.sum())
    param["p_l5_l45"] = float(base_lb[3] / (base_lb[2] + base_lb[3]))
    param["Num_Exp_L45"] = float(base_lb[2] + base_lb[3])
    return param


def simulate(county, values, runs, months, base_seed, flags=None):
    """One simulation per run with the calibrated values applied; returns each run's
    women-level output. Run r uses parameter draw parameter_seed + r and monthly
    seeds base_seed + r x months onwards; flags default to the baseline."""
    sliders = get_slider_params(county=county)
    outputs = []
    for run in range(runs):
        param = calculate_derived_parameters(apply_values(copy.deepcopy(parameter_draw(county, run)), values))
        param.update({"E": reset_E(), "S": reset_S(sliders), "HSS": reset_HSS(sliders)})
        seeds = np.arange(base_seed + run * months, base_seed + (run + 1) * months, dtype=int)
        run_flags = dict(flags) if flags is not None else reset_flags()
        _, people, _ = run_model_dash(param, run_flags, months, months, seeds)
        outputs.append(people)
    return outputs


def save_values(values, path):
    """values.csv: sheet, parameter, county, index, value, and the current workbook value."""
    rows = []
    for county, county_values in values.items():
        draw = parameter_draw(county)
        for (name, index), value in county_values.items():
            workbook = np.nan if name not in draw else draw[name] if index is None else np.asarray(draw[name])[index]
            rows.append({"sheet": SHEETS.get(name, "county_calibrated"), "parameter": name,
                         "county": county, "index": index, "value": value, "workbook_value": float(workbook)})
    pd.DataFrame(rows).to_csv(path, index=False)


def read_values(path):
    """values.csv -> {county: {(parameter, index): value}}."""
    values = {}
    for row in pd.read_csv(path).itertuples():
        index = None if pd.isna(row.index) else int(row.index)
        values.setdefault(row.county, {})[(row.parameter, index)] = float(row.value)
    return values


# ---- Targets ---------------------------------------------------------------

def observed_locations(county):
    """Final delivery shares: home (DHS, by ANC), L2/3 (KHIS 2022 share of facility
    births), and the L5 share of L4/5 births."""
    dhs = DHS[county]
    l2, l3, l4, l5 = KHIS_2022_DELIVERIES[county]
    home = (1 - dhs["anc4"]) * dhs["home_noanc"] + dhs["anc4"] * dhs["home_anc"]
    return home, (l2 + l3) / (l2 + l3 + l4 + l5) * (1 - home), l5 / (l4 + l5)


def county_targets(county, cause_mix="pooled"):
    """Calibration targets for one county, followed by its validation checks.

    The six complication targets are the county's total recorded rate split by the
    cause mix pooled over the calibrated counties (cause_mix="county": its own mix).
    """
    khis, dhs = KHIS[county], DHS[county]
    rate = lambda name, level: khis[name][level] / khis["deliveries"][level]
    complications_l4 = sum(khis["complications_l4"].values()) / khis["deliveries"]["l4"]
    pool = CALIBRATED_COUNTIES if cause_mix == "pooled" else (county,)
    cases = {cause: sum(KHIS[c]["complications_l4"][cause] for c in pool) for cause in CAUSES}
    home, final_l23, _ = observed_locations(county)
    return {
        "initial_home": home,
        "final_l23": final_l23,
        "final_l45": 1 - home - final_l23,
        "preterm_fac": PRETERM_FAC,
        "preterm_l23": SCALE_COMP * rate("preterm", "l23"),
        "cs_l23": rate("cs", "l23"),
        "mcr_ratio_l23_l4": (khis["complications_l23"] / khis["deliveries"]["l23"]) / complications_l4,
        **{f"{cause}_l4": SCALE_COMP * complications_l4 * cases[cause] / sum(cases.values()) for cause in CAUSES},
        "cs_l4": rate("cs", "l4"),
        "elective_cs_l4": PLANNED_CS_SHARE * rate("cs", "l4"),
        "planned_cs_preterm": PRETERM_PLANNED_CS,
        # Validation checks.
        "final_home_given_anc": dhs["home_anc"],
        "final_l23_given_anc": dhs["l23_anc"],
        "final_l45_given_anc": dhs["l45_anc"],
        "preterm_l4": SCALE_COMP * rate("preterm", "l4"),
        "mcr_l4": SCALE_COMP * complications_l4,
        "smo_fac": SMO_REFERENCE[county],
    }


def direct_mmr(deaths, deliveries):
    """Maternal deaths per 100,000 deliveries scaled to direct, non-abortion deaths
    (the deaths the model represents), with an exact Poisson 95% interval."""
    scale = DIRECT_CAUSE_SHARE * 1e5 / deliveries
    lower = chi2.ppf(0.025, 2 * deaths) / 2 if deaths else 0.0
    return DIRECT_CAUSE_SHARE * 1e5 * deaths / deliveries, lower * scale, chi2.ppf(0.975, 2 * deaths + 2) / 2 * scale


def khis_mmr(counties, level):
    """direct_mmr of KHIS deaths at `level` ("l23" or "l4"), pooled over `counties`."""
    return direct_mmr(sum(KHIS[c]["deaths"][level] for c in counties),
                      sum(KHIS[c]["deliveries"][level] for c in counties))


# ---- Model values ----------------------------------------------------------

def mean(values, mask):
    return float(values[mask].mean()) if mask.any() else np.nan


def summarize(people):
    """Model values of every target and check, from one (or several pooled) simulations."""
    initial = people["i_loc"]
    anc = people["i_ANC"] == 1
    final = people["i_loc_new_v2"]
    l23 = final == 1
    l4 = final == 2
    l45 = final >= 2
    cs = people["i_mod"].isin(["ELCS", "EmCS"])
    event_count = sum((people[column] == 1).astype(int) for column in CAUSE_COLUMNS.values())
    mcr_l23 = mean(event_count, l23)
    death = people["i_mat_death"] == 1
    facility = final > 0
    smo = (people["i_severe_new"] == 1) | death
    preterm_l45_starters = (initial >= 2) & (people["i_preterm"] == 1)
    return {
        "initial_home": float((initial == 0).mean()),
        "initial_l23": float((initial == 1).mean()),
        "initial_l45": float((initial >= 2).mean()),
        "final_home_given_anc": mean(final == 0, anc),
        "final_l23_given_anc": mean(final == 1, anc),
        "final_l45_given_anc": mean(final >= 2, anc),
        "final_l23": float(l23.mean()),
        "final_l45": float(l45.mean()),
        "preterm_l23": mean(people["i_preterm"] == 1, l23),
        "cs_l23": mean(cs, l23),
        "mcr_ratio_l23_l4": mcr_l23 / mean(event_count, l4) if mean(event_count, l4) else np.nan,
        **{f"{cause}_l4": mean(people[column] == 1, l4) for cause, column in CAUSE_COLUMNS.items()},
        "cs_l4": mean(cs, l4),
        "elective_cs_l4": mean(people["i_mod"] == "ELCS", l4),
        "planned_cs_preterm": mean(people["i_mod"] == "ELCS", preterm_l45_starters),
        "emergency_cs_l4": mean(people["i_mod"] == "EmCS", l4),
        "mcr_l4": mean(event_count, l4),
        "smo_fac": mean(smo, facility),
        # Expected deaths (sum of individual death probabilities): less Monte Carlo noise.
        "mmr_l23_expected": mean(people["p_mat_death"], l23) * 100_000,
        "mmr_l4_expected": mean(people["p_mat_death"], l4) * 100_000,
        "deaths_l23_expected": float(people.loc[l23, "p_mat_death"].sum()),
        "deaths_l4_expected": float(people.loc[l4, "p_mat_death"].sum()),
        "births_l23": int(l23.sum()),
        "births_l4": int(l4.sum()),
        "preterm_l45_starters": int(preterm_l45_starters.sum()),
        "planned_cs_preterm_count": int(((people["i_mod"] == "ELCS") & preterm_l45_starters).sum()),
        "preterm_l4": mean(people["i_preterm"] == 1, l4),
        "preterm_fac": mean(people["i_preterm"] == 1, facility),
        "transfer_rate_l23": mean(final >= 2, initial == 1),
        "referral_share_l45": mean(initial == 1, l45),
    }
