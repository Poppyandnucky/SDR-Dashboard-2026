"""Do intervention effects depend on a calibration assumption (tau, theta, ...)?

Runs the Transfer & EMT, PROMPTS, blood tracking, single-treatment, and combined
scenarios from
scenario_comparison.py for several calibrations (values.csv from run_calibration.py
recalibrate, e.g. the main run and a tau = 0.25 or OR_recognized_L45 = 2.5 sensitivity;
an empty path means the workbook as is) with the same parameter and monthly seeds, and reports each scenario's change in
maternal mortality relative to its base scenario. Mortality uses expected deaths
(sum of individual death probabilities), which has less Monte Carlo noise.

Usage:
    python Calibration/run_tau_intervention_sensitivity.py \
        --values current= --values tau25=Calibration/results/<run>/values.csv \
        --output tau_intervention.csv
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Calibration.calibration_common import CALIBRATED_COUNTIES, apply_values, read_values
from global_func import reset_E, reset_HSS, reset_S, reset_flags
from model_run import run_model_dash
from parameter_loader import calculate_derived_parameters, get_parameters, get_slider_params
from scenario_comparison import BASE_SEED, ENABLED_SINGLE_INTERVENTION_SUPPLY, configure_scenario


SCENARIOS = {
    "referral_base": "referral_base", "referral_moderate": "referral_base", "referral_high": "referral_base",
    "prompts_base": "prompts_base", "prompts_moderate": "prompts_base", "prompts_high": "prompts_base",
    "blood_base": "blood_base", "blood_high": "blood_base",
    "single_pph_bundle": "referral_base", "single_MgSO4": "referral_base", "single_antibiotics": "referral_base",
    "combined_base": "combined_base", "combined_moderate": "combined_base", "combined_high": "combined_base",
}


# One treatment on alone at the dashboard's enabled supply (as its single toggles do).
SINGLE = {"single_pph_bundle": ("flag_pph_bundle", "pph_bundle"),
          "single_MgSO4": ("flag_MgSO4", "MgSO4"),
          "single_antibiotics": ("flag_antibiotics", "antibiotics")}


def summarize(people):
    facility = people["i_loc_new_v2"] > 0
    return {
        "transfer_rate": float((people["i_transfer"] == 1).mean()),
        "mmr_expected": 1e5 * float(people["p_mat_death"].mean()),
        "facility_mmr_expected": 1e5 * float(people.loc[facility, "p_mat_death"].mean()),
    }


def base_of(scenario):
    return SCENARIOS.get(scenario, scenario.rsplit("_", 1)[0] + "_base")


def run_county(task):
    label, values, county, runs, months, scenarios = task
    values = read_values(values).get(county, {}) if values else {}
    sliders = get_slider_params(county=county)
    rows = []
    for run in range(runs):
        seed = BASE_SEED + run
        month_seeds = np.arange(seed, seed + months, dtype=int)
        for scenario in scenarios:
            param = calculate_derived_parameters(apply_values(get_parameters(county=county, seed=seed), values))
            param.update({"E": reset_E(), "S": reset_S(sliders), "HSS": reset_HSS(sliders)})
            flags = reset_flags()
            if scenario in SINGLE:
                flag, supply = SINGLE[scenario]
                flags[flag] = 1
                param["S"][supply] = ENABLED_SINGLE_INTERVENTION_SUPPLY
            else:
                configure_scenario(scenario, param, flags, sliders)
            _, people, _ = run_model_dash(param, flags, n_months=months, int_period=months, base_seed=month_seeds)
            rows.append({"calibration": label, "county": county, "run": run, "scenario": scenario, **summarize(people)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--values", action="append", required=True,
                        help="label=values.csv (empty path: the workbook as is); repeat for each calibration.")
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--months", type=int, default=36)
    parser.add_argument("--output", default="Calibration/results/tau_intervention_sensitivity.csv")
    parser.add_argument("--scenarios", help="Comma-separated scenario names (default: the set above); "
                        "each is compared with <intervention>_base.")
    args = parser.parse_args()

    scenarios = args.scenarios.split(",") if args.scenarios else list(SCENARIOS)
    scenarios = list(dict.fromkeys([base_of(name) for name in scenarios] + scenarios))
    tasks = [(label, str(Path(path).resolve()) if path else None, county, args.runs, args.months, scenarios)
             for label, path in (item.split("=", 1) for item in args.values)
             for county in CALIBRATED_COUNTIES]
    with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
        frame = pd.DataFrame([row for rows in pool.map(run_county, tasks) for row in rows])

    # Paired change against the scenario's base (same seeds).
    keys = ["calibration", "county", "run"]
    base = frame.set_index(keys + ["scenario"])
    for metric in ("mmr_expected", "facility_mmr_expected"):
        frame[f"{metric}_change"] = [
            row[metric] - base.loc[(row.calibration, row.county, row.run, base_of(row.scenario)), metric]
            for _, row in frame.iterrows()
        ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    summary = frame.groupby(["county", "scenario", "calibration"]).agg(
        transfer_rate=("transfer_rate", "mean"),
        mmr=("mmr_expected", "mean"),
        change=("mmr_expected_change", "mean"),
        change_se=("mmr_expected_change", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
    )
    summary["relative_change"] = summary["change"] / (summary["mmr"] - summary["change"])
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string())
    print(f"\nSaved: {output.resolve()}")


if __name__ == "__main__":
    main()
