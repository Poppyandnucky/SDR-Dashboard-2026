"""Illustrative sensitivity: reported L5 MMR under unobserved external referrals.

L5 hospitals also deliver women referred from outside the county, who are not in
the model. Their number and acuity are not observed, so this does not estimate
them. For a grid of assumed external shares s of L5 deliveries and late-referral
odds ratios, it reports the L5 MMR the facility would record:

    reported L5 MMR = (1 - s) x internal L5 MMR + s x proxy external MMR

The proxy is the model's complication-bearing emergency transfers received at L5,
with their death probabilities raised by the odds ratio (for late arrival and
acuity). Nothing in the model is changed. The observed L5 MMR (KHIS, scaled to
direct, non-abortion deaths, with its exact 95% interval) marks which part of the
grid is consistent with the data; many combinations are.

Usage:
    python Calibration/run_external_l5_sensitivity.py [--values Calibration/results/<run>/values.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Calibration.calibration_common import (
    CALIBRATED_COUNTIES, CAUSE_COLUMNS, DEFAULTS, direct_mmr, read_values, simulate,
)
from Calibration.calibration_data import KHIS_L5


def apply_odds_ratio(probability, odds_ratio):
    """Probabilities with their odds multiplied by odds_ratio (at least 1)."""
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    if odds_ratio < 1:
        raise ValueError("late_referral_rr must be at least 1")
    return odds_ratio * probability / (1.0 - probability + odds_ratio * probability)


def l5_mortality_reporting_adjustment(people, external_share, late_referral_rr):
    """Internal, proxy external, and reported L5 MMR (per 100,000) for one assumed
    external share and late-referral odds ratio; `people` is not modified."""
    if not 0 <= external_share < 1:
        raise ValueError("external_share must be in [0, 1)")
    final_l5 = people["i_loc_new_v2"].eq(3)
    transferred = people["i_transfer_pred"].eq(1) | people["i_transfer_actual"].eq(1)
    complication = people[list(CAUSE_COLUMNS.values())].eq(1).any(axis=1)
    reference = people[final_l5 & transferred & complication]
    internal = float(people.loc[final_l5, "p_mat_death"].mean())
    external = float(apply_odds_ratio(reference["p_mat_death"].to_numpy(), late_referral_rr).mean())
    reported = (1 - external_share) * internal + external_share * external
    return {
        "internal_l5_mmr": 1e5 * internal,
        "proxy_external_mmr": 1e5 * external,
        "reported_l5_mmr": 1e5 * reported,
        "external_share_of_reported_deaths": external_share * external / reported,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--values", help="Calibrated values (values.csv) applied to the workbook.")
    parser.add_argument("--shares", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
    parser.add_argument("--risk-ratios", nargs="+", type=float, default=[1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--runs", type=int, default=DEFAULTS["validation_runs"])
    parser.add_argument("--months", type=int, default=DEFAULTS["validation_months"])
    parser.add_argument("--base-seed", type=int, default=DEFAULTS["simulation_seed"])
    parser.add_argument("--output", default="Calibration/results/external_l5_sensitivity.csv")
    args = parser.parse_args()

    values = read_values(args.values) if args.values else {}
    rows = []
    for county in CALIBRATED_COUNTIES:
        # The same runs and simulation seeds as validation.
        people = pd.concat(simulate(county, values.get(county, {}), args.runs, args.months, args.base_seed * 1000),
                           ignore_index=True)
        observed, lower, upper = direct_mmr(KHIS_L5[county]["deaths"], KHIS_L5[county]["deliveries"])
        for share in args.shares:
            for rr in args.risk_ratios:
                adjusted = l5_mortality_reporting_adjustment(people, share, rr)
                rows.append({
                    "county": county, "external_share": share, "late_referral_rr": rr, **adjusted,
                    "observed_l5_mmr": observed, "observed_lower": lower, "observed_upper": upper,
                    "within_observed_95ci": lower <= adjusted["reported_l5_mmr"] <= upper,
                })
    result = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    for county, grid in result.groupby("county", sort=False):
        first = grid.iloc[0]
        print(f"\n{county}: reported L5 MMR per 100,000 (rows: external share; columns: late-referral "
              f"odds ratio); observed {first.observed_l5_mmr:.0f} ({first.observed_lower:.0f}-"
              f"{first.observed_upper:.0f}); * within the observed interval")
        table = grid.assign(cell=[f"{m:.0f}{'*' if ok else ''}" for m, ok in
                                  zip(grid.reported_l5_mmr, grid.within_observed_95ci)])
        print(table.pivot(index="external_share", columns="late_referral_rr", values="cell").to_string())
    print(f"\nSaved: {output.resolve()}")


if __name__ == "__main__":
    main()
