"""Illustrative sensitivity: reported L5 MMR under unobserved cross-county referrals.

The model has no cross-county referrals. Their number and acuity are not observed,
so this does not estimate them. For a grid of assumed shares s and odds ratios,
it reports the L5 MMR the facility would record. Kisii and Mombasa L5 hospitals
receive women from outside the county (in-referral, share s of L5 deliveries):

    reported L5 MMR = (1 - s) x internal L5 MMR + s x proxy MMR

The proxy is the model's complication-bearing emergency transfers received at L5,
with their death probabilities raised by the odds ratio (for late arrival and
acuity). Makueni's modeled L5 MMR is above the observed value, so in-referral
cannot match it; one possible explanation (a hypothesis, not documented) is that
Makueni refers its most complicated cases out, for example to Machakos Level 5 or
Kenyatta National Hospital, where their deaths are recorded. For Makueni a share
s of the modeled L5 deliveries is removed from the proxy group, at its average
risk or highest risk first, with their modeled deaths:

    reported L5 MMR = (internal L5 deaths - deaths of the women referred out)
                      / (L5 deliveries - women referred out)

Nothing in the model is changed. The observed L5 MMR (KHIS, scaled to
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

# Counties treated as referring complicated cases out of the county (a hypothesis).
OUT_REFERRAL = ("makueni",)


def apply_odds_ratio(probability, odds_ratio):
    """Probabilities with their odds multiplied by odds_ratio (at least 1)."""
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    if odds_ratio < 1:
        raise ValueError("late_referral_rr must be at least 1")
    return odds_ratio * probability / (1.0 - probability + odds_ratio * probability)


def l5_groups(people):
    """Final-L5 deliveries and the proxy group: complication-bearing emergency
    transfers received at L5."""
    final_l5 = people["i_loc_new_v2"].eq(3)
    transferred = people["i_transfer_pred"].eq(1) | people["i_transfer_actual"].eq(1)
    complication = people[list(CAUSE_COLUMNS.values())].eq(1).any(axis=1)
    return final_l5, people[final_l5 & transferred & complication]


def l5_mortality_reporting_adjustment(people, external_share, late_referral_rr):
    """In-referral: internal, proxy, and reported L5 MMR (per 100,000) for one assumed
    external share and late-referral odds ratio; `people` is not modified."""
    if not 0 <= external_share < 1:
        raise ValueError("external_share must be in [0, 1)")
    final_l5, reference = l5_groups(people)
    internal = float(people.loc[final_l5, "p_mat_death"].mean())
    external = float(apply_odds_ratio(reference["p_mat_death"].to_numpy(), late_referral_rr).mean())
    return {
        "internal_l5_mmr": 1e5 * internal,
        "proxy_external_mmr": 1e5 * external,
        "reported_l5_mmr": 1e5 * ((1 - external_share) * internal + external_share * external),
    }


def l5_out_referral_adjustment(people, out_share):
    """Out-referral: reported L5 MMR (per 100,000) if a share of the modeled L5
    deliveries leaves the county, taken from the proxy group at its average risk or
    highest risk first. Their modeled deaths are then recorded elsewhere. NaN if the
    share exceeds the proxy group; `people` is not modified."""
    final_l5, reference = l5_groups(people)
    n_l5 = int(final_l5.sum())
    n_out = int(round(out_share * n_l5))
    risk = np.sort(reference["p_mat_death"].to_numpy())[::-1]
    total = float(people.loc[final_l5, "p_mat_death"].sum())
    if n_out > len(risk):
        return {"proxy average": np.nan, "highest risk first": np.nan}
    return {
        "proxy average": 1e5 * (total - n_out * risk.mean()) / (n_l5 - n_out),
        "highest risk first": 1e5 * (total - risk[:n_out].sum()) / (n_l5 - n_out),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--values", help="Calibrated values (values.csv) applied to the workbook.")
    parser.add_argument("--shares", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
                        help="In-referral shares of L5 deliveries.")
    parser.add_argument("--out-shares", nargs="+", type=float, default=[0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15],
                        help="Out-referral shares of modeled L5 deliveries (OUT_REFERRAL counties).")
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
        observed_values = {"observed_l5_mmr": observed, "observed_lower": lower, "observed_upper": upper}
        if county in OUT_REFERRAL:
            cells = [("out", share, assumption, mmr) for share in args.out_shares
                     for assumption, mmr in l5_out_referral_adjustment(people, share).items()]
        else:
            cells = [("in", share, f"odds ratio {rr:g}", l5_mortality_reporting_adjustment(people, share, rr)["reported_l5_mmr"])
                     for share in args.shares for rr in args.risk_ratios]
        rows += [{"county": county, "direction": direction, "share": share, "assumption": assumption,
                  "reported_l5_mmr": mmr, **observed_values, "within_observed_95ci": lower <= mmr <= upper}
                 for direction, share, assumption, mmr in cells]
    result = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    for county, grid in result.groupby("county", sort=False):
        first = grid.iloc[0]
        print(f"\n{county}: reported L5 MMR per 100,000 (rows: {first.direction}-referral share); observed {first.observed_l5_mmr:.0f} ({first.observed_lower:.0f}-"
              f"{first.observed_upper:.0f}); * within the observed interval; - share exceeds the proxy group")
        table = grid.assign(cell=["-" if pd.isna(m) else f"{m:.0f}{'*' if ok else ''}" for m, ok in
                                  zip(grid.reported_l5_mmr, grid.within_observed_95ci)])
        print(table.pivot(index="share", columns="assumption", values="cell")[grid.assumption.unique()].to_string())
    print(f"\nSaved: {output.resolve()}")


if __name__ == "__main__":
    main()
