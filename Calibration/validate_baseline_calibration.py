"""Validate the baseline model against the calibration targets (10 runs x 60 months).

Validates the workbook as is, or with the values from a recalibration
(--values values.csv) applied.
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
    CALIBRATED_COUNTIES, DEFAULTS, county_targets, khis_mmr, read_values, simulate, summarize,
)
from Calibration.calibration_data import KHIS, SCALE_COMP
from global_func import reset_flags


# Checks, not fitted: the DHS location by ANC (routing fits overall home and L2/3),
# KHIS preterm at L4 (Stage 1 fits all facility births), the total complication
# rate (its six causes are fitted), severe maternal outcomes, and county planned CS
# among preterm (Stage 4 fits it pooled over the counties).
CHECKS = {"final_home_given_anc", "final_l23_given_anc", "final_l45_given_anc", "preterm_l4", "mcr_l4", "smo_fac",
          "planned_cs_preterm"}
# Model shares implied by routing (no observed target).
DIAGNOSTICS = ("initial_l23", "initial_l45", "emergency_cs_l4", "transfer_rate_l23", "referral_share_l45")


def target_role(metric):
    return "validation_only" if metric in CHECKS else "fit_target"


def row(county, metric, modeled, target, role, **extra):
    return {"county": county, "metric": metric, "modeled": modeled, "target": target,
            "relative_error": (modeled - target) / target if target else np.nan, "target_role": role, **extra}


def fit_status(relative_error):
    error = abs(relative_error)
    if error <= 0.10:
        return "accepted_within_10pct"
    if error <= 0.20:
        return "review_10_to_20pct"
    return "outside_20pct"


def simulate_county(county, values, runs, months, base_seed, flags):
    """Mean over runs of each run's model values. Simulation seeds differ from the
    calibration's (base_seed x 1000), so the fit is checked on fresh draws."""
    outputs = simulate(county, values, runs, months, base_seed * 1000, flags)
    return pd.DataFrame([summarize(people) for people in outputs]).mean(numeric_only=True)


def county_rows(county, modeled, cause_mix):
    """Targets and checks, KHIS mortality, county cause rates, and diagnostics for one county."""
    rows = [row(county, metric, float(modeled[metric]), target, target_role(metric))
            for metric, target in county_targets(county, cause_mix).items()]
    # KHIS maternal deaths (direct, non-abortion share) with exact Poisson 95% intervals.
    for level in ("l23", "l4"):
        target, lower, upper = khis_mmr([county], level)
        value = float(modeled[f"mmr_{level}_expected"])
        rows.append(row(county, f"mmr_{level}_expected", value, target, "validation_only",
                        ci_lower=lower, ci_upper=upper, within_95ci=lower <= value <= upper))
    # County cause-specific KHIS rates: checks, since the fitted targets use the pooled cause mix.
    if cause_mix == "pooled":
        khis = KHIS[county]
        for cause, recorded in khis["complications_l4"].items():
            target = SCALE_COMP * recorded / khis["deliveries"]["l4"]
            rows.append(row(county, f"{cause}_l4_khis_county", float(modeled[f"{cause}_l4"]), target,
                            "validation_only"))
    rows += [row(county, metric, float(modeled[metric]), np.nan, "diagnostic_only") for metric in DIAGNOSTICS]
    return rows


def pooled_rows(summary):
    """Targets fitted pooled over the counties: planned CS among preterm women starting
    at L4/5 (Stage 4) and expected MMR at L2/3 and L4 (Stage 6)."""
    planned = summary["planned_cs_preterm_count"].sum() / summary["preterm_l45_starters"].sum()
    rows = [row("pooled", "planned_cs_preterm", planned,
                county_targets(CALIBRATED_COUNTIES[0])["planned_cs_preterm"], "fit_target_pooled")]
    for level in ("l23", "l4"):
        mmr = 1e5 * summary[f"deaths_{level}_expected"].sum() / summary[f"births_{level}"].sum()
        rows.append(row("pooled", f"mmr_{level}_expected", mmr, khis_mmr(CALIBRATED_COUNTIES, level)[0],
                        "fit_target_pooled"))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=DEFAULTS["validation_runs"])
    parser.add_argument("--months", type=int, default=DEFAULTS["validation_months"])
    parser.add_argument("--base-seed", type=int, default=DEFAULTS["simulation_seed"])
    parser.add_argument("--values", help="Calibrated values (values.csv) applied to the workbook.")
    parser.add_argument("--cause-mix", choices=("county", "pooled"), default="pooled")
    parser.add_argument("--facility-capacity-limit", action="store_true",
                        help="Enforce the monthly L4/5 Capacity limit (off by default).")
    parser.add_argument("--output", default="Calibration/results/baseline_validation.csv")
    args = parser.parse_args()

    values = read_values(args.values) if args.values else {}
    flags = {**reset_flags(), "flag_facility_capacity_limit": int(args.facility_capacity_limit)}
    summary = pd.DataFrame({
        county: simulate_county(county, values.get(county, {}), args.runs, args.months, args.base_seed, flags)
        for county in CALIBRATED_COUNTIES
    }).T
    rows = [r for county in CALIBRATED_COUNTIES for r in county_rows(county, summary.loc[county], args.cause_mix)]
    result = pd.DataFrame(rows + pooled_rows(summary))
    result["status"] = result["relative_error"].map(lambda e: fit_status(e) if pd.notna(e) else "not_targeted")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved: {output.resolve()}")


if __name__ == "__main__":
    main()
