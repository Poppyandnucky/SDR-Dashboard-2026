"""Stage 6: re-fit the two shared mortality anchors to pooled KHIS deaths.

Fitted (one value shared by the calibrated counties):
    p_mat_death_facility_baseline  to pooled internal-L4 MMR
    RR_MM_l23_vs_l4                to pooled L2/3 MMR
Targets are KHIS maternal deaths per 100,000 deliveries pooled over Kisii,
Makueni, and Mombasa (calibration_data.KHIS), times the direct, non-abortion
share of maternal deaths (0.618), since the model has no indirect deaths. The
cause weights (MM_weight_*), p_MM_others, and weight_facility_mat keep their values.

Mortality does not feed back into Stages 1-5, so this runs once on the Stage 1-5
values (values.csv in the loop output folder). Modeled MMR uses expected deaths
(the sum of individual death probabilities) over the same parameter draws as the
loop. Update (methods document, closed form):

    A   = T_L4 / M_L4
    p_mat_death_facility_baseline <- p_mat_death_facility_baseline x A
    RR_MM_l23_vs_l4               <- RR_MM_l23_vs_l4 x (T_L23 / M_L23) / A

Both values are added to values.csv for each calibrated county; in the workbook
they go in county_calibrated, overriding the constants that Kakamega keeps.

Usage:
    python Calibration/calibrate_stage6_mortality.py --stage-dir Calibration/results/<run>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Calibration.calibration_common import (
    CALIBRATED_COUNTIES, DEFAULTS, khis_mmr, parameter_draw, read_values, save_values, simulate,
)
from Calibration.stages import stage6_mortality as mortality


ANCHORS = ("p_mat_death_facility_baseline", "RR_MM_l23_vs_l4")


def pooled_modeled(calibrated, anchors, args):
    """Expected deaths per 100,000 deliveries at final L4 and L2/3, pooled over counties."""
    deaths = {"l4": 0.0, "l23": 0.0}
    births = {"l4": 0, "l23": 0}
    for county in CALIBRATED_COUNTIES:
        values = {**calibrated[county], **{(name, None): value for name, value in anchors.items()}}
        people = pd.concat(simulate(county, values, args.runs, args.months, args.base_seed), ignore_index=True)
        for level, code in (("l4", 2), ("l23", 1)):
            at_level = people["i_loc_new_v2"].eq(code)
            deaths[level] += float(people.loc[at_level, "p_mat_death"].sum())
            births[level] += int(at_level.sum())
    return {level: 1e5 * deaths[level] / births[level] for level in deaths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", default="Calibration/results/stages_1_to_5",
                        help="Folder with the Stage 1-5 loop outputs (values.csv).")
    parser.add_argument("--runs", type=int, default=DEFAULTS["validation_runs"])
    parser.add_argument("--months", type=int, default=DEFAULTS["validation_months"])
    parser.add_argument("--base-seed", type=int, default=DEFAULTS["simulation_seed"])
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--tolerance", type=float, default=0.01, help="Relative change.")
    args = parser.parse_args()

    stage_dir = REPO_ROOT / args.stage_dir
    calibrated = read_values(stage_dir / "values.csv")
    calibrated = {county: {key: value for key, value in calibrated[county].items() if key[0] not in ANCHORS}
                  for county in CALIBRATED_COUNTIES}
    start = parameter_draw(CALIBRATED_COUNTIES[0])
    values = {name: float(start[name]) for name in ANCHORS}
    targets = {level: khis_mmr(CALIBRATED_COUNTIES, level)[0] for level in ("l4", "l23")}
    print(f"Pooled KHIS targets: L4 {targets['l4']:.1f}, L2/3 {targets['l23']:.1f} per 100,000", flush=True)

    for iteration in range(1, args.max_iterations + 1):
        modeled = pooled_modeled(calibrated, values, args)
        baseline, l23_rr = mortality.update(
            values["p_mat_death_facility_baseline"], values["RR_MM_l23_vs_l4"],
            modeled["l4"], modeled["l23"], targets["l4"], targets["l23"],
        )
        change = max(abs(baseline / values["p_mat_death_facility_baseline"] - 1),
                     abs(l23_rr / values["RR_MM_l23_vs_l4"] - 1))
        print(f"iteration {iteration}: modeled L4 {modeled['l4']:.1f}, L2/3 {modeled['l23']:.1f} -> "
              f"p_mat_death_facility_baseline {baseline:.6f}, RR_MM_l23_vs_l4 {l23_rr:.4f} "
              f"(change {change:.2%})", flush=True)
        values = {"p_mat_death_facility_baseline": baseline, "RR_MM_l23_vs_l4": l23_rr}
        if change < args.tolerance:
            break
    else:
        print("warning: Stage 6 did not converge", flush=True)

    for county in CALIBRATED_COUNTIES:
        calibrated[county].update({(name, None): value for name, value in values.items()})
    save_values(calibrated, stage_dir / "values.csv")
    print(f"Added Stage 6 values to {stage_dir / 'values.csv'}", flush=True)


if __name__ == "__main__":
    main()
