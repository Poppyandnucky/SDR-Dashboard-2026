"""MENTORS and FQA results for Kisii, Makueni, and Mombasa: tables.

Both raise facility health-worker knowledge, which multiplies supply to give treatment
coverage (IV iron in ANC; oxytocin, MgSO4, antibiotics, and others at delivery):
    MENTORS  knowledge + mentors_implementation_index x mentors_knowledge_improve (0.103,
             Rowe et al. 2018, in-service training)
    FQA      knowledge after MENTORS + fqa_implementation_index x fqa_performance_improve
             (0.062, Ivers et al. 2026, audit and feedback), in ANC and at delivery
Supplies are the workbook's in every scenario (no supply boost), so differences from base
come from knowledge only. PULSE is off throughout. Runs, seeds, and implementation levels
are as in prompts_pulse_results.py, so every scenario is paired with the same parameter draw and
monthly seeds, and each draw is the average of --replicates simulations (default 20).

MENTORS runs at current and high implementation. FQA runs at its current implementation
only, as a range: penetration (how far assessment and feedback reach health workers inside
FQA facilities) is not measured, so the current index (penetration 1) is also run at half
(penetration 0.5).

The supplies-filled scenarios show what MENTORS and FQA could do once the supply gaps FQA
found are closed: oxytocin, MgSO4, antibiotics, and IV iron at
ENABLED_SINGLE_INTERVENTION_SUPPLY (0.95) at L4/5, as scenario_comparison.py's
enable_single_interventions sets them. The PPH bundle (E-MOTIVE) stays off: it is a new
intervention, not a supply gap.

Scenarios:
    base                            no interventions
    mentors_<level>                 MENTORS alone; <level> is current or high
    fqa_<fqa>                       FQA alone; <fqa> is current_lower or current_upper
    mentors_<level>_fqa_<fqa>       MENTORS and FQA
    supplies                        supplies filled alone
    supplies_mentors_<level>_fqa_<fqa>
                                    supplies filled, MENTORS, and FQA
    supplies_mentors_<level>        supplies filled and MENTORS
    ..._rowe_upper                  sensitivity: mentors_knowledge_improve at Rowe et al.'s
                                    upper quartile (0.207) instead of the median (0.103)

Usage:
    python analysis/scripts/mentors_fqa_results.py --runs 20 --replicates 1    # quick check
    python analysis/scripts/mentors_fqa_results.py --runs 200                  # paper version
    python analysis/scripts/mentors_fqa_results.py --scenarios mentors_current supplies_mentors_current
                                    # only these (base, and supplies where needed, are added)
    python analysis/scripts/mentors_fqa_results.py --scenarios ... --reuse analysis/results/mentors_fqa
                                    # take scenarios already in that run's runs_replicates.csv
                                    # (same seeds, so identical if model and workbook are unchanged)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prompts_pulse_results import COMPLICATIONS, COUNTIES, TREATMENTS, average_replicates, interval, measure_pulse, month_seeds, paired
from global_func import reset_E, reset_HSS, reset_S, reset_flags
from model_run import run_model_dash
from parameter_loader import calculate_derived_parameters, get_parameters, get_slider_params
from scenario_comparison import (
    BASE_SEED, FIRST_ANALYSIS_RUN, N_MONTHS, configure_scenario, enable_single_interventions, implementation_for_level,
)

# MENTORS at current and high implementation bound the results; FQA at its current
# implementation, with penetration 0.5 (lower) or 1 (upper, the current index as it stands).
LEVELS = ("current", "high")
FQA_PENETRATION = {"current_lower": 0.5, "current_upper": 1.0}
FQA_LEVELS = tuple(FQA_PENETRATION)
# Rowe et al. 2018 upper quartile (IQR 6.1-20.7 points) for the MENTORS sensitivity.
MENTORS_KNOWLEDGE_IMPROVE_UPPER = 0.207
# name -> (MENTORS level, FQA level, supplies filled, mentors_knowledge_improve or None
# for the workbook value).
SCENARIOS = {
    "base": (None, None, False, None),
    **{f"mentors_{level}": (level, None, False, None) for level in LEVELS},
    **{f"fqa_{fqa}": (None, fqa, False, None) for fqa in FQA_LEVELS},
    **{f"mentors_{level}_fqa_{fqa}": (level, fqa, False, None) for level in LEVELS for fqa in FQA_LEVELS},
    "supplies": (None, None, True, None),
    **{f"supplies_mentors_{level}_fqa_{fqa}": (level, fqa, True, None) for level in LEVELS for fqa in FQA_LEVELS},
    **{f"supplies_mentors_{level}": (level, None, True, None) for level in LEVELS},
    **{f"mentors_{level}_rowe_upper": (level, None, False, MENTORS_KNOWLEDGE_IMPROVE_UPPER) for level in LEVELS},
    **{f"supplies_mentors_{level}_rowe_upper": (level, None, True, MENTORS_KNOWLEDGE_IMPROVE_UPPER)
       for level in LEVELS},
}
# Outcome -> label, in the order of the pathway (treatments -> complications -> deaths).
OUTCOMES = {
    **{f"treated_{column}": f"{label} given per 1,000 births" for column, label in TREATMENTS.items()},
    **{f"per1000_{column}": f"{label} per 1,000 births" for column, label in COMPLICATIONS.items()},
    "mmr": "Maternal deaths per 100,000 births (expected)",
    "facility_mmr_l23_l4": "Maternal deaths per 100,000 L2/3 and L4 births (calibrated part)",
}
RELATIVE = ("mmr", "facility_mmr_l23_l4")


def configure(name, param, flags, sliders):
    """MENTORS and FQA as in scenario_comparison.py, with FQA's current index scaled by
    the scenario's penetration, and supplies filled where the scenario says so."""
    mentors_level, fqa_level, supplies, mentors_knowledge_improve = SCENARIOS[name]
    if mentors_level:
        configure_scenario(f"mentors_{mentors_level}", param, flags, sliders)
        if mentors_knowledge_improve is not None:
            param["mentors_knowledge_improve"] = mentors_knowledge_improve
    if fqa_level:
        configure_scenario("fqa_current", param, flags, sliders)
        for target in (param, param["HSS"]):
            target["fqa_implementation_index"] *= FQA_PENETRATION[fqa_level]
    if supplies:
        enable_single_interventions(param, flags)
        flags["flag_pph_bundle"] = 0


def implementation(name, county):
    """MENTORS and FQA implementation indexes of a scenario."""
    mentors_level, fqa_level, _, _ = SCENARIOS[name]
    mentors = implementation_for_level("mentors", mentors_level, county) if mentors_level else 0.0
    fqa = implementation_for_level("fqa", "current", county) * FQA_PENETRATION[fqa_level] if fqa_level else 0.0
    return mentors, fqa


def run_task(task):
    """The chosen scenarios for one county, run (one parameter draw), and replicate."""
    county, run, replicate, names = task
    sliders = get_slider_params(county=county)
    parameter_seed = BASE_SEED + run
    seeds = month_seeds(parameter_seed, replicate)
    rows = []
    for name in names:
        param = calculate_derived_parameters(get_parameters(county=county, seed=parameter_seed))
        param.update({"E": reset_E(), "S": reset_S(sliders), "HSS": reset_HSS(sliders)})
        flags = reset_flags()
        configure(name, param, flags, sliders)
        _, people, _ = run_model_dash(param, flags, n_months=N_MONTHS, int_period=N_MONTHS, base_seed=seeds)
        mentors, fqa = implementation(name, county)
        rows.append({"county": county, "run": run, "level": name, "replicate": replicate,
                     "mentors_implementation": mentors, "fqa_implementation": fqa,
                     "supplies_filled": SCENARIOS[name][2], **measure_pulse(people)})
    return rows


def reference_of(level):
    """MENTORS + FQA scenarios are also compared with MENTORS alone at the same level
    (FQA's added effect), and supplies-filled ones with supplies filled alone (MENTORS
    and FQA's added effect once supplies are filled)."""
    mentors_level, fqa_level, supplies, _ = SCENARIOS[level]
    if supplies:
        return "supplies" if mentors_level or fqa_level else None
    return f"mentors_{mentors_level}" if mentors_level and fqa_level else None


def summary_table(runs):
    """Mean and 95% interval of each outcome, its paired change from base, and for
    MENTORS + FQA its paired change from its reference (reference_of)."""
    rows = []
    for (county, level), group in runs.groupby(["county", "level"], sort=False):
        for outcome, label in OUTCOMES.items():
            mean, low, high = interval(group[outcome])
            row = {"county": county, "level": level,
                   "mentors_implementation": group.mentors_implementation.iloc[0],
                   "fqa_implementation": group.fqa_implementation.iloc[0],
                   "outcome": label, "value": mean, "value_low": low, "value_high": high}
            if level != "base":
                change = paired(runs, level, outcome, outcome in RELATIVE).loc[county]
                row.update(zip(("change", "change_low", "change_high"), interval(change)))
            if reference_of(level):
                added = paired(runs, level, outcome, outcome in RELATIVE, reference_of(level)).loc[county]
                row.update(zip(("added", "added_low", "added_high"), interval(added)))
            rows.append(row)
    return pd.DataFrame(rows)


def readable(table):
    """One cell per county x scenario: value (95% interval); change from base; and
    for MENTORS + FQA, the added effect over its reference (reference_of)."""
    relative = [OUTCOMES[name] for name in RELATIVE]

    def fmt_change(row, prefix):
        value, low, high = row[prefix], row[f"{prefix}_low"], row[f"{prefix}_high"]
        if row.outcome in relative:
            return f"{100 * value:+.1f}% ({100 * low:+.1f} to {100 * high:+.1f})"
        return f"{value:+.2f} ({low:+.2f} to {high:+.2f})"

    def cell(row):
        text = f"{row.value:.1f} ({row.value_low:.1f}-{row.value_high:.1f})"
        if pd.notna(row.get("change", np.nan)):
            text += f"; vs base {fmt_change(row, 'change')}"
        if pd.notna(row.get("added", np.nan)):
            text += f"; vs {reference_of(row.level)} {fmt_change(row, 'added')}"
        return text
    table = table.assign(cell=table.apply(cell, axis=1), column=table.county.str.title() + " " + table.level)
    order = [f"{c.title()} {lv}" for c in COUNTIES for lv in SCENARIOS if lv in set(table.level)]
    return table.pivot(index="outcome", columns="column", values="cell").loc[list(OUTCOMES.values()), order]


def deaths_averted_per_year(runs):
    """Maternal deaths averted per year (expected deaths): each scenario vs base, and
    vs its reference (reference_of)."""
    years = N_MONTHS / 12
    rows = []
    for county in COUNTIES:
        births = runs[runs.county.eq(county) & runs.level.eq("base")].births.mean() / years
        names = [name for name in SCENARIOS if name in set(runs.level)]
        comparisons = [("vs none", name, "base") for name in names if name != "base"]
        comparisons += [(f"vs {reference_of(name)}", name, reference_of(name)) for name in names if reference_of(name)]
        for comparison, scenario, reference in comparisons:
            averted = -paired(runs, scenario, "expected_deaths", reference=reference).loc[county] / years
            mean, low, high = interval(averted)
            rows.append({"county": county, "scenario": scenario, "comparison": comparison,
                         "births_per_year": births, "deaths_averted_per_year": mean, "low": low, "high": high})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output-dir", default="analysis/results/mentors_fqa")
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), help="default: all")
    parser.add_argument("--reuse", help="earlier output dir whose runs_replicates.csv supplies scenarios it has")
    args = parser.parse_args()

    # Base, and the reference of each chosen scenario, run too so every comparison is paired.
    chosen = set(args.scenarios or SCENARIOS) | {"base"}
    chosen |= {reference_of(name) for name in chosen if reference_of(name)}
    names = [name for name in SCENARIOS if name in chosen]
    output = REPO_ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    run_ids = range(FIRST_ANALYSIS_RUN - 1, FIRST_ANALYSIS_RUN - 1 + args.runs)
    reused = []
    if args.reuse:
        earlier = pd.read_csv(REPO_ROOT / args.reuse / "runs_replicates.csv")
        earlier = earlier[earlier.run.isin(run_ids) & earlier.replicate.lt(args.replicates)]
        for name in names:
            rows = earlier[earlier.level.eq(name)]
            if len(rows) == len(COUNTIES) * args.runs * args.replicates:
                reused += rows.to_dict("records")
        print("Reused from", args.reuse, ":", sorted({row["level"] for row in reused}))
    to_run = [name for name in names if name not in {row["level"] for row in reused}]
    tasks = [(county, run, replicate, to_run) for county in COUNTIES for run in run_ids
             for replicate in range(args.replicates)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = reused + [row for rows in pool.map(run_task, tasks) for row in rows]
    rows.sort(key=lambda row: (COUNTIES.index(row["county"]), row["run"], names.index(row["level"]), row["replicate"]))
    replicates, runs = average_replicates(rows)
    replicates.to_csv(output / "runs_replicates.csv", index=False)
    runs.to_csv(output / "runs.csv", index=False)

    table = summary_table(runs)
    table.to_csv(output / "summary.csv", index=False)
    readable(table).to_csv(output / "table.csv")
    per_year = deaths_averted_per_year(runs)
    per_year.to_csv(output / "deaths_averted_per_year.csv", index=False)

    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print(readable(table).to_string())
    print("\nMaternal deaths averted per year (95% interval):")
    print(per_year.round(2).to_string(index=False))
    print(f"\n{len(names)} scenarios, {args.runs} runs x {args.replicates} replicates x {N_MONTHS} months each. Saved: {output}")


if __name__ == "__main__":
    main()
