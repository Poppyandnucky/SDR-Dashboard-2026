"""PROMPTS and PULSE results for Kisii, Makueni, and Mombasa: tables (figures: paper_figures.py).

Each run draws one parameter set (seed 4200 + run, runs 21 onwards, as in
scenario_comparison.py) and simulates every scenario with that draw and the same
monthly seeds, so differences from the base scenario are paired (common random
numbers). Each draw is simulated --replicates times (default 20) with different monthly
seeds, and its results are the average over replicates, which shrinks simulation noise.
Monthly seeds come from (parameter seed, replicate), so no two runs share them.
Results are the mean over runs with a 95% uncertainty interval (2.5th-97.5th
percentile over runs, i.e. over parameter draws and the simulation noise left after
averaging). runs_replicates.csv keeps each replicate.

PROMPTS scenarios (saved in --prompts-dir):
    base, current, high             as defined in scenario_comparison.py (implementation
                                    index; registered mothers' odds of 4+ ANC x OR_anc4p,
                                    from the Vatsa et al. 2025 trial; registered mothers
                                    with 4+ ANC screened with sensitivity sen_risk_trad_target)
    high_anc_only                   high, without the screening effect
    high_screening_only             high, without the ANC effect (OR 1)
    dose_<index>                    other implementation indices for the dose-response curve
    optimistic_<level>              sensitivity analysis: current, high with RR on
                                    4+ ANC 1.02 / 1.35 (PROMPTS_RR_BY_LEVEL) and
                                    screening sensitivity 0.95 (PROMPTS_OPTIMISTIC_SEN_RISK)

PULSE (LB_effect.pulse_effect) raises PROMPTS registration (odds ratio
pulse_or_prompts_registration, 1.47, scaled by implementation) and facility
health-worker knowledge (pulse_implementation_index x pulse_performance_improve,
0.062). Alone it works through knowledge only; with PROMPTS it also registers more
mothers. PULSE runs at its current implementation only, as a range: penetration (how
far the dashboard's feedback reaches health workers inside PULSE facilities) is not
measured, so the current index (penetration 1) is also run at half (penetration 0.5).
FQA's feedback on PULSE (flag_fqa_pulse_feedback) is off throughout.

PULSE scenarios (saved in --pulse-dir):
    base                            no interventions
    pulse_<pulse>                   PULSE alone; <pulse> is current_lower or current_upper
    prompts_<level>                 PROMPTS alone (current, high), as in scenario_comparison.py
    prompts_<level>_pulse_<pulse>   PROMPTS and PULSE
    prompts_<level>_pulse_current_upper_registration, ..._knowledge
                                    the same, with only PULSE's registration or knowledge
                                    effect

Maternal deaths are expected deaths (the sum of individual death probabilities,
p_mat_death), which have much less Monte Carlo noise than sampled deaths. Home
death risk (p_MM_home) is the Kakamega calibration value.

Usage:
    python analysis/scripts/prompts_pulse_results.py --runs 20 --replicates 1    # quick check
    python analysis/scripts/prompts_pulse_results.py --runs 200                  # paper version
    python analysis/scripts/prompts_pulse_results.py --analysis pulse            # PULSE only
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

# Importing calibration_common also points the model at the repository workbook.
from Calibration.calibration_common import CALIBRATED_COUNTIES as COUNTIES
from global_func import reset_E, reset_HSS, reset_S, reset_flags
from model_run import run_model_dash
from parameter_loader import calculate_derived_parameters, get_parameters, get_slider_params
from scenario_comparison import (
    BASE_SEED, CURRENT_IMPLEMENTATION_INDEX, FIRST_ANALYSIS_RUN, IMPLEMENTATION_INDEX_BY_LEVEL, N_MONTHS,
    PROMPTS_OPTIMISTIC_SEN_RISK, PROMPTS_RR_BY_LEVEL, configure_scenario, implementation_for_level,
)

# ---- Outcomes (both analyses) ----------------------------------------------

# Complications and other outcomes per 1,000 births: column -> label.
COMPLICATIONS = {"i_anemia_new": "Anaemia at delivery", "i_pph_new": "PPH", "i_eclampsia_new": "Eclampsia",
                 "i_mat_sepsis_new": "Maternal sepsis", "i_severe_new": "Severe maternal complications",
                 "i_stillbirth": "Stillbirths", "i_neo_death": "Neonatal deaths"}
TREATMENTS = {"i_oxytocin": "Oxytocin", "i_antibiotics": "Antibiotics", "i_MgSO4": "MgSO4",
              "i_iv_iron": "IV iron (ANC)"}
RELATIVE = ("mmr", "facility_mmr_l23_l4")
# Final delivery location -> label, for deaths by place.
PLACES = {0: "home", 1: "l23", 2: "l4", 3: "l5"}
PLACE_LABELS = {"home": "Home", "l23": "L2/3", "l4": "L4", "l5": "L5"}

# ---- PROMPTS scenarios -----------------------------------------------------

# Scenarios defined in scenario_comparison.py, and those added here:
# name -> (implementation index or None for the county's current one, ANC effect on, screening effect on).
# Current and high implementation bound the results.
LEVELS = ("base", "current", "high")
MAIN_LEVELS = ("current", "high")
EXTRA = {
    "high_anc_only": (IMPLEMENTATION_INDEX_BY_LEVEL["high"], True, False),
    "high_screening_only": (IMPLEMENTATION_INDEX_BY_LEVEL["high"], False, True),
    **{f"dose_{index:.2f}": (index, True, True) for index in (0.10, 0.25, 0.50, 0.75)},
}
# Optimistic sensitivity analysis: name -> level.
OPTIMISTIC = {f"optimistic_{level}": level for level in MAIN_LEVELS}
PROMPTS_SCENARIOS = (*LEVELS, *EXTRA, *OPTIMISTIC)
# Outcome -> label, in the order of the PROMPTS pathway (ANC -> recognition -> place
# and mode of birth -> deaths). Shares are per birth unless stated.
OUTCOMES = {
    "anc4": "4+ ANC visits",
    "highrisk_identified": "High-risk women identified in ANC",
    "home": "Home births",
    "l23": "Births at L2/3 (after transfer)",
    "l45": "Births at L4/5 (after transfer)",
    "transfers_per_1000": "Emergency transfers from L2/3 per 1,000 births",
    "planned_cs_l4": "Planned CS among L4 births",
    "preterm": "Preterm births",
    "mmr": "Maternal deaths per 100,000 births (expected)",
    "facility_mmr_l23_l4": "Maternal deaths per 100,000 L2/3 and L4 births (calibrated part)",
    **{f"per1000_{column}": f"{label} per 1,000 births" for column, label in COMPLICATIONS.items()},
}
RATES = ("transfers_per_1000", "mmr", "facility_mmr_l23_l4", *(f"per1000_{column}" for column in COMPLICATIONS))

# ---- PULSE scenarios -------------------------------------------------------

# PROMPTS at current and high implementation bound the results; PULSE at its current
# implementation, with penetration 0.5 (lower) or 1 (upper, the current index as it stands).
PULSE_PENETRATION = {"current_lower": 0.5, "current_upper": 1.0}
PULSE_LEVELS = tuple(PULSE_PENETRATION)
# name -> (PROMPTS level, PULSE level, PULSE pathway: "both", "registration", or "knowledge").
PULSE_SCENARIOS = {
    "base": (None, None, None),
    **{f"pulse_{pulse}": (None, pulse, "both") for pulse in PULSE_LEVELS},
    **{f"prompts_{level}": (level, None, None) for level in MAIN_LEVELS},
    **{f"prompts_{level}_pulse_{pulse}": (level, pulse, "both") for level in MAIN_LEVELS for pulse in PULSE_LEVELS},
    **{f"prompts_{level}_pulse_current_upper_{pathway}": (level, "current_upper", pathway)
       for level in MAIN_LEVELS for pathway in ("registration", "knowledge")},
}
# Outcome -> label, in the order of the PULSE pathway. Shares are per birth unless stated.
PULSE_OUTCOMES = {
    "prompts_registered": "Registered with PROMPTS",
    "anc4": "4+ ANC visits",
    "highrisk_identified": "High-risk women identified in ANC",
    "l45": "Births at L4/5 (after transfer)",
    **{f"treated_{column}": f"{label} given per 1,000 births" for column, label in TREATMENTS.items()},
    "mmr": "Maternal deaths per 100,000 births (expected)",
    "facility_mmr_l23_l4": "Maternal deaths per 100,000 L2/3 and L4 births (calibrated part)",
    **{f"per1000_{column}": f"{label} per 1,000 births" for column, label in COMPLICATIONS.items()},
}
PULSE_RATES = ("mmr", "facility_mmr_l23_l4", *(f"treated_{column}" for column in TREATMENTS),
               *(f"per1000_{column}" for column in COMPLICATIONS))


def prompts_implementation(name, county):
    """PROMPTS implementation index of a PROMPTS scenario."""
    if name == "base":
        return 0.0
    name = OPTIMISTIC.get(name, name)
    if name in LEVELS:
        return CURRENT_IMPLEMENTATION_INDEX["prompts"][county] if name == "current" else IMPLEMENTATION_INDEX_BY_LEVEL[name]
    return EXTRA[name][0]


def configure_prompts(name, param, flags, sliders):
    """Set PROMPTS for a scenario: the defined levels exactly as scenario_comparison.py does."""
    if name in LEVELS:
        configure_scenario(f"prompts_{name}", param, flags, sliders)
        return
    if name in OPTIMISTIC:
        level = OPTIMISTIC[name]
        configure_scenario(f"prompts_{level}", param, flags, sliders)
        param["prompts_rr_anc4p"] = PROMPTS_RR_BY_LEVEL[level]
        param["sen_risk_trad_target"] = PROMPTS_OPTIMISTIC_SEN_RISK
        return
    index, anc_effect, screening_effect = EXTRA[name]
    flags["flag_PROMPTS"] = 1
    param["HSS"]["prompts_implementation_index"] = index
    param["prompts_implementation_index"] = index
    if not anc_effect:
        param["OR_anc4p"] = 1.0
    if not screening_effect:
        # Registered mothers with 4+ ANC are screened at sen_risk_trad_target;
        # hold screening at its baseline instead.
        param["sen_risk_trad_target"] = param["sen_risk_trad"]
        param["spec_risk_trad_target"] = param["spec_risk_trad"]


def pulse_implementation(name, county):
    """PULSE implementation index of a PULSE scenario."""
    pulse_level = PULSE_SCENARIOS[name][1]
    if not pulse_level:
        return 0.0
    return implementation_for_level("pulse", "current", county) * PULSE_PENETRATION[pulse_level]


def configure_pulse(name, param, flags, sliders):
    """PROMPTS as in scenario_comparison.py, then PULSE at its current level scaled by
    the scenario's penetration, with one pathway switched off for the mechanism scenarios."""
    prompts_level, pulse_level, pathway = PULSE_SCENARIOS[name]
    if prompts_level:
        configure_scenario(f"prompts_{prompts_level}", param, flags, sliders)
    if pulse_level:
        configure_scenario("pulse_current", param, flags, sliders)
        for target in (param, param["HSS"]):
            target["pulse_implementation_index"] *= PULSE_PENETRATION[pulse_level]
    if pathway == "registration":
        param["pulse_performance_improve"] = 0.0
    elif pathway == "knowledge":
        param["pulse_or_prompts_registration"] = 1.0


def measure(people):
    """Model values of each outcome for one run."""
    final = people["i_loc_new_v2"]
    births = len(people)
    l4 = final.eq(2)
    l23_l4 = final.isin([1, 2])
    expected_deaths = float(people["p_mat_death"].sum())
    by_place = people.groupby(final)["p_mat_death"].sum()
    return {
        "births": births,
        "anc4": float(people["i_ANC"].eq(1).mean()),
        "highrisk_identified": float(people.loc[people["i_risk"].eq(1), "i_risk_pred"].eq(1).mean()),
        "home": float(final.eq(0).mean()),
        "l23": float(final.eq(1).mean()),
        "l45": float(final.ge(2).mean()),
        "transfers_per_1000": 1000 * float((people["i_loc"].eq(1) & final.ge(2)).sum()) / births,
        "planned_cs_l4": float(people.loc[l4, "i_mod"].eq("ELCS").mean()),
        "preterm": float(people["i_preterm"].eq(1).mean()),
        "expected_deaths": expected_deaths,
        "sampled_deaths": int(people["i_mat_death"].eq(1).sum()),
        "mmr": 1e5 * expected_deaths / births,
        "facility_mmr_l23_l4": 1e5 * float(people.loc[l23_l4, "p_mat_death"].sum()) / int(l23_l4.sum()),
        # Expected deaths at each place per 100,000 births (they sum to mmr).
        **{f"deaths_{name}": 1e5 * float(by_place.get(code, 0.0)) / births for code, name in PLACES.items()},
        **{f"per1000_{column}": 1000 * float(people[column].eq(1).sum()) / births for column in COMPLICATIONS},
    }


def measure_pulse(people):
    """measure, plus PROMPTS registration and treatments given."""
    births = len(people)
    return {
        **measure(people),
        "prompts_registered": float(people["i_prompts_registered"].eq(1).mean()),
        **{f"treated_{column}": 1000 * float(people[column].eq(1).sum()) / births for column in TREATMENTS},
    }


# analysis -> (scenarios, configure, implementation, measure).
ANALYSES = {
    "prompts": (PROMPTS_SCENARIOS, configure_prompts, prompts_implementation, measure),
    "pulse": (tuple(PULSE_SCENARIOS), configure_pulse, pulse_implementation, measure_pulse),
}


def month_seeds(parameter_seed, replicate):
    """Monthly seeds for one replicate of one parameter draw, independent of those of
    every other draw and replicate."""
    return np.random.SeedSequence([parameter_seed, replicate]).generate_state(N_MONTHS)


def average_replicates(rows):
    """One row per county, run, and scenario: the average over replicates."""
    replicates = pd.DataFrame(rows)
    runs = replicates.drop(columns="replicate").groupby(["county", "run", "level"], sort=False).mean().reset_index()
    return replicates, runs


def run_task(task):
    """All scenarios of one analysis for one county, run (one parameter draw), and replicate."""
    analysis, county, run, replicate = task
    scenarios, configure, implementation, measure_people = ANALYSES[analysis]
    sliders = get_slider_params(county=county)
    parameter_seed = BASE_SEED + run
    seeds = month_seeds(parameter_seed, replicate)
    rows = []
    for name in scenarios:
        param = calculate_derived_parameters(get_parameters(county=county, seed=parameter_seed))
        param.update({"E": reset_E(), "S": reset_S(sliders), "HSS": reset_HSS(sliders)})
        flags = reset_flags()
        configure(name, param, flags, sliders)
        _, people, _ = run_model_dash(param, flags, n_months=N_MONTHS, int_period=N_MONTHS, base_seed=seeds)
        rows.append({"county": county, "run": run, "level": name, "replicate": replicate,
                     "implementation": implementation(name, county), **measure_people(people)})
    return analysis, rows


def interval(values):
    return values.mean(), values.quantile(0.025), values.quantile(0.975)


def paired(runs, level, outcome, relative=False, reference="base"):
    """Per-run change from the reference scenario for one scenario and outcome."""
    ref = runs[runs.level.eq(reference)].set_index(["county", "run"])[outcome]
    group = runs[runs.level.eq(level)].set_index(["county", "run"])[outcome]
    change = group - ref.loc[group.index]
    return change / ref.loc[group.index] if relative else change


def pulse_reference(level):
    """PROMPTS + PULSE scenarios (whole or one pathway) are also compared with PROMPTS
    alone at the same level."""
    prompts_level, pulse_level, _ = PULSE_SCENARIOS.get(level, (None, None, None))
    return f"prompts_{prompts_level}" if prompts_level and pulse_level else None


def summary_table(runs, outcomes, reference_of=lambda level: None):
    """Mean and 95% interval of each outcome, its paired change from base (relative
    change for maternal deaths), and, where reference_of names one, its paired change
    from that reference (for PROMPTS + PULSE, PULSE's added effect over PROMPTS alone)."""
    rows = []
    for (county, level), group in runs.groupby(["county", "level"], sort=False):
        for outcome, label in outcomes.items():
            mean, low, high = interval(group[outcome])
            row = {"county": county, "level": level, "implementation": group.implementation.iloc[0],
                   "outcome": label, "value": mean, "value_low": low, "value_high": high}
            if level != "base":
                change = paired(runs, level, outcome, outcome in RELATIVE).loc[county]
                row.update(zip(("change", "change_low", "change_high"), interval(change)))
            if reference_of(level):
                added = paired(runs, level, outcome, outcome in RELATIVE, reference_of(level)).loc[county]
                row.update(zip(("added", "added_low", "added_high"), interval(added)))
            rows.append(row)
    return pd.DataFrame(rows)


def prompts_readable(table):
    """One cell per county x level: value (95% interval), and the change from base."""
    rates = [OUTCOMES[name] for name in RATES]
    relative = [OUTCOMES[name] for name in RELATIVE]

    def cell(row):
        pct = row.outcome not in rates
        fmt = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.1f}")
        text = f"{fmt(row.value)} ({fmt(row.value_low)}-{fmt(row.value_high)})"
        if pd.notna(row.get("change", np.nan)):
            if row.outcome in relative:
                text += f"; {100 * row.change:+.1f}% ({100 * row.change_low:+.1f} to {100 * row.change_high:+.1f})"
            elif pct:
                text += f"; {100 * row.change:+.1f} pts"
            else:
                text += f"; {row.change:+.1f}"
        return text
    table = table[table.level.isin(("base", *MAIN_LEVELS))]
    table = table.assign(cell=table.apply(cell, axis=1), column=table.county.str.title() + " " + table.level)
    order = [f"{c.title()} {lv}" for c in COUNTIES for lv in ("base", *MAIN_LEVELS)]
    return table.pivot(index="outcome", columns="column", values="cell").loc[list(OUTCOMES.values()), order]


def pulse_readable(table):
    """One cell per county x scenario: value (95% interval); change from base; and
    for PROMPTS + PULSE, PULSE's added effect over PROMPTS alone."""
    rates = [PULSE_OUTCOMES[name] for name in PULSE_RATES]
    relative = [PULSE_OUTCOMES[name] for name in RELATIVE]

    def fmt_change(row, prefix):
        value, low, high = row[prefix], row[f"{prefix}_low"], row[f"{prefix}_high"]
        if row.outcome in relative:
            return f"{100 * value:+.1f}% ({100 * low:+.1f} to {100 * high:+.1f})"
        if row.outcome in rates:
            return f"{value:+.1f} ({low:+.1f} to {high:+.1f})"
        return f"{100 * value:+.1f} pts ({100 * low:+.1f} to {100 * high:+.1f})"

    def cell(row):
        fmt = (lambda v: f"{v:.1f}") if row.outcome in rates else (lambda v: f"{100 * v:.1f}%")
        text = f"{fmt(row.value)} ({fmt(row.value_low)}-{fmt(row.value_high)})"
        if pd.notna(row.get("change", np.nan)):
            text += f"; vs base {fmt_change(row, 'change')}"
        if pd.notna(row.get("added", np.nan)):
            text += f"; PULSE added {fmt_change(row, 'added')}"
        return text
    shown = [name for name, (_, _, pathway) in PULSE_SCENARIOS.items() if pathway in (None, "both")]
    table = table[table.level.isin(shown)]
    table = table.assign(cell=table.apply(cell, axis=1), column=table.county.str.title() + " " + table.level)
    order = [f"{c.title()} {lv}" for c in COUNTIES for lv in shown]
    return table.pivot(index="outcome", columns="column", values="cell").loc[list(PULSE_OUTCOMES.values()), order]


def prompts_deaths_averted_per_year(runs):
    """Maternal deaths averted per year in each county (expected deaths, paired with base)."""
    years = N_MONTHS / 12
    rows = []
    for county in COUNTIES:
        births = runs[runs.county.eq(county) & runs.level.eq("base")].births.mean() / years
        for level in (*MAIN_LEVELS, *OPTIMISTIC):
            averted = -paired(runs, level, "expected_deaths").loc[county] / years
            mean, low, high = interval(averted)
            rows.append({"county": county, "level": level, "births_per_year": births,
                         "deaths_averted_per_year": mean, "low": low, "high": high})
    return pd.DataFrame(rows)


def pulse_deaths_averted_per_year(runs):
    """Maternal deaths averted per year (expected deaths): PULSE alone vs base, and
    PULSE added to PROMPTS (PROMPTS + PULSE vs PROMPTS alone)."""
    years = N_MONTHS / 12
    rows = []
    for county in COUNTIES:
        births = runs[runs.county.eq(county) & runs.level.eq("base")].births.mean() / years
        comparisons = [("PULSE alone vs none", None, pulse, f"pulse_{pulse}", "base") for pulse in PULSE_LEVELS]
        comparisons += [("PULSE added to PROMPTS", level, pulse, f"prompts_{level}_pulse_{pulse}", f"prompts_{level}")
                        for level in MAIN_LEVELS for pulse in PULSE_LEVELS]
        for comparison, level, pulse, scenario, reference in comparisons:
            averted = -paired(runs, scenario, "expected_deaths", reference=reference).loc[county] / years
            mean, low, high = interval(averted)
            rows.append({"county": county, "prompts": level, "pulse": pulse, "comparison": comparison,
                         "births_per_year": births, "deaths_averted_per_year": mean, "low": low, "high": high})
    return pd.DataFrame(rows)


def averted_by_place(runs):
    """Mean maternal deaths averted per 100,000 births at each place of delivery
    (paired with base); the places sum to the change in mmr."""
    rows = []
    for county in COUNTIES:
        for level in MAIN_LEVELS:
            total = -paired(runs, level, "mmr").loc[county].mean()
            for name in PLACES.values():
                averted = -paired(runs, level, f"deaths_{name}").loc[county].mean()
                rows.append({"county": county, "level": level, "place": PLACE_LABELS[name],
                             "averted_per_100k": averted, "share_of_averted": averted / total})
    return pd.DataFrame(rows)


def save_prompts(runs, output):
    table = summary_table(runs, OUTCOMES)
    table.to_csv(output / "summary.csv", index=False)
    prompts_readable(table).to_csv(output / "table.csv")
    per_year = prompts_deaths_averted_per_year(runs)
    per_year.to_csv(output / "deaths_averted_per_year.csv", index=False)
    averted_by_place(runs).to_csv(output / "averted_by_place.csv", index=False)
    return prompts_readable(table), per_year


def save_pulse(runs, output):
    table = summary_table(runs, PULSE_OUTCOMES, pulse_reference)
    table.to_csv(output / "summary.csv", index=False)
    pulse_readable(table).to_csv(output / "table.csv")
    per_year = pulse_deaths_averted_per_year(runs)
    per_year.to_csv(output / "deaths_averted_per_year.csv", index=False)
    return pulse_readable(table), per_year


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", choices=("both", *ANALYSES), default="both")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--prompts-dir", default="analysis/results/prompts")
    parser.add_argument("--pulse-dir", default="analysis/results/pulse")
    args = parser.parse_args()

    analyses = tuple(ANALYSES) if args.analysis == "both" else (args.analysis,)
    outputs = {"prompts": REPO_ROOT / args.prompts_dir, "pulse": REPO_ROOT / args.pulse_dir}
    tasks = [(analysis, county, run, replicate) for analysis in analyses for county in COUNTIES
             for run in range(FIRST_ANALYSIS_RUN - 1, FIRST_ANALYSIS_RUN - 1 + args.runs)
             for replicate in range(args.replicates)]
    rows = {analysis: [] for analysis in analyses}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for analysis, task_rows in pool.map(run_task, tasks):
            rows[analysis].extend(task_rows)

    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    for analysis in analyses:
        output = outputs[analysis]
        output.mkdir(parents=True, exist_ok=True)
        replicates, runs = average_replicates(rows[analysis])
        replicates.to_csv(output / "runs_replicates.csv", index=False)
        runs.to_csv(output / "runs.csv", index=False)
        readable, per_year = (save_prompts if analysis == "prompts" else save_pulse)(runs, output)
        print(f"\n{analysis.upper()}\n{readable.to_string()}")
        print("\nMaternal deaths averted per year (95% interval):")
        print(per_year.round(1).to_string(index=False))
        print(f"Saved: {output}")
    print(f"\n{args.runs} runs x {args.replicates} replicates x {N_MONTHS} months per scenario.")


if __name__ == "__main__":
    main()
