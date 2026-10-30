"""PROMPTS results for Kisii, Makueni, and Mombasa: tables and figures.

Each run draws one parameter set (seed 4200 + run, runs 21 onwards, as in
scenario_comparison.py) and simulates every scenario with that draw and the same
monthly seeds, so differences from the base scenario are paired. Results are the
mean over runs with a 95% uncertainty interval (2.5th-97.5th percentile over runs,
i.e. over parameter draws and simulation noise).

Scenarios:
    base, current, moderate, high   as defined in scenario_comparison.py
                                    (implementation index; RR on 4+ ANC 1.02 / 1.18 / 1.35;
                                    registered mothers with 4+ ANC screened with
                                    sensitivity sen_risk_trad_target, 0.95)
    high_anc_only                   high, without the screening effect
    high_screening_only             high, without the ANC effect (RR 1)
    dose_<index>                    other implementation indices for the dose-response
                                    curve; RR interpolated through the defined levels

Maternal deaths are expected deaths (the sum of individual death probabilities,
p_mat_death), which have much less Monte Carlo noise than sampled deaths. Home
death risk (p_MM_home) is the Kakamega calibration value.

Usage:
    python analysis/prompts_results.py --runs 20          # quick check
    python analysis/prompts_results.py --runs 200         # paper version
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

# Importing calibration_common also points the model at the repository workbook.
from Calibration.calibration_common import CALIBRATED_COUNTIES as COUNTIES
from global_func import reset_E, reset_HSS, reset_S, reset_flags
from model_run import run_model_dash
from parameter_loader import calculate_derived_parameters, get_parameters, get_slider_params
from scenario_comparison import (
    BASE_SEED, CURRENT_IMPLEMENTATION_INDEX, FIRST_ANALYSIS_RUN, IMPLEMENTATION_INDEX_BY_LEVEL, N_MONTHS,
    PROMPTS_RR_BY_LEVEL, configure_scenario,
)

# Scenarios defined in scenario_comparison.py, and those added here:
# name -> (implementation index or None for the county's current one, ANC effect on, screening effect on).
LEVELS = ("base", "current", "moderate", "high")
MAIN_LEVELS = ("moderate", "high")
EXTRA = {
    "high_anc_only": (IMPLEMENTATION_INDEX_BY_LEVEL["high"], True, False),
    "high_screening_only": (IMPLEMENTATION_INDEX_BY_LEVEL["high"], False, True),
    **{f"dose_{index:.2f}": (index, True, True) for index in (0.10, 0.25, 0.75)},
}
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
}
RATES = ("transfers_per_1000", "mmr", "facility_mmr_l23_l4")
RELATIVE = ("mmr", "facility_mmr_l23_l4")
CASCADE = ("anc4", "highrisk_identified", "home", "l45", "planned_cs_l4")
# Final delivery location -> label, for deaths by place.
PLACES = {0: "home", 1: "l23", 2: "l4", 3: "l5"}
PLACE_LABELS = {"home": "Home", "l23": "L2/3", "l4": "L4", "l5": "L5"}
LEVEL_COLORS = {"base": "#c9c9c9", "current": "#9aa5b1", "moderate": "#5b8db8", "high": "#1f4e79"}
COUNTY_COLORS = {"kisii": "#1f4e79", "makueni": "#c0504d", "mombasa": "#9bbb59"}


def implementation(name, county):
    """Implementation index of a scenario."""
    if name == "base":
        return 0.0
    if name in LEVELS:
        return CURRENT_IMPLEMENTATION_INDEX["prompts"][county] if name == "current" else IMPLEMENTATION_INDEX_BY_LEVEL[name]
    return EXTRA[name][0]


def rr_for(index, county):
    """RR on 4+ ANC at an implementation index, interpolated through the defined levels."""
    points = sorted({0.0: 1.0, CURRENT_IMPLEMENTATION_INDEX["prompts"][county]: PROMPTS_RR_BY_LEVEL["current"],
                     **{IMPLEMENTATION_INDEX_BY_LEVEL[lv]: PROMPTS_RR_BY_LEVEL[lv] for lv in MAIN_LEVELS}}.items())
    return float(np.interp(index, [p[0] for p in points], [p[1] for p in points]))


def configure(name, param, flags, sliders, county):
    """Set PROMPTS for a scenario: the defined levels exactly as scenario_comparison.py does."""
    if name in LEVELS:
        configure_scenario(f"prompts_{name}", param, flags, sliders)
        return
    index, anc_effect, screening_effect = EXTRA[name]
    rr = rr_for(index, county) if anc_effect else 1.0
    flags["flag_PROMPTS"] = 1
    param["HSS"].update({"prompts_implementation_index": index, "prompts_rr_anc4p": rr})
    param["prompts_implementation_index"] = index
    param["prompts_rr_anc4p"] = rr
    if not screening_effect:
        # Registered mothers with 4+ ANC are screened at the workbook targets
        # (sen_risk_trad_target 0.95); hold screening at its baseline instead.
        param["sen_risk_trad_target"] = param["sen_risk_trad"]
        param["spec_risk_trad_target"] = param["spec_risk_trad"]


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
    }


def run_task(task):
    """All scenarios for one county and one run (one parameter draw)."""
    county, run, scenarios = task
    sliders = get_slider_params(county=county)
    parameter_seed = BASE_SEED + run
    month_seeds = np.arange(parameter_seed, parameter_seed + N_MONTHS, dtype=int)
    rows = []
    for name in scenarios:
        param = calculate_derived_parameters(get_parameters(county=county, seed=parameter_seed))
        param.update({"E": reset_E(), "S": reset_S(sliders), "HSS": reset_HSS(sliders)})
        flags = reset_flags()
        configure(name, param, flags, sliders, county)
        _, people, _ = run_model_dash(param, flags, n_months=N_MONTHS, int_period=N_MONTHS, base_seed=month_seeds)
        rows.append({"county": county, "run": run, "level": name,
                     "implementation": implementation(name, county), **measure(people)})
    return rows


def interval(values):
    return values.mean(), values.quantile(0.025), values.quantile(0.975)


def paired(runs, level, outcome, relative=False):
    """Per-run change from base for one scenario and outcome."""
    base = runs[runs.level.eq("base")].set_index(["county", "run"])[outcome]
    group = runs[runs.level.eq(level)].set_index(["county", "run"])[outcome]
    change = group - base.loc[group.index]
    return change / base.loc[group.index] if relative else change


def summary_table(runs):
    """Mean and 95% interval of each outcome, and of its paired change from base
    (relative change for maternal deaths), for every scenario."""
    rows = []
    for (county, level), group in runs.groupby(["county", "level"], sort=False):
        for outcome, label in OUTCOMES.items():
            mean, low, high = interval(group[outcome])
            row = {"county": county, "level": level, "implementation": group.implementation.iloc[0],
                   "outcome": label, "value": mean, "value_low": low, "value_high": high}
            if level != "base":
                change = paired(runs, level, outcome, outcome in RELATIVE).loc[county]
                row.update(zip(("change", "change_low", "change_high"), interval(change)))
            rows.append(row)
    return pd.DataFrame(rows)


def readable(table):
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


def deaths_averted_per_year(runs):
    """Maternal deaths averted per year in each county (expected deaths, paired with base)."""
    years = N_MONTHS / 12
    rows = []
    for county in COUNTIES:
        births = runs[runs.county.eq(county) & runs.level.eq("base")].births.mean() / years
        for level in ("current", *MAIN_LEVELS):
            averted = -paired(runs, level, "expected_deaths").loc[county] / years
            mean, low, high = interval(averted)
            rows.append({"county": county, "level": level, "births_per_year": births,
                         "deaths_averted_per_year": mean, "low": low, "high": high})
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


# ---- Figures ---------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))


def mmr_figure(table, path):
    """Relative change in maternal mortality, moderate and high, by county."""
    plt = _plt()
    mmr = table[table.outcome.eq(OUTCOMES["mmr"]) & table.level.isin(MAIN_LEVELS)]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    width = 0.8 / len(MAIN_LEVELS)
    for i, level in enumerate(MAIN_LEVELS):
        rows = mmr[mmr.level.eq(level)].set_index("county").loc[list(COUNTIES)]
        x = np.arange(len(COUNTIES)) + (i - (len(MAIN_LEVELS) - 1) / 2) * width
        ax.bar(x, -100 * rows.change, width, color=LEVEL_COLORS[level], label=level.title())
        ax.errorbar(x, -100 * rows.change, yerr=[100 * (rows.change_high - rows.change), 100 * (rows.change - rows.change_low)],
                    fmt="none", ecolor="black", capsize=3, linewidth=1)
    ax.set_xticks(np.arange(len(COUNTIES)), [c.title() for c in COUNTIES])
    ax.set_ylabel("Reduction in maternal mortality (%)")
    ax.set_title("PROMPTS: change in maternal mortality vs no PROMPTS")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(title="Implementation", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def averted_figure(averted, path):
    """Stacked bars: deaths averted per 100,000 births by place of delivery."""
    plt = _plt()
    colors = {"Home": "#c0504d", "L2/3": "#9bbb59", "L4": "#4f81bd", "L5": "#1f3864"}
    fig, ax = plt.subplots(figsize=(7, 3.8))
    labels, positions, x = [], [], 0.0
    for county in COUNTIES:
        for level in MAIN_LEVELS:
            rows = averted[averted.county.eq(county) & averted.level.eq(level)].set_index("place")
            bottom_pos = bottom_neg = 0.0
            for place in PLACE_LABELS.values():
                value = rows.loc[place, "averted_per_100k"]
                bottom = bottom_pos if value >= 0 else bottom_neg
                ax.bar(x, value, 0.7, bottom=bottom, color=colors[place], label=place if x == 0 else None)
                if value >= 0:
                    bottom_pos += value
                else:
                    bottom_neg += value
            positions.append(x)
            labels.append(f"{county.title()}\n{level}")
            x += 1
        x += 0.5
    ax.set_xticks(positions, labels, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Maternal deaths averted\nper 100,000 births")
    ax.set_title("PROMPTS: where maternal deaths are averted (place of delivery)")
    ax.legend(title="Place of delivery", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def cascade_figure(table, path):
    """Care cascade: each pathway indicator at base, moderate, and high, by county."""
    plt = _plt()
    labels = {"anc4": "4+ ANC", "highrisk_identified": "High-risk\nidentified", "home": "Home\nbirths",
              "l45": "Births at\nL4/5", "planned_cs_l4": "Planned CS\n(L4 births)"}
    levels = ("base", *MAIN_LEVELS)
    fig, axes = plt.subplots(1, len(COUNTIES), figsize=(10, 3.6), sharey=True)
    width = 0.8 / len(levels)
    for ax, county in zip(axes, COUNTIES):
        for i, level in enumerate(levels):
            rows = table[table.county.eq(county) & table.level.eq(level)].set_index("outcome")
            values = [100 * rows.loc[OUTCOMES[name], "value"] for name in CASCADE]
            x = np.arange(len(CASCADE)) + (i - (len(levels) - 1) / 2) * width
            ax.bar(x, values, width, color=LEVEL_COLORS[level], label=level.title())
        ax.set_xticks(np.arange(len(CASCADE)), [labels[name] for name in CASCADE], fontsize=7)
        ax.set_title(county.title())
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Percent")
    handles, names = axes[0].get_legend_handles_labels()
    fig.suptitle("PROMPTS: care pathway indicators")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, names, title="PROMPTS", loc="lower center", ncol=len(levels), frameon=False, fontsize=8)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))


def mechanism_figure(table, path):
    """Relative change in maternal mortality at high implementation: ANC effect only,
    screening effect only, and both."""
    plt = _plt()
    parts = {"high_anc_only": "More 4+ ANC only", "high_screening_only": "Better screening only", "high": "Both (PROMPTS high)"}
    colors = {"high_anc_only": "#5b8db8", "high_screening_only": "#9bbb59", "high": "#1f4e79"}
    mmr = table[table.outcome.eq(OUTCOMES["mmr"])]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    width = 0.8 / len(parts)
    for i, (level, label) in enumerate(parts.items()):
        rows = mmr[mmr.level.eq(level)].set_index("county").loc[list(COUNTIES)]
        x = np.arange(len(COUNTIES)) + (i - (len(parts) - 1) / 2) * width
        ax.bar(x, -100 * rows.change, width, color=colors[level], label=label)
        ax.errorbar(x, -100 * rows.change, yerr=[100 * (rows.change_high - rows.change), 100 * (rows.change - rows.change_low)],
                    fmt="none", ecolor="black", capsize=3, linewidth=1)
    ax.set_xticks(np.arange(len(COUNTIES)), [c.title() for c in COUNTIES])
    ax.set_ylabel("Reduction in maternal mortality (%)")
    ax.set_title("PROMPTS: contribution of each mechanism (high implementation)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def dose_figure(table, path):
    """Reduction in maternal mortality by implementation index, with each county's
    current implementation marked."""
    plt = _plt()
    mmr = table[table.outcome.eq(OUTCOMES["mmr"])]
    dose_levels = ["current", "moderate", "high", *[name for name in EXTRA if name.startswith("dose_")]]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for county in COUNTIES:
        rows = mmr[mmr.county.eq(county) & mmr.level.isin(dose_levels)].sort_values("implementation")
        x = np.r_[0, 100 * rows.implementation]
        y, low, high = (np.r_[0, -100 * rows[col]] for col in ("change", "change_high", "change_low"))
        ax.plot(x, y, color=COUNTY_COLORS[county], marker="o", markersize=3, label=county.title())
        ax.fill_between(x, low, high, color=COUNTY_COLORS[county], alpha=0.15, linewidth=0)
        current = rows[rows.level.eq("current")].iloc[0]
        ax.plot(100 * current.implementation, -100 * current.change, marker="D", markersize=8,
                markerfacecolor="white", markeredgecolor=COUNTY_COLORS[county], markeredgewidth=1.5)
    ax.set_xlabel("PROMPTS implementation (%)")
    ax.set_ylabel("Reduction in maternal mortality (%)")
    ax.set_title("PROMPTS: dose-response (diamonds: current implementation)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output-dir", default="analysis/results/prompts")
    args = parser.parse_args()

    output = REPO_ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    scenarios = [*LEVELS, *EXTRA]
    tasks = [(county, run, scenarios) for county in COUNTIES
             for run in range(FIRST_ANALYSIS_RUN - 1, FIRST_ANALYSIS_RUN - 1 + args.runs)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = pd.DataFrame([row for rows in pool.map(run_task, tasks) for row in rows])
    runs.to_csv(output / "runs.csv", index=False)

    table = summary_table(runs)
    table.to_csv(output / "summary.csv", index=False)
    readable(table).to_csv(output / "table.csv")
    per_year = deaths_averted_per_year(runs)
    per_year.to_csv(output / "deaths_averted_per_year.csv", index=False)
    averted = averted_by_place(runs)
    averted.to_csv(output / "averted_by_place.csv", index=False)
    mmr_figure(table, output / "mmr_change")
    averted_figure(averted, output / "averted_by_place")
    cascade_figure(table, output / "cascade")
    mechanism_figure(table, output / "mechanism")
    dose_figure(table, output / "dose_response")

    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print(readable(table).to_string())
    print("\nMaternal deaths averted per year (95% interval):")
    print(per_year.round(1).to_string(index=False))
    print(f"\n{args.runs} runs x {N_MONTHS} months per scenario. Saved: {output}")


if __name__ == "__main__":
    main()
