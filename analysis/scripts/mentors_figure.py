"""Preliminary MENTORS figure in the layout of Figure 1 (PROMPTS), from the
mentors_fqa_results.py runs (local only).

    A  reduction in maternal mortality by MENTORS implementation (no MENTORS, current,
       high; diamonds: current)
    B  from treatments to deaths at high implementation

Two versions: MENTORS alone (vs no intervention) and MENTORS with filled supplies
(oxytocin, MgSO4, antibiotics and IV iron at 95% at L4/5; Panel A starts from filled
supplies alone, Panel B is MENTORS' addition to filled supplies). The knowledge gain is
mentors_knowledge_improve (0.103, Rowe et al. 2018 median) x mentors_implementation_index.

All result folders use the same parameter draws and monthly seeds, so every comparison is
paired. Intervals are 95% uncertainty intervals over parameter draws.

Usage:
    python analysis/scripts/mentors_figure.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from paper_figures import (
    COUNTIES, COUNTY_COLORS, PULSE_CHAIN, REPO_ROOT, _plt, chain_plot, county_legend, improvement, panel_label, style,
)
from scenario_comparison import CURRENT_IMPLEMENTATION_INDEX, IMPLEMENTATION_INDEX_BY_LEVEL

RESULT_DIRS = ("analysis/results/mentors_fqa", "analysis/results/mentors_supplies")
# version -> (scenario at each level, reference for Panel B, scenario at no MENTORS, title).
VERSIONS = {
    "alone": ("mentors_{level}", "base", "base", "MENTORS alone"),
    "supplies": ("supplies_mentors_{level}", "supplies", "supplies", "MENTORS with filled supplies"),
}
# Panel B: PULSE's chain without the ANC rows (MENTORS works through facility knowledge only).
MENTORS_CHAIN = {group: rows for group, rows in PULSE_CHAIN.items() if group != "Care"}


def load_replicates():
    """Every replicate of every scenario, one row per county, run, replicate and scenario."""
    frames = [pd.read_csv(REPO_ROOT / folder / "runs_replicates.csv") for folder in RESULT_DIRS]
    return pd.concat(frames, ignore_index=True).drop_duplicates(["county", "run", "replicate", "level"])


def implementation(county, level):
    if level == "current":
        return CURRENT_IMPLEMENTATION_INDEX["mentors"][county]
    return IMPLEMENTATION_INDEX_BY_LEVEL[level]


def dose_points(replicates, county, version):
    """x (% implementation) and reduction in maternal mortality vs no intervention with its
    95% interval, at no MENTORS, current and high."""
    scenario, _, start, _ = VERSIONS[version]
    names = [start, scenario.format(level="current"), scenario.format(level="high")]
    x = np.array([0, 100 * implementation(county, "current"), 100 * implementation(county, "high")])
    bands = np.array([(0.0, 0.0, 0.0) if name == "base" else improvement(replicates, county, name, "base", "mmr", -1)
                      for name in names])
    return x, bands


def figure(replicates, version, path):
    plt = _plt()
    scenario, reference, _, title = VERSIONS[version]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": (1, 1.1), "wspace": 0.55})

    for county in COUNTIES:
        x, bands = dose_points(replicates, county, version)
        y, low, high = bands.T
        ax_a.plot(x, y, color=COUNTY_COLORS[county], marker="o", markersize=3, label=county.title())
        # Light fill with dotted edges so the three overlapping bands stay readable.
        ax_a.fill_between(x, low, high, color=COUNTY_COLORS[county], alpha=0.07, linewidth=0)
        for edge in (low, high):
            ax_a.plot(x, edge, color=COUNTY_COLORS[county], linestyle=":", linewidth=0.9)
        ax_a.plot(x[1], y[1], marker="D", markersize=7, markerfacecolor="white",
                  markeredgecolor=COUNTY_COLORS[county], markeredgewidth=1.5)
    ax_a.axhline(0, color="black", linewidth=0.8)
    ax_a.set_xlim(-3, 100)
    ax_a.set_xlabel("MENTORS implementation index (%)", fontsize=9)
    ax_a.set_ylabel("Reduction in maternal mortality (%)", fontsize=9)
    ax_a.set_title("Maternal mortality by implementation", fontsize=10)
    start_note = "\nStart (0%): filled supplies alone" if version == "supplies" else ""
    ax_a.legend(frameon=False, fontsize=8, title=f"Diamonds: current\nDotted: 95% interval{start_note}",
                title_fontsize=7, loc="upper left")
    style(ax_a)
    panel_label(ax_a, "A")

    chain_plot(ax_b, replicates, MENTORS_CHAIN, ["high"], lambda level: scenario.format(level=level), lambda level: reference)
    ax_b.set_xlabel("Improvement vs " + ("filled supplies alone" if version == "supplies" else "no MENTORS")
                    + " (%)\n(% of the no-intervention value)", fontsize=9)
    ax_b.set_title("From treatments to deaths, high implementation", fontsize=10)
    panel_label(ax_b, "B")
    fig.suptitle(f"Preliminary: {title}", fontsize=11, y=1.0)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def figure_data(replicates):
    """The plotted values: mean and 95% interval, one row per point."""
    rows = []
    for version, (scenario, reference, _, _) in VERSIONS.items():
        for county in COUNTIES:
            x, bands = dose_points(replicates, county, version)
            for level, implementation_pct, band in zip(("none", "current", "high"), x, bands):
                rows.append((version, "A", "Deaths", "Maternal deaths", county, level, implementation_pct, "base", *band))
            for group, outcomes in MENTORS_CHAIN.items():
                for column, label, direction in outcomes:
                    rows.append((version, "B", group, label, county, "high", 100 * implementation(county, "high"), reference,
                                 *improvement(replicates, county, scenario.format(level="high"), reference, column, direction)))
    return pd.DataFrame(rows, columns=["version", "panel", "stage", "outcome", "county", "mentors_level",
                                       "implementation_pct", "reference", "improvement_pct", "low_95", "high_95"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="analysis/results/mentors_figure")
    args = parser.parse_args()
    output = REPO_ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    replicates = load_replicates()
    for version in VERSIONS:
        figure(replicates, version, output / f"figure_mentors_{version}")
    data = figure_data(replicates)
    data.to_csv(output / "figure_mentors_data.csv", index=False)
    print(data[data.panel.eq("A")].round(2).to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
