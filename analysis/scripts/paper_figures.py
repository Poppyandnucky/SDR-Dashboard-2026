"""The paper's two figures and Table 1, from the PROMPTS and PULSE results.

Figure 1 (PROMPTS):
    A  reduction in maternal mortality by implementation (dose-response; diamonds:
       current implementation)
    B  improvement along the chain at high implementation: care steps ->
       complications -> deaths (% change vs no PROMPTS, reductions shown as positive)
Figure 2 (PULSE at its current implementation added to PROMPTS at current): nine panels,
    care -> treatments -> complications -> deaths, in natural units; improvement from no
    intervention split into PROMPTS alone and what PULSE adds (bar: penetration 1; tick:
    penetration 0.5)
Supplementary PULSE figure: every outcome, PROMPTS at current and high, one panel per
    outcome (figure_s_pulse_panels). pulse_chain_figure (Figure 1B's layout as % change) is
    kept but no longer drawn.
Table 1: PROMPTS maternal mortality and deaths averted per year by county and level;
    footnotes in table1_prompts_notes.txt.

Neonatal deaths are left out: the model does not represent them well enough.
Every estimate is the mean over parameter draws of a paired change (or value) averaged
over each draw's replicates (runs_replicates.csv); every interval is the 2.5th-97.5th
percentile over draws, covering parameter uncertainty and the simulation noise left
after averaging. Figures 1 and 2 show relative changes (% of the no-intervention
value); the Figure 2 panels version and Table 1's deaths averted, natural units.

Run prompts_pulse_results.py first.

Usage:
    python analysis/scripts/paper_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from prompts_pulse_results import COUNTIES, N_MONTHS, OUTCOMES as PROMPTS_OUTCOMES

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_LEVELS = ("current", "high")
COUNTY_COLORS = {"kisii": "#1f4e79", "makueni": "#c0504d", "mombasa": "#9bbb59"}
MMR = PROMPTS_OUTCOMES["mmr"]
# Chain rows: group -> [(runs.csv column, label, +1 if an increase is an improvement, -1 if a decrease)].
PROMPTS_CHAIN = {
    "Care": [("anc4", "4+ ANC visits", 1), ("highrisk_identified", "High-risk women identified", 1),
             ("l45", "Births at L4/5", 1)],
    "Complications": [("per1000_i_anemia_new", "Anaemia at delivery", -1), ("preterm", "Preterm births", -1),
                      ("per1000_i_severe_new", "Severe maternal complications", -1)],
    "Deaths": [("mmr", "Maternal deaths", -1)],
}
# Figure 2 rows, as PROMPTS_CHAIN.
PULSE_CHAIN = {
    "Care": [("anc4", "4+ ANC visits", 1), ("highrisk_identified", "High-risk women identified", 1)],
    "Treatments": [("treated_i_iv_iron", "IV iron in ANC", 1), ("treated_i_oxytocin", "Oxytocin", 1),
                   ("treated_i_MgSO4", "MgSO4", 1), ("treated_i_antibiotics", "Antibiotics", 1)],
    "Complications": [("per1000_i_anemia_new", "Anaemia at delivery", -1), ("per1000_i_pph_new", "PPH", -1),
                      ("per1000_i_eclampsia_new", "Eclampsia", -1), ("per1000_i_mat_sepsis_new", "Maternal sepsis", -1),
                      ("per1000_i_severe_new", "Severe maternal complications", -1)],
    "Deaths": [("mmr", "Maternal deaths", -1)],
}
# Supplementary Figure 2 panels: group -> [(runs.csv column, label with units, scale, +1 if an increase is an improvement, -1 if a decrease)].
PULSE_PANELS = {
    "Care": [("prompts_registered", "Registered with PROMPTS\n(percentage points)", 100, 1),
             ("anc4", "4+ ANC visits\n(percentage points)", 100, 1),
             ("highrisk_identified", "High-risk women identified\n(percentage points)", 100, 1)],
    "Treatments": [("treated_i_iv_iron", "IV iron in ANC\n(per 1,000 births)", 1, 1),
                   ("treated_i_oxytocin", "Oxytocin\n(per 1,000 births)", 1, 1),
                   ("treated_i_MgSO4", "MgSO4\n(per 1,000 births)", 1, 1),
                   ("treated_i_antibiotics", "Antibiotics\n(per 1,000 births)", 1, 1)],
    "Complications": [("per1000_i_anemia_new", "Anaemia at delivery\n(per 1,000 births)", 1, -1),
                      ("per1000_i_pph_new", "PPH\n(per 1,000 births)", 1, -1),
                      ("per1000_i_eclampsia_new", "Eclampsia\n(per 1,000 births)", 1, -1),
                      ("per1000_i_mat_sepsis_new", "Maternal sepsis\n(per 1,000 births)", 1, -1),
                      ("per1000_i_severe_new", "Severe maternal complications\n(per 1,000 births)", 1, -1)],
    "Deaths": [("mmr", "Maternal deaths\n(per 100,000 births)", 1, -1)],
}
# Figure 2: nine of the panels above in a 3 x 3 grid, one row per stage (complications and
# deaths share the last row), read left to right.
PULSE_MAIN_PANELS = [
    ("Care", "prompts_registered", "Registered with PROMPTS\n(percentage points)", 100, 1),
    ("Care", "anc4", "4+ ANC visits\n(percentage points)", 100, 1),
    ("Care", "highrisk_identified", "High-risk women identified\n(percentage points)", 100, 1),
    ("Treatments", "treated_i_iv_iron", "IV iron in ANC\n(per 1,000 births)", 1, 1),
    ("Treatments", "treated_i_oxytocin", "Oxytocin\n(per 1,000 births)", 1, 1),
    ("Treatments", "treated_i_MgSO4", "MgSO4\n(per 1,000 births)", 1, 1),
    ("Complications", "per1000_i_anemia_new", "Anaemia at delivery\n(per 1,000 births)", 1, -1),
    ("Complications", "per1000_i_eclampsia_new", "Eclampsia\n(per 1,000 births)", 1, -1),
    ("Deaths", "mmr", "Maternal deaths\n(per 100,000 births)", 1, -1),
]
LEVEL_MARKERS = {"current": "o", "high": "s"}


def panel_label(ax, letter):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", ha="right", va="bottom")


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def marker_style(county, level):
    return dict(fmt=LEVEL_MARKERS.get(level, "o"), color=COUNTY_COLORS[county],
                markerfacecolor="white" if level == "current" else COUNTY_COLORS[county],
                markersize=4.5, capsize=2, linewidth=1)


def by_replicate(replicates, county, scenario, column, reference=None, relative=False, scale=1.0):
    """Draw x replicate table of one outcome: its value in the scenario, or with a
    reference its change from the reference paired within each replicate; relative
    divides the change by the no-intervention (base) value."""
    keyed = replicates[replicates.county.eq(county)].set_index(["run", "replicate"])[["level", column]]
    values = keyed[keyed.level.eq(scenario)][column]
    if reference is not None:
        values = values - keyed[keyed.level.eq(reference)][column].loc[values.index]
    if relative:
        values = values / keyed[keyed.level.eq("base")][column].loc[values.index]
    return (scale * values).unstack("replicate")


def parameter_interval(table):
    """Mean and 95% interval over parameter draws from a draw x replicate table: each
    draw is averaged over its replicates, then the 2.5th and 97.5th percentiles are taken
    over draws. The interval covers parameter uncertainty and the simulation noise left
    after averaging."""
    draw_mean = table.mean(axis=1)
    return draw_mean.mean(), draw_mean.quantile(0.025), draw_mean.quantile(0.975)


def improvement(replicates, county, scenario, reference, column, direction):
    """% improvement of one outcome from the reference, as % of the no-intervention value."""
    return parameter_interval(by_replicate(replicates, county, scenario, column, reference, True, 100 * direction))


def chain_plot(ax, runs, chain, levels, scenario_of, reference_of):
    """Dots with 95% intervals, one row per outcome grouped by stage of the chain,
    one dot per county (color) and level (marker)."""
    series = [(county, level) for county in COUNTIES for level in levels]
    offsets = np.linspace(-0.28, 0.28, len(series))
    ticks, labels, y = [], [], 0.0
    for group, outcomes in chain.items():
        ax.text(-0.02, y, group, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=8.5, fontweight="bold")
        y += 0.7
        for column, label, direction in outcomes:
            for (county, level), offset in zip(series, offsets):
                value, low, high = improvement(runs, county, scenario_of(level), reference_of(level), column, direction)
                ax.errorbar(value, y + offset, xerr=[[value - low], [high - value]], **marker_style(county, level))
            ticks.append(y)
            labels.append(label)
            y += 1
        y += 0.2
    ax.set_yticks(ticks, labels, fontsize=8)
    ax.set_ylim(y - 0.5, -0.4)
    ax.axvline(0, color="black", linewidth=0.8)
    style(ax)


def county_legend(fig, levels=None, **kwargs):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COUNTY_COLORS[c], marker="o", linestyle="", label=c.title()) for c in COUNTIES]
    for level in levels or ():
        handles.append(Line2D([], [], color="black", marker=LEVEL_MARKERS[level], linestyle="",
                              markerfacecolor="white" if level == "current" else "black",
                              label=f"{level.title()} implementation"))
    fig.legend(handles=handles, frameon=False, fontsize=8, **kwargs)


def figure1(prompts, replicates, path):
    plt = _plt()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": (1, 1.1), "wspace": 0.55})

    mmr = prompts[prompts.outcome.eq(MMR)]
    dose_levels = ["current", "high", *sorted(lv for lv in mmr.level.unique() if lv.startswith("dose_"))]
    for county in COUNTIES:
        rows = mmr[mmr.county.eq(county) & mmr.level.isin(dose_levels)].sort_values("implementation")
        x = np.r_[0, 100 * rows.implementation]
        bands = np.array([improvement(replicates, county, level, "base", "mmr", -1) for level in rows.level])
        y, low, high = (np.r_[0, bands[:, k]] for k in range(3))
        ax_a.plot(x, y, color=COUNTY_COLORS[county], marker="o", markersize=3, label=county.title())
        # Light fill with dotted edges so the three overlapping bands stay readable.
        ax_a.fill_between(x, low, high, color=COUNTY_COLORS[county], alpha=0.07, linewidth=0)
        for edge in (low, high):
            ax_a.plot(x, edge, color=COUNTY_COLORS[county], linestyle=":", linewidth=0.9)
        current = rows[rows.level.eq("current")].iloc[0]
        ax_a.plot(100 * current.implementation, y[1 + list(rows.level).index("current")], marker="D", markersize=7,
                  markerfacecolor="white", markeredgecolor=COUNTY_COLORS[county], markeredgewidth=1.5)
    ax_a.axhline(0, color="black", linewidth=0.8)
    ax_a.set_xlabel("PROMPTS implementation (% of mothers registered)", fontsize=9)
    ax_a.set_ylabel("Reduction in maternal mortality (%)", fontsize=9)
    ax_a.set_title("Maternal mortality by implementation", fontsize=10)
    ax_a.legend(frameon=False, fontsize=8, title="Diamonds: current\nDotted: 95% interval", title_fontsize=7,
                loc="upper left")
    style(ax_a)
    panel_label(ax_a, "A")

    chain_plot(ax_b, replicates, PROMPTS_CHAIN, ["high"], lambda level: level, lambda level: "base")
    ax_b.set_xlabel("Improvement vs no PROMPTS (%)", fontsize=9)
    ax_b.set_title("From care to deaths, high implementation", fontsize=10)
    panel_label(ax_b, "B")
    save(fig, path)


def pulse_chain_figure(runs, path, level="high"):
    """Supplementary: Figure 1B's layout for PULSE (current implementation) added to PROMPTS at the given
    level: one row per outcome, one bar per county. Each bar starts at no intervention: the
    light part is PROMPTS alone, the dark part what PULSE adds on top at penetration 1
    (whisker: its 95% interval), the black tick where PULSE's addition ends at penetration
    0.5; all as % of the no-intervention value. Improvements point right."""
    from matplotlib.patches import Patch
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 7))
    offsets = np.linspace(-0.27, 0.27, len(COUNTIES))
    ticks, labels, y = [], [], 0.0
    for group, outcomes in PULSE_CHAIN.items():
        ax.text(-0.02, y, group, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=8.5, fontweight="bold")
        y += 0.7
        for column, label, direction in outcomes:
            for county, offset in zip(COUNTIES, offsets):
                start = improvement(runs, county, f"prompts_{level}", "base", column, direction)[0]
                mean, low, high = improvement(runs, county, f"prompts_{level}_pulse_current_upper", f"prompts_{level}", column, direction)
                lower = improvement(runs, county, f"prompts_{level}_pulse_current_lower", f"prompts_{level}", column, direction)[0]
                color = COUNTY_COLORS[county]
                ax.barh(y + offset, start, height=0.25, color=color, alpha=0.3, linewidth=0)
                ax.barh(y + offset, mean, left=start, height=0.25, color=color, linewidth=0)
                if start < 0:
                    # PROMPTS alone lowers it: fade the part of PULSE's bar that only makes up the loss.
                    ax.barh(y + offset, start, height=0.25, color="white", alpha=0.6, linewidth=0)
                ax.errorbar(start + mean, y + offset, xerr=[[mean - low], [high - mean]], color="black",
                            linewidth=0.8, capsize=1.5)
                ax.plot([start + lower] * 2, [y + offset - 0.15, y + offset + 0.15], color="black", linewidth=1.4)
            ticks.append(y)
            labels.append(label)
            y += 1
        y += 0.2
    ax.set_yticks(ticks, labels, fontsize=8)
    ax.set_ylim(y - 0.5, -0.4)
    ax.set_xlim(left=min(ax.get_xlim()[0], -4))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Improvement vs no intervention (% of no-intervention value)", fontsize=9)
    ax.set_title(f"PULSE (current) added to PROMPTS ({level} implementation)", fontsize=10)
    style(ax)
    from matplotlib.lines import Line2D
    handles = [Patch(color=COUNTY_COLORS[c], label=c.title()) for c in COUNTIES]
    handles += [Patch(color="grey", alpha=0.3, label="PROMPTS alone"),
                Patch(color="grey", label="Added by PULSE, penetration 1\n(whisker: 95% interval)"),
                Line2D([], [], color="black", marker="|", linestyle="", markersize=9, markeredgewidth=1.4,
                       label="Added by PULSE, penetration 0.5")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    save(fig, path)


def figure2_panels(runs, path):
    """Supplementary version of Figure 2: every outcome, PROMPTS at current and high, one
    small panel per outcome (bars as pulse_bar). Improvements point right."""
    plt = _plt()
    ncols = max(len(panels) for panels in PULSE_PANELS.values())
    fig, axes = plt.subplots(len(PULSE_PANELS), ncols, figsize=(14, 10),
                             gridspec_kw={"hspace": 0.95, "wspace": 0.15})
    bars = [(county, level) for county in COUNTIES for level in MAIN_LEVELS]
    y = np.arange(len(bars)) + np.repeat(np.arange(len(COUNTIES)) * 0.6, len(MAIN_LEVELS))
    for row, (group, panels) in enumerate(PULSE_PANELS.items()):
        for col in range(ncols):
            ax = axes[row, col]
            if col >= len(panels):
                ax.axis("off")
                continue
            column, label, scale, direction = panels[col]
            for (county, level), position in zip(bars, y):
                pulse_bar(ax, runs, county, level, column, scale * direction, position)
            # Level labels only; county is shown by colour (legend), PROMPTS level by the legend title.
            ax.set_yticks(y, [level.title() for _, level in bars] if col == 0 else [], fontsize=7.5)
            if col == 0:
                ax.text(-0.25, 1.42, group, transform=ax.transAxes, fontsize=10, fontweight="bold")
            ax.set_ylim(y[-1] + 0.6, -0.6)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(label, fontsize=8.5)
            style(ax)
    from matplotlib.patches import Patch
    counties = [Patch(color=COUNTY_COLORS[c], label=c.title()) for c in COUNTIES]
    # Legend in the empty cells beside the deaths panel, top-aligned with it: counties in
    # the first column, bar parts in the second.
    axes[-1, 1].legend(handles=counties + pulse_handles(), loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2,
                       frameon=False, fontsize=9, alignment="left",
                       title="Rows: PROMPTS at current or high implementation. Improvement from no intervention\n"
                             "(reductions in complications and deaths shown as positive)",
                       title_fontsize=8.5)
    save(fig, path)


def pulse_bar(ax, runs, county, level, column, scale, position, height=0.8):
    """One bar from no intervention, in natural units: light part PROMPTS alone at the
    level, dark part what PULSE (current) adds at penetration 1 (whisker: 95% interval of
    the addition), black tick where the addition ends at penetration 0.5."""
    prompts = by_replicate(runs, county, f"prompts_{level}", column, "base", scale=scale).mean().mean()
    added = by_replicate(runs, county, f"prompts_{level}_pulse_current_upper", column, f"prompts_{level}", scale=scale)
    mean, low, high = parameter_interval(added)
    lower = by_replicate(runs, county, f"prompts_{level}_pulse_current_lower", column, f"prompts_{level}",
                         scale=scale).mean().mean()
    color = COUNTY_COLORS[county]
    ax.barh(position, prompts, height=height, color=color, alpha=0.3, linewidth=0)
    ax.barh(position, mean, left=prompts, height=height, color=color, linewidth=0)
    ax.errorbar(prompts + mean, position, xerr=[[mean - low], [high - mean]], color="black",
                linewidth=0.8, capsize=1.5)
    ax.plot([prompts + lower] * 2, [position - 0.45 * height, position + 0.45 * height], color="black", linewidth=1.4)


def pulse_handles():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    return [Patch(color="grey", alpha=0.3, label="PROMPTS alone"),
            Patch(color="grey", label="Added by PULSE (current), penetration 1 (whisker: 95% interval)"),
            Line2D([], [], color="black", marker="|", linestyle="", markersize=9, markeredgewidth=1.4,
                   label="Added by PULSE (current), penetration 0.5")]


def figure2(runs, path, level="current"):
    """PULSE (current) added to PROMPTS at the given level, in natural units: nine panels
    (PULSE_MAIN_PANELS) in a 3 x 3 grid, one bar per county. Improvements point right."""
    plt = _plt()
    fig, axes = plt.subplots(3, 3, figsize=(10, 7.6), gridspec_kw={"hspace": 0.75, "wspace": 0.15})
    y = np.arange(len(COUNTIES))
    for letter, ax, (group, column, label, scale, direction) in zip("ABCDEFGHI", axes.flat, PULSE_MAIN_PANELS):
        for county, position in zip(COUNTIES, y):
            pulse_bar(ax, runs, county, level, column, scale * direction, position, height=0.65)
        # County names on the first column only, so they don't run into the panel to the left.
        ax.set_yticks(y, [county.title() for county in COUNTIES] if ax in axes[:, 0] else [], fontsize=8)
        for tick, county in zip(ax.get_yticklabels(), COUNTIES):
            tick.set_color(COUNTY_COLORS[county])
            tick.set_fontweight("bold")
        ax.set_ylim(y[-1] + 0.6, -0.6)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{group}: {label}", fontsize=8.5)
        style(ax)
        panel_label(ax, letter)
    fig.legend(handles=pulse_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False,
               fontsize=8, title=f"Improvement from no intervention, PROMPTS at {level} implementation "
                                 "(reductions in complications and deaths shown as positive)", title_fontsize=8)
    save(fig, path)


def exceedance_figure(replicates, path):
    """Acceptability-style curves, as in cost-effectiveness analysis: for each threshold
    reduction in maternal mortality, the share of parameter draws in which PROMPTS
    reaches at least that reduction (each draw averaged over its replicates)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    thresholds = np.linspace(-2, 12, 281)
    for county in COUNTIES:
        for level, style_ in (("current", "--"), ("high", "-")):
            reduction = by_replicate(replicates, county, level, "mmr", "base", True, -100).mean(axis=1)
            share = [(reduction >= t).mean() for t in thresholds]
            ax.plot(thresholds, share, style_, color=COUNTY_COLORS[county], linewidth=1.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Reduction in maternal mortality of at least (%)", fontsize=9)
    ax.set_ylabel("Share of parameter draws", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_title("Chance PROMPTS reaches a given reduction", fontsize=10)
    style(ax)
    # One legend: county colours, then line styles for the level.
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COUNTY_COLORS[c], linewidth=1.5, label=c.title()) for c in COUNTIES]
    handles += [Line2D([], [], color="black", linestyle="-", label="High implementation"),
                Line2D([], [], color="black", linestyle="--", label="Current implementation")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    save(fig, path)


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


TABLE1_NOTES = """\
Table 1 notes
Values are means over 200 parameter draws (each averaged over 20 replicates), with 95%
    intervals (2.5th to 97.5th percentile over draws).
Births per year (with the county): live births simulated in the no-PROMPTS baseline,
    averaged over draws and replicates; the model draws them from the workbook's monthly
    live births by place of delivery (county_supply, base_LB: home, L2/3, L4, L5).
Baseline MMR: maternal deaths per 100,000 births without PROMPTS. Its wide interval is
    uncertainty in the calibrated death risk; the reductions are much narrower because each
    scenario is compared with the baseline within the same draw and replicate.
Implementation: PROMPTS implementation index (adoption x fidelity x penetration), the
    share of mothers registered; current is the county's present implementation, high 95%.
Reduction in MMR: % reduction in maternal deaths per 100,000 births from the baseline
    (positive = fewer deaths; a negative lower bound means an increase cannot be ruled out).
Deaths averted per year: expected maternal deaths averted per year in the county; the
    counts are small because maternal deaths are rare.
"""


def table1(replicates):
    """PROMPTS by county, one row per county: baseline maternal mortality, then for current
    and high implementation the implementation index, the % reduction in maternal mortality
    and the maternal deaths averted per year (95% intervals over parameter draws).
    Reductions are positive, as in the figures. Columns are (group, measure) pairs."""
    years = N_MONTHS / 12

    def text(table, fmt, unit=""):
        mean, low, high = parameter_interval(table)
        return f"{mean:{fmt}}{unit} ({low:{fmt}} to {high:{fmt}})"

    rows = []
    for county in COUNTIES:
        births = replicates[replicates.county.eq(county) & replicates.level.eq("base")].births.mean() / years
        row = {("", "County"): f"{county.title()} ({births:,.0f} births/year)",
               ("Baseline", "MMR per 100,000 births"): text(by_replicate(replicates, county, "base", "mmr"), ".1f")}
        for level in MAIN_LEVELS:
            group = f"{level.title()} implementation"
            implementation = replicates[replicates.county.eq(county) & replicates.level.eq(level)].implementation.iloc[0]
            reduction = by_replicate(replicates, county, level, "mmr", "base", True, -100)
            averted = by_replicate(replicates, county, level, "expected_deaths", "base", scale=-1 / years)
            row[(group, "Implementation")] = f"{100 * implementation:.1f}%"
            row[(group, "Reduction in MMR")] = text(reduction, ".1f", "%")
            row[(group, "Deaths averted per year")] = text(averted, ".1f")
        rows.append(row)
    return pd.DataFrame(rows, columns=pd.MultiIndex.from_tuples(rows[0]))


def save_table1(table, output):
    """CSV with flat 'group: measure' headers, and an Excel copy with the grouped header."""
    flat = table.copy()
    flat.columns = [f"{group}: {measure}" if group else measure for group, measure in table.columns]
    flat.to_csv(output / "table1_prompts.csv", index=False)
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    book = Workbook()
    sheet = book.active
    sheet.append([group for group, _ in table.columns])
    sheet.append([measure for _, measure in table.columns])
    for row in table.itertuples(index=False):
        sheet.append(list(row))
    groups = [group for group, _ in table.columns]
    for group in dict.fromkeys(g for g in groups if g):
        first, last = groups.index(group) + 1, len(groups) - groups[::-1].index(group)
        sheet.merge_cells(start_row=1, start_column=first, end_row=1, end_column=last)
    for cell in [*sheet[1], *sheet[2]]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    book.save(output / "table1_prompts.xlsx")
    (output / "table1_prompts_notes.txt").write_text(TABLE1_NOTES)
    return flat


def figure_data(prompts, replicates, pulse):
    """The numbers behind Figures 1 and 2, one row per plotted value: mean and 95%
    interval over parameter draws, as plotted (improvements positive)."""
    rows = []

    def add(figure, panel, stage, column, label, county, level, quantity, unit, values, implementation=None):
        mean, low, high = values
        rows.append({"figure": figure, "panel": panel, "stage": stage, "outcome": label, "column": column,
                     "county": county.title(), "prompts_level": level,
                     "prompts_implementation_pct": None if implementation is None else round(100 * implementation, 1),
                     "quantity": quantity, "unit": unit, "mean": mean, "low_95": low, "high_95": high})

    mmr = prompts[prompts.outcome.eq(MMR)]
    dose_levels = ["current", "high", *sorted(lv for lv in mmr.level.unique() if lv.startswith("dose_"))]
    for county in COUNTIES:
        levels = mmr[mmr.county.eq(county) & mmr.level.isin(dose_levels)].sort_values("implementation")
        for level, implementation in zip(levels.level, levels.implementation):
            add("Figure 1", "A", "Deaths", "mmr", "Maternal deaths", county, level,
                "Reduction vs no PROMPTS", "% of no-PROMPTS value",
                improvement(replicates, county, level, "base", "mmr", -1), implementation)
    for stage, outcomes in PROMPTS_CHAIN.items():
        for column, label, direction in outcomes:
            for county in COUNTIES:
                add("Figure 1", "B", stage, column, label, county, "high", "Improvement vs no PROMPTS",
                    "% of no-PROMPTS value", improvement(replicates, county, "high", "base", column, direction))

    level = "current"
    for letter, (stage, column, label, scale, direction) in zip("ABCDEFGHI", PULSE_MAIN_PANELS):
        label, unit = label.split("\n")
        unit = unit.strip("()")
        scale *= direction
        for county in COUNTIES:
            parts = {
                "PROMPTS alone vs no intervention": (f"prompts_{level}", "base"),
                "Added by PULSE (current), penetration 1": (f"prompts_{level}_pulse_current_upper", f"prompts_{level}"),
                "Added by PULSE (current), penetration 0.5": (f"prompts_{level}_pulse_current_lower", f"prompts_{level}"),
                "PROMPTS + PULSE (penetration 1) vs no intervention": (f"prompts_{level}_pulse_current_upper", "base"),
            }
            for quantity, (scenario, reference) in parts.items():
                add("Figure 2", letter, stage, column, label, county, level, quantity, f"improvement, {unit}",
                    parameter_interval(by_replicate(pulse, county, scenario, column, reference, scale=scale)))
    return pd.DataFrame(rows).round({"mean": 3, "low_95": 3, "high_95": 3})


def figure_data_wide(data):
    """figure_data for reading: one row per plotted point or bar part, counties as columns,
    each cell "mean (low to high)"."""
    data = data.copy()
    number = lambda x: f"{x:.2f}".replace("-0.00", "0.00")
    data["cell"] = [f"{number(m)} ({number(lo)} to {number(hi)})"
                    for m, lo, hi in zip(data["mean"], data.low_95, data.high_95)]
    level_names = {"current": "Current", "high": "High"}
    figure_1a = data.figure.eq("Figure 1") & data.panel.eq("A")
    # Current implementation differs by county, so Figure 1A's rows name each county's value.
    current = data[figure_1a & data.prompts_level.eq("current")]
    level_names["current"] = "Current (" + ", ".join(
        f"{c} {p:g}%" for c, p in zip(current.county, current.prompts_implementation_pct)) + ")"
    data.loc[figure_1a, "quantity"] = [
        level_names.get(lv, f"{p:g}% implementation") for lv, p in zip(data.prompts_level[figure_1a],
                                                                       data.prompts_implementation_pct[figure_1a])]
    keys = ["figure", "panel", "stage", "outcome", "unit", "quantity"]
    wide = data.pivot_table(index=keys, columns="county", values="cell", aggfunc="first", sort=False)
    return wide[[c.title() for c in COUNTIES]].reset_index().rename_axis(columns=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-dir", default="analysis/results/prompts")
    parser.add_argument("--pulse-dir", default="analysis/results/pulse")
    parser.add_argument("--output-dir", default="analysis/results/paper")
    args = parser.parse_args()

    prompts_dir, pulse_dir, output = (REPO_ROOT / d for d in (args.prompts_dir, args.pulse_dir, args.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    prompts = pd.read_csv(prompts_dir / "runs_replicates.csv")
    summary = pd.read_csv(prompts_dir / "summary.csv")
    figure1(summary, prompts, output / "figure1_prompts")
    exceedance_figure(prompts, output / "figure_s_prompts_exceedance")
    pulse = pd.read_csv(pulse_dir / "runs_replicates.csv")
    figure2(pulse, output / "figure2_pulse")
    figure2_panels(pulse, output / "figure_s_pulse_panels")
    data = figure_data(summary, prompts, pulse)
    data.to_csv(output / "figures_1_2_data.csv", index=False)
    figure_data_wide(data).to_csv(output / "figures_1_2_data_wide.csv", index=False)
    print(save_table1(table1(prompts), output).to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
