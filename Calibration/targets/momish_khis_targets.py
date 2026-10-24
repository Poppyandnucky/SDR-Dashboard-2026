"""KHIS targets for Kisii, Makueni, and Mombasa (Stages 3 and 5).

L2/3: preterm share, complication-event ratio L2/3:L4, CS rate.
L4:   cause-specific complication rates and event-rate MCR (x scale_comp), CS rate (raw).
L2/3 and L4 maternal mortality (MOH 711 maternal deaths per 100,000 deliveries)
with exact Poisson 95% intervals, for Stage 6 validation.

Sources (MOMISH KHIS extract, 3rd version):
- Health-related indicators: MOH 711 complications, caesarean sections
  ("Maternal Health") and pre-term babies ("Neonatal Health"), by facility-month.
- Mothers' condition: total deliveries by facility-month (the denominator).

Rules:
- Common period for every county, January 2022 to June 2025 (the Kisii and
  Mombasa extracts; Makueni's also covers 2019-2021 and July 2025). The base is
  every facility-month with deliveries, so numerator-only rows are excluded.
- Facility-months absent from the indicator extract count as zero events. KHIS
  (DHIS2) exports omit unreported or zero values; the indicator file covers
  only about a quarter of L2/3 deliveries but nearly all L4/L5 deliveries, so
  restricting to matched rows would inflate L2/3 rates.
- Levels as coded in KHIS, except Makueni County Referral Hospital (Level 5).
- L5 is excluded from rate comparisons because it receives external referrals:
  the complication ratio compares L2/3 with internal L4.
- Complication events = APH + PPH + eclampsia + obstructed labour + ruptured
  uterus + sepsis, per delivery. Diagnoses (each complication, MCR, preterm)
  are multiplied by scale_comp (4.48) for every cause; CS is a recorded
  procedure and is not scaled.

Variant ``surgical_l23_as_l4`` moves L2/3 facilities that report caesarean
sections on more than 10% of deliveries to L4, as a sensitivity: these are
mostly private maternity and nursing homes with theatres.

Usage:
    python Calibration/targets/momish_khis_targets.py "/path/to/3rd version" \
        Calibration/targets/momish_khis_targets.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import chi2


SCALE_COMP = 4.48
SURGICAL_CS_SHARE = 0.10
# KHIS indicator -> workbook L4 target name.
COMPLICATION_TARGETS = {
    "MOH 711 APH (Ante partum Haemorrage)": "aph_l4_target",
    "MOH 711 PPH (Post Partum Haemorrage)": "pph_l4_target",
    "MOH 711 Eclampsia": "eclampsia_l4_target",
    "MOH 711 Obstructed Labour": "obstructed_labor_l4_target",
    "MOH 711 Ruptured Uterus": "ruptured_uterus_l4_target",
    "MOH 711 Sepsis": "sepsis_l4_target",
}
COMPLICATIONS = tuple(COMPLICATION_TARGETS)



CS = "MOH 711 Caesarian Sections"
PRETERM = "MOH 711 Pre-term babies"
DEATHS = (
    "MOH 711 Rev 2020_Maternal deaths 15-19Years", "MOH 711 Rev 2020_Maternal deaths 20-24 Years",
    "MOH 711 Rev 2020_Maternal deaths 25+ Years",
)
LEVEL_OVERRIDES = {"Makueni County Referral Hospital": "Level 5"}


def load(folder: Path, start: int = 202201, end: int = 202506) -> pd.DataFrame:
    health = folder / "Health-related indicators-20250717_v3.xlsx"
    maternal = pd.read_excel(health, sheet_name="Maternal Health")
    neonatal = pd.read_excel(health, sheet_name="Neonatal Health")[["organisationunitid", "periodid", PRETERM]]
    deliveries = pd.read_excel(folder / "Mothers_ condition_20250717_v3.xlsx", sheet_name="Facility Level")[
        ["County", "Level", "organisationunitid", "organisationunitname", "periodid", "Total deliveries"]
    ]
    frame = (
        deliveries
        .merge(maternal[["organisationunitid", "periodid", *COMPLICATIONS, CS, *DEATHS]],
               on=["organisationunitid", "periodid"], how="left")
        .merge(neonatal, on=["organisationunitid", "periodid"], how="left")
    )
    for column in (*COMPLICATIONS, CS, PRETERM, *DEATHS, "Total deliveries"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    frame = frame[frame["Total deliveries"].gt(0) & frame["periodid"].between(start, end)].copy()
    for facility, level in LEVEL_OVERRIDES.items():
        frame.loc[frame["organisationunitname"].eq(facility), "Level"] = level
    frame["events"] = frame[list(COMPLICATIONS)].sum(axis=1)
    frame["deaths"] = frame[list(DEATHS)].sum(axis=1)
    frame["group"] = frame["Level"].map({"Level 2": "L23", "Level 3": "L23", "Level 4": "L4", "Level 5": "L5"})
    return frame.dropna(subset=["group"])


def reclassify_surgical_l23(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    l23 = frame[frame["group"].eq("L23")]
    totals = l23.groupby("organisationunitid")[[CS, "Total deliveries"]].sum()
    surgical = totals.index[totals[CS] / totals["Total deliveries"] > SURGICAL_CS_SHARE]
    frame.loc[frame["organisationunitid"].isin(surgical) & frame["group"].eq("L23"), "group"] = "L4"
    return frame


def mortality(totals: pd.DataFrame, group: str, name: str) -> dict:
    """MMR per 100,000 deliveries with an exact (Garwood) Poisson 95% interval."""
    deaths, deliveries = totals.loc[group, "deaths"], totals.loc[group, "Total deliveries"]
    lower = chi2.ppf(0.025, 2 * deaths) / 2 if deaths > 0 else 0.0
    upper = chi2.ppf(0.975, 2 * deaths + 2) / 2
    scale = 100_000 / deliveries
    return {
        f"{name}_deaths": deaths, name: deaths * scale,
        f"{name}_lower": lower * scale, f"{name}_upper": upper * scale,
    }


def summarize(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows = []
    for county, data in frame.groupby("County"):
        totals = data.groupby("group")[["events", "deaths", *COMPLICATIONS, CS, PRETERM, "Total deliveries"]].sum()
        rate = lambda column, group: totals.loc[group, column] / totals.loc[group, "Total deliveries"]
        rows.append({
            "county": county.lower(), "variant": variant,
            "period": f"{data['periodid'].min()}-{data['periodid'].max()}",
            "facility_months": len(data),
            "deliveries_l23": totals.loc["L23", "Total deliveries"],
            "deliveries_l4": totals.loc["L4", "Total deliveries"],
            "mcr_l23": rate("events", "L23"), "mcr_l4": rate("events", "L4"),
            "mcr_ratio_l23_l4": rate("events", "L23") / rate("events", "L4"),
            "preterm_l23_raw": rate(PRETERM, "L23"),
            "preterm_l23": SCALE_COMP * rate(PRETERM, "L23"),
            "preterm_l4_raw": rate(PRETERM, "L4"),
            "cs_l23": rate(CS, "L23"), "cs_l4": rate(CS, "L4"),
            "p_mcr_l4": SCALE_COMP * rate("events", "L4"),
            **{name: SCALE_COMP * rate(column, "L4") for column, name in COMPLICATION_TARGETS.items()},
            **mortality(totals, "L23", "mmr_l23_khis"),
            **mortality(totals, "L4", "mmr_l4_khis"),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("output")
    parser.add_argument("--start", type=int, default=202201, help="First period (YYYYMM).")
    parser.add_argument("--end", type=int, default=202506, help="Last period (YYYYMM).")
    args = parser.parse_args()

    frame = load(Path(args.folder), args.start, args.end)
    result = pd.concat([
        summarize(frame, "khis_levels"),
        summarize(reclassify_surgical_l23(frame), "surgical_l23_as_l4"),
    ], ignore_index=True)
    result.to_csv(args.output, index=False)
    pd.set_option("display.width", 250)
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
