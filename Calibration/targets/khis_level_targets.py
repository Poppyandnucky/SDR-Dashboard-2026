"""KHIS 2022 facility-level delivery split for routing calibration (Phase 2).

Reads the MOMISH workbook ``L23_ProportionLB_20260430.xlsx``. Kisii, Makueni,
and Mombasa use ``Total deliveries`` from the ``Facility Level`` sheet;
Kakamega uses MOH 711 live births from ``live_birth`` with levels from
``Facilities``. KHIS records the facility where delivery occurred, so these
are final (post-transfer) locations.

Two outputs per county:
- ``base``: KHIS levels with reviewed corrections in ``LEVEL_OVERRIDES``.
- ``l4_name_reclass``: sensitivity moving Level 4 facilities whose names or
  ownership do not indicate a hospital to L2/3.

Usage:
    python Calibration/targets/khis_level_targets.py /path/to/L23_ProportionLB_20260430.xlsx \
        Calibration/targets/khis_level_targets.csv
"""

from __future__ import annotations

import argparse
import re

import pandas as pd


YEAR = 2022
LEVEL_OVERRIDES = {
    # Makueni County Referral Hospital is Level 5; KHIS labels it Level 4.
    ("makueni", "Makueni County Referral Hospital"): 5,
}
# Public maternity hospital; Level 4 despite the "maternity" name.
HOSPITAL_EXCEPTIONS = {"Mrima Maternity Hospital"}
HOSPITAL_NAME = re.compile(r"hosp|hospial", re.IGNORECASE)


def load_khis(path: str) -> pd.DataFrame:
    deliveries = pd.read_excel(path, sheet_name="Facility Level")
    deliveries = deliveries[deliveries["periodid"].astype(str).str[:4] == str(YEAR)]
    deliveries = (
        deliveries.assign(births=pd.to_numeric(deliveries["Total deliveries"], errors="coerce"))
        .groupby(["County", "Level", "organisationunitname"], as_index=False)["births"].sum()
    )
    deliveries = pd.DataFrame({
        "county": deliveries["County"].str.lower(),
        "facility": deliveries["organisationunitname"],
        "level": deliveries["Level"].str.extract(r"(\d)", expand=False).astype(int),
        "births": deliveries["births"],
        "measure": "total deliveries",
    })

    live_births = pd.read_excel(path, sheet_name="live_birth")
    kakamega = pd.DataFrame({
        "county": "kakamega",
        "facility": live_births["organisationunitname"],
        "level": live_births["Level Facility"],
        "births": live_births[YEAR],
        "measure": "MOH 711 live births",
    })
    frame = pd.concat([deliveries, kakamega], ignore_index=True)
    frame = frame[frame["births"] > 0]
    if frame["level"].isna().any():
        raise ValueError("Facilities with births must have a KHIS level")
    frame["level"] = frame["level"].astype(int)

    for (county, facility), level in LEVEL_OVERRIDES.items():
        mask = frame["county"].eq(county) & frame["facility"].eq(facility)
        if mask.sum() != 1:
            raise ValueError(f"Level override did not match exactly one facility: {facility}")
        frame.loc[mask, "level"] = level
    return frame


def reclassify_non_hospital_l4(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    non_hospital = (
        frame["level"].eq(4)
        & ~frame["facility"].str.contains(HOSPITAL_NAME)
        & ~frame["facility"].isin(HOSPITAL_EXCEPTIONS)
    )
    frame.loc[non_hospital, "level"] = 3
    return frame


def summarize(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    by_level = frame.pivot_table(
        index="county", columns="level", values="births", aggfunc="sum", fill_value=0
    ).reindex(columns=[2, 3, 4, 5], fill_value=0)
    facility = by_level.sum(axis=1)
    l45 = by_level[4] + by_level[5]
    out = pd.DataFrame({
        "variant": variant,
        "births_l2": by_level[2],
        "births_l3": by_level[3],
        "births_l4": by_level[4],
        "births_l5": by_level[5],
        "l23_given_facility": (by_level[2] + by_level[3]) / facility,
        "l5_given_l45": by_level[5] / l45,
        "measure": frame.groupby("county")["measure"].first(),
    })
    return out.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook")
    parser.add_argument("output")
    args = parser.parse_args()

    frame = load_khis(args.workbook)
    result = pd.concat([
        summarize(frame, "base"),
        summarize(reclassify_non_hospital_l4(frame), "l4_name_reclass"),
    ], ignore_index=True)
    result.insert(1, "year", YEAR)
    result.to_csv(args.output, index=False)
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
