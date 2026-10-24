"""Kakamega L2/3-to-L4/5 transfer rate (tau) from Rescue referral records.

tau = P(transferred before delivery | labour began at L2/3)
    = annual transfers / (L2/3 births recorded in KHIS + annual transfers).

KHIS records births where delivery occurred, so transferred women are counted
at L4/5 and are added back to the L2/3 denominator. Rescue operated at only
part of the county, so the denominator is restricted to L2/3 facilities that
originated at least one Rescue transfer. Origin levels come from KHIS rather
than Rescue's own level field. Rescue scaled up from November 2021, so the
steady-state window is July--September 2022. Transfers through other channels
are absent, so every estimate is a lower bound for the covered facilities.

The accepted value is the ``all_reasons`` estimate rounded to 0.08.

A second file, ``rescue_transfer_mix.csv``, describes L2/3 transfers over the
full period (origins classified by KHIS level): the share received at L5, the
share by primary reason, and the severe-type share of maternal-complication
case types.

Usage:
    python Calibration/targets/rescue_transfer_rate.py \
        "/path/to/Facility Readiness - Rescue_referrals.xlsx" \
        /path/to/L23_ProportionLB_20260430.xlsx \
        Calibration/targets/rescue_transfer_rate.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


YEAR = 2022
WINDOW = ("2022-07-01", "2022-10-01")
# Case types that occur after delivery or outside labour; they do not change
# the delivery location. "Others" is ambiguous and is excluded only from the
# pre-delivery estimate.
NOT_PRE_DELIVERY = {
    "postpartum hemorrhage", "retained placenta", "birth asphyxia", "abortion",
    "ectopic pregnancy", "anemia", "sepsis", "others", "nan",
}
COLUMNS = [
    "row_id", "case_id", "received", "year", "quarter", "reason", "kind",
    "transfer_from", "from_level", "transfer_to", "to_level", "ward", "client",
    "source", "case_types",
]


def normalize(name: object) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower())


def load_facilities(path: str) -> pd.DataFrame:
    facilities = pd.read_excel(path, sheet_name="Facilities")
    facilities.columns = [str(column).strip() for column in facilities.columns]
    births = pd.read_excel(path, sheet_name="live_birth")[["organisationunitname", YEAR]]
    return facilities.merge(births, on="organisationunitname", how="left")


def facility_lookup(facilities: pd.DataFrame) -> dict[str, int]:
    lookup = {}
    for index, row in facilities.iterrows():
        for column in ("organisationunitname", "facility_alternate_name_1", "facility_alternate_name_2"):
            if pd.notna(row[column]):
                lookup[normalize(row[column])] = index
    return lookup


# Case-type groups for the L2/3 transfer mix. A transfer with several case types
# takes the first matching group in REASON_ORDER.
REASON_GROUPS = {
    "preterm": {"prematurity", "pre term labor", "pprom"},
    "maternal_complication": {
        "postpartum hemorrhage", "antepartum hemorrhage", "eclampsia", "sepsis",
        "placenta praevia", "retained placenta", "pre-eclampsia",
        "pih(pregnancy induced hypertension)", "anemia",
    },
    "labour_emergency": {
        "fetal distress", "prolonged labour", "prolonged labor", "obstructed labour",
        "cord prolapse", "malpresentation", "breech presentation", "malposition",
        "multiple pregnancies",
    },
    "normal_labour": {"normal labor"},
}
REASON_ORDER = ("preterm", "maternal_complication", "labour_emergency", "normal_labour")
# Maternal-complication case types treated as non-severe (hypertension without
# seizures, anaemia); hemorrhage, sepsis, and eclampsia are treated as severe.
NON_SEVERE_TYPES = {"pre-eclampsia", "pih(pregnancy induced hypertension)", "anemia"}


def case_types(value: object) -> set[str]:
    return {item.strip().lower() for item in str(value).split(",")}


def primary_reason(value: object) -> str:
    types = case_types(value)
    for reason in REASON_ORDER:
        if types & REASON_GROUPS[reason]:
            return reason
    return "other_unknown"


def pre_delivery(case_types: object) -> bool:
    types = {item.strip().lower() for item in str(case_types).split(",")}
    return not types <= NOT_PRE_DELIVERY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rescue_workbook")
    parser.add_argument("khis_workbook")
    parser.add_argument("output")
    args = parser.parse_args()

    facilities = load_facilities(args.khis_workbook)
    lookup = facility_lookup(facilities)
    l23 = facilities[facilities["Level Facility"].isin([2, 3])]

    rescue = pd.read_excel(args.rescue_workbook)
    rescue.columns = COLUMNS
    rescue["facility_index"] = rescue["transfer_from"].map(lambda name: lookup.get(normalize(name)))
    transfers = rescue[rescue["kind"].eq("Transfer") & rescue["facility_index"].isin(l23.index)]

    covered = transfers["facility_index"].astype(int).unique()
    covered_births = l23.loc[covered, YEAR].sum()

    start, end = WINDOW
    in_window = transfers[(transfers["received"] >= start) & (transfers["received"] < end)]
    months = 3
    estimates = {
        "all_reasons": in_window,
        "pre_delivery": in_window[in_window["case_types"].map(pre_delivery)],
    }
    rows = []
    for name, records in estimates.items():
        annual = len(records) * 12 / months
        rows.append({
            "estimate": name,
            "window": f"{start} to {end}",
            "transfers_in_window": len(records),
            "annual_transfers": annual,
            "covered_l23_facilities": len(covered),
            "l23_facilities_with_births": int((l23[YEAR] > 0).sum()),
            "covered_l23_births": covered_births,
            "covered_share_of_l23_births": covered_births / l23[YEAR].sum(),
            "tau": annual / (covered_births + annual),
        })
    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False)
    print(result.round(4).to_string(index=False))

    # Receiving level of L2/3 transfers over the full period (L6 counted with L5).
    destination = transfers["to_level"]
    destination = destination[destination.isin(["Level 4", "Level 5", "Level 6"])]
    period = f"{rescue['received'].min():%Y-%m-%d} to {rescue['received'].max():%Y-%m-%d}"
    mix_rows = [{
        "quantity": "l5_share_of_l23_transfers",
        "value": destination.isin(["Level 5", "Level 6"]).mean(),
        "n": len(destination), "period": period,
    }]
    # Reason mix of L2/3 transfers over the full period.
    reasons = transfers["case_types"].map(primary_reason)
    for reason in (*REASON_ORDER, "other_unknown"):
        mix_rows.append({
            "quantity": f"share_{reason}", "value": reasons.eq(reason).mean(),
            "n": len(reasons), "period": period,
        })
    maternal_types = transfers["case_types"].map(case_types).explode()
    maternal_types = maternal_types[maternal_types.isin(REASON_GROUPS["maternal_complication"])]
    mix_rows.append({
        "quantity": "severe_type_share_of_maternal_mentions",
        "value": 1 - maternal_types.isin(NON_SEVERE_TYPES).mean(),
        "n": len(maternal_types), "period": period,
    })
    mix = pd.DataFrame(mix_rows)
    mix_path = Path(args.output).with_name("rescue_transfer_mix.csv")
    mix.to_csv(mix_path, index=False)
    print(mix.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
