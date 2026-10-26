"""Every external number the calibration uses, for Kisii, Makueni, and Mombasa.

The county data blocks are the outputs of the scripts in Calibration/targets/
run on the raw files (README), written out. The literature values are at the end. Rates are
formed from these counts in calibration_common.county_targets.
"""

# DHS 2022 (dhs_routing_targets.R), survey-weighted: 4+ ANC coverage, and final
# delivery location among women without and with 4+ ANC.
DHS = {
    "kisii": {
        "anc4": 0.629786142907805,
        "home_noanc": 0.132360262672562,
        "home_anc": 0.0172562650881773,
        "l23_anc": 0.170145484899875,
        "l45_anc": 0.812598250011949,
    },
    "makueni": {
        "anc4": 0.756951619202676,
        "home_noanc": 0.107630049888196,
        "home_anc": 0.0908771709521505,
        "l23_anc": 0.360880539640316,
        "l45_anc": 0.548242289407534,
    },
    "mombasa": {
        "anc4": 0.651698876680068,
        "home_noanc": 0.0357532748284192,
        "home_anc": 0.0410224772834912,
        "l23_anc": 0.195444093737032,
        "l45_anc": 0.763533428979477,
    },
}

# KHIS 2022 deliveries at L2, L3, L4, and L5 (khis_level_targets.py); Makueni
# County Referral Hospital is counted as L5.
KHIS_2022_DELIVERIES = {
    "kisii": (1006, 4731, 18449, 8291),
    "makueni": (1550, 2517, 14323, 3462),
    "mombasa": (1634, 5641, 18084, 7644),
}

# MOMISH KHIS, January 2022-June 2025 (momish_khis_targets.py): deliveries and
# recorded events at L2/3 and internal L4. L5 is excluded because it receives
# external referrals. Recorded diagnoses are under-recorded by SCALE_COMP.
SCALE_COMP = 4.48
KHIS = {
    "kisii": {
        "deliveries": {"l23": 20324, "l4": 59266},
        "preterm": {"l23": 268, "l4": 1166},
        "cs": {"l23": 805, "l4": 11852},
        "complications_l23": 250,
        "complications_l4": {"aph": 125, "pph": 317, "eclampsia": 102, "ol": 1239, "rupture": 19, "sepsis": 55},
        "deaths": {"l23": 2, "l4": 18},
    },
    "makueni": {
        "deliveries": {"l23": 14899, "l4": 48545},
        "preterm": {"l23": 205, "l4": 1737},
        "cs": {"l23": 971, "l4": 13461},
        "complications_l23": 379,
        "complications_l4": {"aph": 222, "pph": 1038, "eclampsia": 200, "ol": 507, "rupture": 44, "sepsis": 144},
        "deaths": {"l23": 4, "l4": 35},
    },
    "mombasa": {
        "deliveries": {"l23": 25434, "l4": 61076},
        "preterm": {"l23": 363, "l4": 2465},
        "cs": {"l23": 2165, "l4": 19073},
        "complications_l23": 616,
        "complications_l4": {"aph": 373, "pph": 1077, "eclampsia": 313, "ol": 1100, "rupture": 24, "sepsis": 34},
        "deaths": {"l23": 1, "l4": 31},
    },
}

# MOMISH KHIS deliveries and maternal deaths (all causes) at L5, January
# 2022-June 2025, used only by the illustrative cross-county referral sensitivity
# (L5 is not a calibration target).
KHIS_L5 = {
    "kisii": {"deliveries": 27764, "deaths": 72},
    "makueni": {"deliveries": 11551, "deaths": 14},
    "mombasa": {"deliveries": 23980, "deaths": 121},
}

# Rescue referral records, Nov 2021-Oct 2022 (rescue_transfer_rate.py): 289 of 499
# L2/3 emergency transfers with a known destination went to L5; of 533 with a
# reason, 248 were labour emergencies, 110 maternal complications, and 67 preterm
# (15 normal labour and 93 other or unknown have no model pathway).
RESCUE = {"to_l5": 289, "with_destination": 499,
          "labour_emergency": 248, "maternal_complication": 110, "preterm": 67}

# ---- Literature values ------------------------------------------------------

# Preterm share of all facility births after transfer (Waiswa et al. 2020).
PRETERM_FAC = 0.134
# Planned share of caesarean sections in sub-Saharan Africa (Sandie et al. 2023).
PLANNED_CS_SHARE = 0.40
# Planned CS among preterm births: CS among preterm births (0.265, Ghosh et al. 2022)
# x elective share of preterm births after CS in 54 Nigerian referral hospitals,
# 1232 / (1232 + 3245): 1,232 preterm of 8,246 elective and 3,245 preterm of 13,478
# emergency caesarean births (Adeniran et al. 2024, BJOG 131:78-87).
PRETERM_PLANNED_CS = 0.265 * 1232 / (1232 + 3245)
# The model produces only direct, non-abortion maternal deaths, so KHIS MMR is
# scaled to that share (Say et al. 2014: 810,000 of 1,310,000 deaths), as in the
# Kakamega calibration.
DIRECT_CAUSE_SHARE = 810 / 1310
# Severe maternal outcomes per facility birth (workbook p_SMO_fac: Beyene et al. 2022,
# South Ethiopia, and KHIS); a validation check only.
SMO_REFERENCE = {"kisii": 0.0358, "makueni": 0.04, "mombasa": 0.0442}
