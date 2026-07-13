from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import parameter_loader
from global_func import get_P_l45, reset_E, reset_HSS, reset_S, reset_flags
from model_run import run_model_dash
from parameter_loader import calculate_derived_parameters, get_parameters, get_slider_params


WORKBOOK_PATH = Path(
    "/Users/meibinchen/Library/CloudStorage/OneDrive-JohnsHopkins/"
    "Kakamega SDR Project/MOMISH interventions/SDR Parameters.xlsx"
)
COUNTIES = ["kakamega", "mombasa", "makueni", "kisii"]
N_MONTHS = 36
N_RUNS = 200
BASE_SEED = 4200
ENABLED_SINGLE_INTERVENTION_SUPPLY = 0.95
PROMPTS_HIGH_SENS_TRAD = 0.95


def clip01(value):
    return float(np.clip(value, 0.0, 1.0))


CURRENT_IMPLEMENTATION_INDEX = {
    "prompts": {
        "kakamega": 0.0,
        "kisii": 0.151,
        "mombasa": 0.187,
        "makueni": 0.116,
    },
    "mentors": {
        "kakamega": 0.0,
        "kisii": 0.0713,
        "mombasa": 0.0583,
        "makueni": 0.1099,
    },
    "referral": {
        "kakamega": 0.0,
        "kisii": 0.0,
        "mombasa": 0.18,
        "makueni": 0.0,
    },
    "pulse": {
        "kakamega": 0.0,
        "kisii": 0.0455,
        "mombasa": 0.0126,
        "makueni": 0.003,
    },
    "fqa": {
        "kakamega": 0.0,
        "kisii": 0.0165,
        "mombasa": 0.0067,
        "makueni": 0.0296,
    },
}


IMPLEMENTATION_INDEX_BY_LEVEL = {
    "moderate": 0.50,
    "high": 0.95,
}


PROMPTS_RR_BY_LEVEL = {
    "current": 1.02,
    "moderate": 1.18,
    "high": 1.35,
}


BLOOD_ADOPTION_BY_LEVEL = {
    "current": 0.25,
    "moderate": 0.50,
    "high": 0.95,
}


SCENARIOS = [
    {"scenario": "mentors_base", "intervention": "MENTORS", "level": "base", "description": "No HSS; MENTORS off"},
    {"scenario": "mentors_current", "intervention": "MENTORS", "level": "current", "description": "County-specific current implementation index; all single interventions on"},
    {"scenario": "mentors_moderate", "intervention": "MENTORS", "level": "moderate", "description": "50% implementation index; all single interventions on"},
    {"scenario": "mentors_high", "intervention": "MENTORS", "level": "high", "description": "95% implementation index; all single interventions on"},
    {"scenario": "prompts_base", "intervention": "PROMPTS", "level": "base", "description": "All off"},
    {"scenario": "prompts_current", "intervention": "PROMPTS", "level": "current", "description": "County-specific current implementation index; 1.02 RR effectiveness"},
    {"scenario": "prompts_moderate", "intervention": "PROMPTS", "level": "moderate", "description": "50% implementation index; 1.18 RR effectiveness"},
    {"scenario": "prompts_high", "intervention": "PROMPTS", "level": "high", "description": "95% implementation index; 1.35 RR effectiveness"},
    {"scenario": "prompts_high_shift", "intervention": "PROMPTS", "level": "high_shift", "description": "PROMPTS high plus conservative delivery shift and matched facility expansion"},
    {"scenario": "blood_base", "intervention": "Blood", "level": "base", "description": "All off"},
    {"scenario": "blood_current", "intervention": "Blood", "level": "current", "description": "Blood tracking on; 25% adoption"},
    {"scenario": "blood_moderate", "intervention": "Blood", "level": "moderate", "description": "Blood tracking on; 50% adoption"},
    {"scenario": "blood_high", "intervention": "Blood", "level": "high", "description": "Blood tracking on; 95% adoption"},
    {"scenario": "referral_base", "intervention": "Referral", "level": "base", "description": "All off"},
    {"scenario": "referral_current", "intervention": "Referral", "level": "current", "description": "County-specific current implementation index; 1 hour delay shift"},
    {"scenario": "referral_moderate", "intervention": "Referral", "level": "moderate", "description": "50% implementation index; 1 hour delay shift"},
    {"scenario": "referral_high", "intervention": "Referral", "level": "high", "description": "95% implementation index; 1 hour delay shift"},
    {"scenario": "pulse_base", "intervention": "PULSE", "level": "base", "description": "All off"},
    {"scenario": "pulse_current", "intervention": "PULSE", "level": "current", "description": "County-specific current implementation index"},
    {"scenario": "pulse_moderate", "intervention": "PULSE", "level": "moderate", "description": "50% implementation index"},
    {"scenario": "pulse_high", "intervention": "PULSE", "level": "high", "description": "95% implementation index"},
    {"scenario": "fqa_base", "intervention": "FQA", "level": "base", "description": "PULSE current; FQA off; all single interventions on"},
    {"scenario": "fqa_current", "intervention": "FQA", "level": "current", "description": "County-specific current FQA index; PULSE current; all single interventions on; 10% amplification"},
    {"scenario": "fqa_moderate", "intervention": "FQA", "level": "moderate", "description": "50% FQA implementation index; PULSE current; all single interventions on; 10% amplification"},
    {"scenario": "fqa_high", "intervention": "FQA", "level": "high", "description": "95% FQA implementation index; PULSE current; all single interventions on; 30% amplification"},
]


def enable_single_interventions(param, flags):
    treatment_flags = {
        "flag_pph_bundle": "pph_bundle",
        "flag_iv_iron": "iv_iron",
        "flag_MgSO4": "MgSO4",
        "flag_antibiotics": "antibiotics",
        "flag_oxytocin": "oxytocin",
    }
    for flag, supply in treatment_flags.items():
        flags[flag] = 1
        param["S"][supply] = ENABLED_SINGLE_INTERVENTION_SUPPLY


def apply_prompts_sensitivity(param, implementation_index):
    baseline = float(param["sen_risk_trad"])
    scale = clip01(implementation_index / IMPLEMENTATION_INDEX_BY_LEVEL["high"])
    effective_sensitivity = baseline + (PROMPTS_HIGH_SENS_TRAD - baseline) * scale
    param["sen_risk_trad"] = clip01(effective_sensitivity)
    param["sen_risk_trad_target"] = param["sen_risk_trad"]


def current_implementation(intervention, county):
    return CURRENT_IMPLEMENTATION_INDEX[intervention].get(county, 0.0)


def implementation_for_level(intervention, level, county):
    if level == "current":
        return current_implementation(intervention, county)
    return IMPLEMENTATION_INDEX_BY_LEVEL[level]


def configure_conservative_shift(param, flags, slider_params):
    flags["flag_CHV"] = 1
    flags["flag_ANC"] = 1
    flags["flag_LB"] = 1
    flags["flag_capacity"] = 1

    param["HSS"]["P_ANC"] = 0.70
    p_l45_exp = get_P_l45(param["HSS"]["P_ANC"], slider_params)
    min_l45 = p_l45_exp if p_l45_exp is not None else 0.0
    param["HSS"]["P_L45"] = max(min_l45, 0.53)
    param["HSS"]["CHV_memory"] = "Logistic Decay"
    param["HSS"]["tau_decay"] = 6
    param["HSS"]["capacity_added"] = 0.25


def configure_scenario(name, param, flags, slider_params=None):
    hss = param["HSS"]
    county = param.get("county", "kakamega")

    if name.endswith("_base") and not name.startswith("fqa_"):
        if name == "fqa_base":
            flags["flag_pulse"] = 1
            pulse_index = current_implementation("pulse", county)
            hss["pulse_implementation_index"] = pulse_index
            hss["fqa_implementation_index"] = 0.0
            param["pulse_implementation_index"] = pulse_index
            param["fqa_implementation_index"] = 0.0
            enable_single_interventions(param, flags)
        return

    if name.startswith("mentors_"):
        level = name.removeprefix("mentors_")
        implementation_index = implementation_for_level("mentors", level, county)
        flags["flag_MENTOR"] = 1
        hss.update({
            "mentor_implementation_index": implementation_index,
        })
        param["mentors_implementation_index"] = implementation_index
        enable_single_interventions(param, flags)
        return

    if name.startswith("prompts_"):
        level = "high" if name == "prompts_high_shift" else name.removeprefix("prompts_")
        implementation_index = implementation_for_level("prompts", level, county)
        prompts_rr_anc4p = PROMPTS_RR_BY_LEVEL[level]
        flags["flag_PROMPTS"] = 1
        hss.update({
            "prompts_implementation_index": implementation_index,
            "prompts_rr_anc4p": prompts_rr_anc4p,
        })
        param["prompts_implementation_index"] = implementation_index
        param["prompts_rr_anc4p"] = prompts_rr_anc4p
        apply_prompts_sensitivity(param, implementation_index)
        if name == "prompts_high_shift":
            if slider_params is None:
                raise ValueError("slider_params is required for prompts_high_shift")
            configure_conservative_shift(param, flags, slider_params)
        return

    if name.startswith("blood_"):
        level = name.removeprefix("blood_")
        adoption = BLOOD_ADOPTION_BY_LEVEL[level]
        flags["flag_blood"] = 1
        flags["flag_blood_tracking"] = 1
        hss["blood_adoption"] = adoption
        return

    if name.startswith("referral_"):
        level = name.removeprefix("referral_")
        flags["flag_transfer"] = 1
        hss["P_transfer"] = implementation_for_level("referral", level, county)
        return

    if name.startswith("pulse_"):
        level = name.removeprefix("pulse_")
        implementation_index = implementation_for_level("pulse", level, county)
        flags["flag_pulse"] = 1
        hss["pulse_implementation_index"] = implementation_index
        hss["fqa_implementation_index"] = 0.0
        param["pulse_implementation_index"] = implementation_index
        param["fqa_implementation_index"] = 0.0
        return

    if name.startswith("fqa_"):
        level = name.removeprefix("fqa_")
        pulse_implementation_index = current_implementation("pulse", county)
        flags["flag_pulse"] = 1
        hss["pulse_implementation_index"] = pulse_implementation_index
        param["pulse_implementation_index"] = pulse_implementation_index
        enable_single_interventions(param, flags)
        if level != "base":
            flags["flag_fqa"] = 1
            fqa_implementation_index = implementation_for_level("fqa", level, county)
            fqa_pulse_modifier = 0.30 if level == "high" else 0.10
            hss["fqa_pulse_modifier_level"] = "High" if level == "high" else "Low"
            hss["fqa_pulse_modifier"] = fqa_pulse_modifier
            hss["fqa_implementation_index"] = fqa_implementation_index
            param["fqa_pulse_modifier"] = fqa_pulse_modifier
            param["fqa_implementation_index"] = fqa_implementation_index
        else:
            hss["fqa_implementation_index"] = 0.0
            param["fqa_implementation_index"] = 0.0
        return

    raise ValueError(f"Unknown scenario: {name}")


def safe_rate(df, column, mask=None):
    if column not in df:
        return np.nan
    values = df[column] if mask is None else df.loc[mask, column]
    return float(values.mean()) if len(values) else np.nan


def summarize(individuals):
    transfer = individuals["i_transfer"] == 1
    facility = individuals["i_loc_new_v2"] > 0
    anemia_with_anc = (individuals["i_anemia"] == 1) & (individuals["i_ANC"] == 1)
    pph_case = individuals["i_pph"] == 1
    eclampsia_case = individuals["i_eclampsia"] == 1
    sepsis_case = individuals["i_mat_sepsis"] == 1
    ol_case = individuals["i_OL"] == 1
    return {
        "anc": safe_rate(individuals, "i_ANC"),
        "initial_l45": float((individuals["i_loc"] >= 2).mean()),
        "final_l45": float((individuals["i_loc_new_v2"] >= 2).mean()),
        "free_referral": safe_rate(individuals, "i_free_referral"),
        "self_referral": safe_rate(individuals, "i_self_referral"),
        "emergency_transfer": safe_rate(individuals, "i_transfer"),
        "delay_lt1h_given_transfer": float((individuals.loc[transfer, "travel_time_transfer"] == 0).mean()),
        "delay_1_2h_given_transfer": float((individuals.loc[transfer, "travel_time_transfer"] == 1).mean()),
        "delay_2plus_given_transfer": float((individuals.loc[transfer, "travel_time_transfer"] == 2).mean()),
        "iv_iron": safe_rate(individuals, "i_iv_iron"),
        "pph_bundle": safe_rate(individuals, "i_pph_bundle"),
        "mgso4": safe_rate(individuals, "i_MgSO4"),
        "antibiotics": safe_rate(individuals, "i_antibiotics"),
        "oxytocin": safe_rate(individuals, "i_oxytocin"),
        "iv_iron_given_anemia_anc": safe_rate(individuals, "i_iv_iron", anemia_with_anc),
        "pph_bundle_given_pph": safe_rate(individuals, "i_pph_bundle", pph_case),
        "mgso4_given_eclampsia": safe_rate(individuals, "i_MgSO4", eclampsia_case),
        "antibiotics_given_sepsis": safe_rate(individuals, "i_antibiotics", sepsis_case),
        "oxytocin_given_ol": safe_rate(individuals, "i_oxytocin", ol_case),
        "pph": safe_rate(individuals, "i_pph_new"),
        "maternal_sepsis": safe_rate(individuals, "i_mat_sepsis_new"),
        "eclampsia": safe_rate(individuals, "i_eclampsia_new"),
        "obstructed_labor": safe_rate(individuals, "i_OL_final"),
        "aph": safe_rate(individuals, "i_aph"),
        "uterine_rupture": safe_rate(individuals, "i_ruptured_uterus"),
        "severe_complication": safe_rate(individuals, "i_severe_new"),
        "maternal_deaths_per_100k": safe_rate(individuals, "i_mat_death") * 100000,
        "facility_maternal_deaths_per_100k": safe_rate(individuals, "i_mat_death", facility) * 100000,
        "neonatal_deaths_per_1000": safe_rate(individuals, "i_neo_death") * 1000,
    }


LOCATION_LABELS = {
    0: "Home",
    1: "L2/3",
    2: "L4",
    3: "L5",
}


def summarize_by_location(individuals):
    """Summarize outcomes by final delivery location for one model run."""
    rows = []
    total_births = len(individuals)
    for location_code, location in LOCATION_LABELS.items():
        at_location = individuals["i_loc_new_v2"] == location_code
        location_df = individuals.loc[at_location]
        births = len(location_df)
        anemia_with_anc = (
            (location_df["i_anemia"] == 1) & (location_df["i_ANC"] == 1)
        )
        rows.append({
            "location": location,
            "location_code": location_code,
            "births": births,
            "delivery_share": births / total_births if total_births else np.nan,
            "anc": safe_rate(location_df, "i_ANC"),
            "pph": safe_rate(location_df, "i_pph_new"),
            "maternal_sepsis": safe_rate(location_df, "i_mat_sepsis_new"),
            "eclampsia": safe_rate(location_df, "i_eclampsia_new"),
            "obstructed_labor": safe_rate(location_df, "i_OL_final"),
            "aph": safe_rate(location_df, "i_aph"),
            "uterine_rupture": safe_rate(location_df, "i_ruptured_uterus"),
            "severe_complication": safe_rate(location_df, "i_severe_new"),
            "maternal_deaths": float(location_df["i_mat_death"].sum()),
            "maternal_deaths_per_100k": (
                safe_rate(location_df, "i_mat_death") * 100000
            ),
            "neonatal_deaths": float(location_df["i_neo_death"].sum()),
            "neonatal_deaths_per_1000": (
                safe_rate(location_df, "i_neo_death") * 1000
            ),
            "iv_iron_given_anemia_anc": safe_rate(
                location_df, "i_iv_iron", anemia_with_anc
            ),
            "pph_bundle_given_pph": safe_rate(
                location_df, "i_pph_bundle", location_df["i_pph"] == 1
            ),
            "mgso4_given_eclampsia": safe_rate(
                location_df, "i_MgSO4", location_df["i_eclampsia"] == 1
            ),
            "antibiotics_given_sepsis": safe_rate(
                location_df, "i_antibiotics", location_df["i_mat_sepsis"] == 1
            ),
            "oxytocin_given_ol": safe_rate(
                location_df, "i_oxytocin", location_df["i_OL"] == 1
            ),
        })
    return rows


def run_scenarios(county):
    parameter_loader.WORKBOOK_PATH = WORKBOOK_PATH
    slider_params = get_slider_params(county=county)
    rows = []
    location_rows = []

    for run in range(N_RUNS):
        parameter_seed = BASE_SEED + run
        month_seeds = np.arange(parameter_seed, parameter_seed + N_MONTHS, dtype=int)
        overall_baseline_summary = None
        overall_baseline_location_summary = None
        for scenario_def in SCENARIOS:
            scenario = scenario_def["scenario"]
            is_standard_base = (
                scenario_def["level"] == "base"
                and scenario_def["intervention"] != "FQA"
            )
            if is_standard_base and overall_baseline_summary is not None:
                scenario_summary = overall_baseline_summary.copy()
                location_summary = [
                    row.copy() for row in overall_baseline_location_summary
                ]
            else:
                param = get_parameters(county=county, seed=parameter_seed)
                param = calculate_derived_parameters(param)
                param.update({
                    "E": reset_E(),
                    "S": reset_S(slider_params),
                    "HSS": reset_HSS(slider_params),
                })
                flags = reset_flags()
                configure_scenario(scenario, param, flags, slider_params)
                _, individuals, _ = run_model_dash(
                    param,
                    flags,
                    n_months=N_MONTHS,
                    int_period=N_MONTHS,
                    base_seed=month_seeds,
                )
                scenario_summary = summarize(individuals)
                location_summary = summarize_by_location(individuals)
                if is_standard_base:
                    overall_baseline_summary = scenario_summary.copy()
                    overall_baseline_location_summary = [
                        row.copy() for row in location_summary
                    ]
            rows.append({
                "scenario": scenario,
                "intervention": scenario_def["intervention"],
                "level": scenario_def["level"],
                "run": run + 1,
                **scenario_summary,
            })
            for location_result in location_summary:
                location_rows.append({
                    "scenario": scenario,
                    "intervention": scenario_def["intervention"],
                    "level": scenario_def["level"],
                    "run": run + 1,
                    **location_result,
                })
        if (run + 1) % 10 == 0 or run == 0:
            print(f"{county.title()}: completed {run + 1}/{N_RUNS} matched runs", flush=True)
    run_results = pd.DataFrame(rows)
    metric_columns = [
        column for column in run_results.columns
        if column not in {"scenario", "intervention", "level", "run"}
    ]
    means = run_results.groupby(
        ["scenario", "intervention", "level"], sort=False
    )[metric_columns].mean().reset_index()

    differences = means.copy()
    overall_baseline = means.loc[
        means["scenario"] == "mentors_base", metric_columns
    ].iloc[0]
    differences[metric_columns] = (
        means[metric_columns].to_numpy() - overall_baseline.to_numpy()
    )
    differences.insert(3, "comparison_baseline", "overall_no_flags")

    location_run_results = pd.DataFrame(location_rows)
    location_keys = [
        "scenario", "intervention", "level", "location", "location_code"
    ]
    location_metric_columns = [
        column for column in location_run_results.columns
        if column not in {*location_keys, "run"}
    ]
    location_means = location_run_results.groupby(
        location_keys, sort=False
    )[location_metric_columns].mean().reset_index()
    baseline_by_location = location_means.loc[
        location_means["scenario"] == "mentors_base",
        ["location_code", *location_metric_columns],
    ].set_index("location_code")
    location_differences = location_means.copy()
    for index, row in location_differences.iterrows():
        baseline = baseline_by_location.loc[row["location_code"]]
        location_differences.loc[index, location_metric_columns] = (
            row[location_metric_columns].to_numpy(dtype=float)
            - baseline.to_numpy(dtype=float)
        )
    location_differences.insert(3, "comparison_baseline", "overall_no_flags")

    return (
        pd.DataFrame(SCENARIOS),
        run_results,
        means,
        differences,
        location_run_results,
        location_means,
        location_differences,
    )


if __name__ == "__main__":
    process_columns = [
        "anc", "initial_l45", "final_l45", "free_referral",
        "emergency_transfer", "delay_lt1h_given_transfer",
        "delay_1_2h_given_transfer", "delay_2plus_given_transfer",
        "iv_iron_given_anemia_anc", "pph_bundle_given_pph",
        "mgso4_given_eclampsia", "antibiotics_given_sepsis",
        "oxytocin_given_ol",
    ]
    outcome_columns = [
        "pph", "maternal_sepsis", "eclampsia", "obstructed_labor",
        "severe_complication", "maternal_deaths_per_100k",
        "neonatal_deaths_per_1000",
    ]
    for county in COUNTIES:
        (
            definitions,
            run_results,
            means,
            differences,
            location_run_results,
            location_means,
            location_differences,
        ) = run_scenarios(county)
        process = means[["scenario", "intervention", "level", *process_columns]].copy()
        process[process_columns] *= 100
        outcomes = means[["scenario", "intervention", "level", *outcome_columns]].copy()
        outcomes[[c for c in outcome_columns if "deaths_per" not in c]] *= 100
        outcome_differences = differences[
            ["scenario", "intervention", "level", "comparison_baseline", *outcome_columns]
        ].copy()
        outcome_differences[[c for c in outcome_columns if "deaths_per" not in c]] *= 100

        results_json = Path(f"scenario_comparison_results_{county}.json")
        results_dir = Path(f"scenario_comparison_results_{county}")
        payload = {
            "metadata": {
                "county": county,
                "n_months": N_MONTHS,
                "n_runs": N_RUNS,
                "base_seed": BASE_SEED,
            },
            "definitions": definitions.to_dict(orient="records"),
            "run_results": json.loads(run_results.to_json(orient="records")),
            "means": json.loads(means.to_json(orient="records")),
            "differences": json.loads(differences.to_json(orient="records")),
            "process": json.loads(process.to_json(orient="records")),
            "outcomes": json.loads(outcomes.to_json(orient="records")),
            "outcome_differences": json.loads(outcome_differences.to_json(orient="records")),
            "location_run_results": json.loads(
                location_run_results.to_json(orient="records")
            ),
            "location_means": json.loads(location_means.to_json(orient="records")),
            "location_differences": json.loads(
                location_differences.to_json(orient="records")
            ),
        }
        results_json.write_text(json.dumps(payload, indent=2))
        results_dir.mkdir(exist_ok=True)
        output_tables = {
            "scenario_definitions.csv": definitions,
            "run_level_results.csv": run_results,
            "mean_results.csv": means,
            "differences_from_base.csv": differences,
            "process_indicators.csv": process,
            "outcome_results.csv": outcomes,
            "outcome_differences.csv": outcome_differences,
            "location_run_level_results.csv": location_run_results,
            "location_mean_results.csv": location_means,
            "location_differences_from_base.csv": location_differences,
        }
        for filename, table in output_tables.items():
            table.to_csv(results_dir / filename, index=False)
        print(f"{county.title()} results written to {results_dir.resolve()}", flush=True)
