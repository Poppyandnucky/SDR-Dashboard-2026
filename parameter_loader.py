"""
Parameter loader for SDR county-specific simulations.

Usage
-----
from parameter_loader import get_parameters, get_slider_params, calculate_derived_parameters

param = get_parameters("/path/to/SDR Parameters.xlsx", county="kisii", seed=123)
slider_params = get_slider_params("/path/to/SDR Parameters.xlsx", county="kisii")
param = calculate_derived_parameters(param)

Notes
-----
- County-specific sheets override shared/default sheets.
- Missing county values are ignored by default so you can fall back to defaults or Kakamega.
- Uncertain parameters are sampled through global_func.sample_from_ci.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from global_func import (
    sample_from_ci,
    odds_prob,
    comps_riskstatus_vs_lowrisk,
    comp2_comp1_anemia,
    P_RDS,
)

FACILITY_ORDER = ["home", "L2/3", "L4", "L5"]
FACILITY_SUFFIX = {"L2/3": "L2/3", "L4": "L4", "L5": "L5"}

# Excel names that are stored as multiple rows but expected as numpy arrays in the model.
SAMPLED_ARRAYS = {
    "E_Preterm_LMP": ["E_Preterm_LMP_preterm", "E_Preterm_LMP_atterm"],
    "E_Postterm_LMP": ["E_Postterm_LMP_preterm", "E_Postterm_LMP_atterm"],
    "p_PL_GA": ["p_PL_GA_37", "p_PL_GA_38", "p_PL_GA_39", "p_PL_GA_40", "p_PL_GA_41", "p_PL_GA_42"],
    "p_OL": ["p_OL_notprolonged", "p_OL_prolonged"],
}

# Disability-weight labels expected by existing DALY code.
DW_NAME_MAP = {
    "low_pph": "low pph",
    "high_pph": "high pph",
    "maternal_sepsis": "maternal sepsis",
    "obstructed_labor": "obstructed labor",
    "maternal_death": "maternal death",
    "neonatal_death": "neonatal death",
    "preterm_comp": "preterm comp",
    "neonatal_sepsis": "neonatal sepsis",
}

@dataclass(frozen=True)
class ParameterWorkbook:
    path: Path
    sheets: dict[str, pd.DataFrame]

    @classmethod
    def load(cls, path: str | Path) -> "ParameterWorkbook":
        path = Path(path)
        xls = pd.ExcelFile(path)
        sheets = {name: pd.read_excel(path, sheet_name=name) for name in xls.sheet_names}
        return cls(path=path, sheets=sheets)

    def sheet(self, name: str) -> pd.DataFrame:
        if name not in self.sheets:
            raise KeyError(f"Workbook is missing required sheet: {name}")
        return self.sheets[name].copy()


@lru_cache(maxsize=8)
def _load_workbook_cached(path: str, modified_ns: int) -> ParameterWorkbook:
    return ParameterWorkbook.load(path)


def load_parameter_workbook(path: str | Path) -> ParameterWorkbook:
    """Load a workbook once per path and modification timestamp."""
    resolved = Path(path).expanduser().resolve()
    return _load_workbook_cached(str(resolved), resolved.stat().st_mtime_ns)


def _clean_county(county: str) -> str:
    return str(county).strip().lower()


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value)


def _to_float_or_none(value: Any) -> float | None:
    """Convert workbook values to float, treating placeholders as missing."""
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sample_or_value(row: Mapping[str, Any], rng: np.random.Generator) -> Any:
    value = row.get("value")
    kind = row.get("kind")
    ci_lower = row.get("ci_lower")
    ci_upper = row.get("ci_upper")
    n = row.get("n")

    # Treat blank/fixed kinds or rows without CIs as deterministic.
    if _is_missing(kind) or str(kind).lower() == "fixed" or _is_missing(ci_lower) or _is_missing(ci_upper):
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                parsed = ast.literal_eval(stripped)
                array = np.asarray(parsed, dtype=float)
                if array.ndim != 1 or array.size == 0:
                    raise ValueError(f"Expected a non-empty numeric parameter array, got {value!r}")
                return array
        elif isinstance(value, (list, tuple, np.ndarray)):
            array = np.asarray(value, dtype=float)
            if array.ndim != 1 or array.size == 0:
                raise ValueError(f"Expected a non-empty numeric parameter array, got {value!r}")
            return array

        converted = _to_float_or_none(value)
        if converted is None:
            raise ValueError(f"Expected numeric parameter value, got {value!r}")
        return converted

    n_arg = None if _is_missing(n) else int(n)
    return float(sample_from_ci(float(value), float(ci_lower), float(ci_upper), n=n_arg, kind=str(kind), size=1, rng=rng)[0])


def _scalar_table(df: pd.DataFrame, key_col: str, value_col: str = "value") -> dict[str, float]:
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        key = row.get(key_col)
        val = row.get(value_col)
        if not _is_missing(key) and not _is_missing(val):
            out[str(key)] = float(val)
    return out


def _sampled_params(wb: ParameterWorkbook, rng: np.random.Generator) -> dict[str, Any]:
    rows = wb.sheet("sampled_params")
    sampled = {row["parameter_name"]: _sample_or_value(row, rng) for _, row in rows.iterrows()}

    params: dict[str, Any] = {}
    used = set()
    for new_name, old_names in SAMPLED_ARRAYS.items():
        params[new_name] = np.array([sampled[name] for name in old_names], dtype=float)
        used.update(old_names)

    for name, value in sampled.items():
        if name not in used:
            params[name] = value
    return params


def _intervention_params(wb: ParameterWorkbook, rng: np.random.Generator) -> dict[str, Any]:
    rows = wb.sheet("interv_params").dropna(subset=["parameter_name"])
    return {row["parameter_name"]: _sample_or_value(row, rng) for _, row in rows.iterrows() if not _is_missing(row.get("value"))}


def _constants(wb: ParameterWorkbook) -> dict[str, Any]:
    rows = wb.sheet("constants")
    return _scalar_table(rows, "parameter_name")


def _array_constants(wb: ParameterWorkbook) -> dict[str, np.ndarray]:
    rows = wb.sheet("array_constants")
    out: dict[str, np.ndarray] = {}
    for name, g in rows.dropna(subset=["parameter_name", "value"]).groupby("parameter_name", sort=False):
        g = g.sort_values("index") if "index" in g.columns else g
        arr = g["value"].to_numpy(dtype=float, copy=True)
        # Preserve GA_sequence as integers, matching existing code.
        if name == "GA_sequence":
            arr = arr.astype(int)
        out[str(name)] = arr
    return out


def _county_rows(wb: ParameterWorkbook, sheet: str, county: str) -> pd.DataFrame:
    df = wb.sheet(sheet)
    if "county" not in df.columns:
        return df
    county_clean = _clean_county(county)
    return df[df["county"].astype(str).str.strip().str.lower().eq(county_clean)].copy()


def _county_demographics(wb: ParameterWorkbook, county: str, rng: np.random.Generator) -> dict[str, Any]:
    rows = _county_rows(wb, "county_demographics", county)
    return {
        row["parameter_name"]: _sample_or_value(row, rng)
        for _, row in rows.iterrows()
        if not _is_missing(row.get("value"))
    }


def _county_calibrated(wb: ParameterWorkbook, county: str) -> dict[str, Any]:
    rows = _county_rows(wb, "county_calibrated", county)
    out: dict[str, Any] = {}
    for name, g in rows.dropna(subset=["parameter_name"]).groupby("parameter_name", sort=False):
        if "index" in g.columns and g["index"].notna().any():
            indexed = g[g["index"].notna()].copy()
            values = np.full(int(indexed["index"].max()) + 1, np.nan, dtype=float)
            for _, row in indexed.iterrows():
                if not _is_missing(row.get("value")):
                    values[int(row["index"])] = float(row["value"])
            if np.isfinite(values).any():
                out[str(name)] = values
        else:
            non_missing = g[g["value"].notna()]
            if not non_missing.empty:
                out[str(name)] = float(non_missing.iloc[0]["value"])
    return out


def _merge_county_params(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Apply county values while retaining fallback entries for partial arrays."""
    for name, value in overrides.items():
        if isinstance(value, np.ndarray) and np.isnan(value).any() and name in base:
            fallback = np.asarray(base[name], dtype=float)
            if fallback.shape == value.shape:
                value = np.where(np.isnan(value), fallback, value)
        base[name] = value


def _county_supply(wb: ParameterWorkbook, county: str, rng: np.random.Generator) -> dict[str, Any]:
    rows = _county_rows(wb, "county_supply", county)
    out: dict[str, Any] = {}

    model_facility_order = ["home", "L2/3", "L4", "L5"]

    for name, g in rows.dropna(subset=["parameter_name"]).groupby("parameter_name", sort=False):
        name = str(name)

        if "facility_level" in g.columns and g["facility_level"].notna().any():
            lookup = {
                str(row["facility_level"]).strip(): _sample_or_value(row, rng)
                for _, row in g.iterrows()
                if _to_float_or_none(row.get("value")) is not None
            }

            values = [lookup.get(level) for level in model_facility_order]
            if all(value is not None for value in values):
                out[name] = np.array(values, dtype=float)
            elif len(lookup) == 1:
                out[name] = float(next(iter(lookup.values())))

        else:
            g = g.sort_values("index") if "index" in g.columns else g
            values = [
                _sample_or_value(row, rng)
                for _, row in g.iterrows()
                if _to_float_or_none(row.get("value")) is not None
            ]
            if not values:
                continue
            out[name] = np.array(values, dtype=float) if len(values) > 1 else float(values[0])

    return out


def _county_facilities(wb: ParameterWorkbook, county: str) -> dict[str, Any]:
    rows = _county_rows(wb, "county_facilities", county).dropna(subset=["parameter_name", "value", "facility_level"])
    out: dict[str, Any] = {}

    def vals_for(parameter_name: str, levels=("L2/3", "L4", "L5")) -> list[float] | None:
        sub = rows[rows["parameter_name"].eq(parameter_name)]
        if sub.empty:
            return None
        lookup = {str(r["facility_level"]): float(r["value"]) for _, r in sub.iterrows()}
        return [lookup[level] for level in levels if level in lookup]

    count = vals_for("count")
    if count and len(count) == 3:
        out["num_L2/3"], out["num_L4"], out["num_L5"] = [int(x) for x in count]

    staff_map = {
        "num_surgical": "base_surgical",
        "num_nurses": "base_nurse",
        "num_anesthetists": "base_anesthetist",
    }
    for src, dst in staff_map.items():
        vals = vals_for(src)
        if vals and len(vals) == 3:
            out[dst] = [int(x) for x in vals]

    equipment_map = {
        "num_dopplers": "num_dopplers",
        "num_CTGs": "num_CTGs",
    }
    for src, prefix in equipment_map.items():
        vals = vals_for(src)
        if vals and len(vals) == 3:
            out[f"{prefix}_L2/3"], out[f"{prefix}_L4"], out[f"{prefix}_L5"] = [int(x) for x in vals]

    # Optional raw facility variables if you want them later.
    for name, g in rows.groupby("parameter_name", sort=False):
        if str(name) in {"count", "num_surgical", "num_nurses", "num_anesthetists", "num_dopplers", "num_CTGs"}:
            continue
        vals = vals_for(str(name))
        if vals:
            out[str(name)] = vals
    return out


def _county_specific_params(wb: ParameterWorkbook, county: str, rng: np.random.Generator) -> dict[str, Any]:
    params: dict[str, Any] = {}
    params.update(_county_demographics(wb, county, rng))
    params.update(_county_facilities(wb, county))
    params.update(_county_supply(wb, county, rng))
    params.update(_county_calibrated(wb, county))
    params.update(_calibration_targets(wb, county))
    return params


def _calibration_targets(wb: ParameterWorkbook, county: str) -> dict[str, float]:
    rows = _county_rows(wb, "calibration_targets", county)
    out = {}
    for _, row in rows.iterrows():
        name = row.get("target_name")
        value = row.get("value")
        if not _is_missing(name) and not _is_missing(value):
            out[str(name)] = float(value)
    return out


def _cost_dict(wb: ParameterWorkbook) -> dict[str, float]:
    rows = wb.sheet("cost_params")
    return {str(row["cost_item"]): float(row["value"]) for _, row in rows.iterrows() if not _is_missing(row.get("value"))}


def _disability_weights(wb: ParameterWorkbook, rng: np.random.Generator) -> dict[str, float]:
    rows = wb.sheet("disability_weights")
    out = {}
    for _, row in rows.iterrows():
        name = str(row["condition"])
        key = DW_NAME_MAP.get(name, name)
        pseudo_row = {
            "value": row.get("mean"),
            "ci_lower": row.get("ci_lower"),
            "ci_upper": row.get("ci_upper"),
            "n": None,
            "kind": "mean",
        }
        out[key] = _sample_or_value(pseudo_row, rng)
    return out


def get_parameters(
    workbook_path: str | Path,
    county: str = "kakamega",
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    strict_county: bool = False,
    fallback_county: str | None = "kakamega",
) -> dict[str, Any]:
    """Build the parameter dictionary for one county.

    Parameters
    ----------
    workbook_path:
        Path to `SDR Parameters.xlsx`.
    county:
        County name in the workbook, e.g. "kakamega", "kisii", "makueni", "mombasa".
    rng, seed:
        Use either a numpy Generator or a seed for reproducible uncertainty draws.
    strict_county:
        If True, raise an error when the county has no row in the `counties` sheet.
    fallback_county:
        County used for missing county-specific values. Set to None to disable fallback.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    wb = load_parameter_workbook(workbook_path)
    county_clean = _clean_county(county)

    counties = wb.sheet("counties")
    available = set(counties["county"].astype(str).str.strip().str.lower())
    if strict_county and county_clean not in available:
        raise ValueError(f"Unknown county {county!r}. Available counties: {sorted(available)}")

    param: dict[str, Any] = {}
    # Shared/default inputs.
    param.update(_constants(wb))
    param.update(_array_constants(wb))
    param.update(_sampled_params(wb, rng))
    param.update(_intervention_params(wb, rng))
    param["cost_dict"] = _cost_dict(wb)
    param["DW"] = _disability_weights(wb, rng)

    # County-specific overrides. Load fallback first so non-missing requested-county
    # values replace it while blank workbook cells inherit the fallback county.
    fallback_clean = _clean_county(fallback_county) if fallback_county else None
    if fallback_clean and county_clean != fallback_clean:
        param.update(_county_specific_params(wb, fallback_clean, rng))
    _merge_county_params(param, _county_specific_params(wb, county, rng))

    # Derived convenience values used in your current code.
    if "base_LB" in param:
        base_lb = np.asarray(param["base_LB"], dtype=float)
        if base_lb.size == 4 and base_lb.sum() > 0:
            param["base_p_45"] = float((base_lb[2] + base_lb[3]) / base_lb.sum())
            param["p_l5_l45"] = float(base_lb[3] / (base_lb[2] + base_lb[3])) if (base_lb[2] + base_lb[3]) > 0 else np.nan
            param["Num_Exp_L45"] = float(base_lb[2] + base_lb[3])

    param["county"] = county_clean
    return param


def get_slider_params(workbook_path: str | Path, county: str = "kakamega") -> dict[str, Any]:
    """Load dashboard slider defaults for one county."""
    wb = load_parameter_workbook(workbook_path)
    county_clean = _clean_county(county)

    out: dict[str, Any] = {}

    # Lookup-table slider arrays, e.g. p_l45_anc_slider.
    lookup = wb.sheet("lookup_tables")
    for table_name, g in lookup.dropna(subset=["table_name", "key", "value"]).groupby("table_name", sort=False):
        out[str(table_name)] = g[["key", "value"]].to_numpy(dtype=float, copy=True)

    sliders = wb.sheet("slider_parameters")
    if "county" in sliders.columns:
        sliders = sliders[sliders["county"].astype(str).str.strip().str.lower().eq(county_clean)]

    for _, row in sliders.iterrows():
        name = row.get("parameter_name")
        value = row.get("default_value")
        typ = str(row.get("type", "scalar")).lower()
        if _is_missing(name) or _is_missing(value):
            continue
        if typ == "array":
            # Allows comma-separated values in Excel, if you choose to store arrays that way.
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    parsed = ast.literal_eval(stripped)
                    out[str(name)] = np.array(parsed, dtype=float)
                else:
                    out[str(name)] = np.array([float(x.strip()) for x in value.split(",") if x.strip()], dtype=float)
            else:
                out[str(name)] = np.array([float(value)], dtype=float)
        else:
            out[str(name)] = float(value)

    # Sensible fallbacks from the main county parameter table.
    params = get_parameters(workbook_path, county=county)
    derived_params = calculate_derived_parameters(params)
    out.setdefault("base_knowledge_L45_slider", params.get("base_knowledge_L45"))
    out.setdefault("base_p_45_slider", params.get("base_p_45"))
    out.setdefault("p_ANC_base_slider", params.get("p_ANC_base"))
    out.setdefault("S_pph_bundle_slider", params.get("S_pph_bundle", np.zeros(4)))
    out.setdefault("S_iv_iron_slider", params.get("S_iv_iron"))
    out.setdefault("S_MgSO4_slider", params.get("S_MgSO4"))
    out.setdefault("S_antibiotics_slider", params.get("S_antibiotics"))
    out.setdefault("S_oxytocin_slider", derived_params.get("S_oxytocin"))
    out.setdefault("t_l23_l45_notsevere_slider", params.get("t_l23_l45_notsevere"))

    # reset_S indexes these defaults by facility level. Some county workbook
    # rows use a scalar when the same default applies at every level.
    for name in (
        "S_pph_bundle_slider",
        "S_MgSO4_slider",
        "S_antibiotics_slider",
        "S_oxytocin_slider",
    ):
        value = np.asarray(out[name], dtype=float)
        if value.ndim == 0 or value.size == 1:
            out[name] = np.full(4, float(value.reshape(-1)[0]), dtype=float)
        elif value.ndim == 1 and value.size == 4:
            out[name] = value.copy()
        else:
            raise ValueError(f"{name} must be a scalar or a four-element array, got shape {value.shape}")
    return out


def calculate_derived_parameters(param: dict[str, Any]) -> dict[str, Any]:
    """Keep your existing derived-parameter logic in one reusable place."""
    param = dict(param)  # avoid mutating the input unexpectedly

    param["p_anemia_anc"] = odds_prob(param["or_anc_anemia"], param["p_comp_anemia"], (1 - param["p_ANC_base"]))
    param["severe_highrisk"] = param["p_comp_severe_lowrisk"] * param["RR_comp_severe_highrisk_vs_lowrisk"]
    param["severe"] = np.array([param["p_comp_severe_lowrisk"], param["severe_highrisk"]])
    param["OL"] = np.asarray(param["p_OL"], dtype=float) * param["p_OL_scale"]

    param["OL_highrisk"], param["OL_lowrisk"] = comps_riskstatus_vs_lowrisk(
        param["OL"][0], param["p_highrisk"], param["RR_comp_highrisk_vs_lowrisk"]
    )
    param["ruptured_uterus_highrisk"], param["ruptured_uterus_lowrisk"] = comps_riskstatus_vs_lowrisk(
        param["p_ruptured_uterus"], param["p_highrisk"], param["RR_comp_highrisk_vs_lowrisk"]
    )
    param["aph_highrisk"], param["aph_lowrisk"] = comps_riskstatus_vs_lowrisk(
        param["p_aph"], param["p_highrisk"], param["RR_comp_highrisk_vs_lowrisk"]
    )
    param["eclampsia_highrisk"], param["eclampsia_lowrisk"] = comps_riskstatus_vs_lowrisk(
        param["p_eclampsia"], param["p_highrisk"], param["RR_comp_highrisk_vs_lowrisk"]
    )

    param["eclampsia_highrisk_anemia"] = comp2_comp1_anemia(param["eclampsia_highrisk"], param["or_anemia_eclampsia"])
    param["eclampsia_lowrisk_anemia"] = comp2_comp1_anemia(param["eclampsia_lowrisk"], param["or_anemia_eclampsia"])
    param["pph_OL_anemia"] = comp2_comp1_anemia(param["p_pph_OL"], param["or_anemia_pph"])
    param["mat_sepsis_OL_anemia"] = comp2_comp1_anemia(param["p_mat_sepsis_OL"], param["or_anemia_sepsis"])
    param["pph_elective_CS_anemia"] = comp2_comp1_anemia(param["p_pph_elective_CS"], param["or_anemia_pph"])
    param["mat_sepsis_elective_CS_anemia"] = comp2_comp1_anemia(param["p_mat_sepsis_elective_CS"], param["or_anemia_sepsis"])
    param["pph_emergency_CS_anemia"] = comp2_comp1_anemia(param["p_pph_emergency_CS"], param["or_anemia_pph"])
    param["mat_sepsis_emergency_CS_anemia"] = comp2_comp1_anemia(param["p_mat_sepsis_emergency_CS"], param["or_anemia_sepsis"])
    param["pph_other_anemia"] = comp2_comp1_anemia(param["p_pph_other"], param["or_anemia_pph"])
    param["mat_sepsis_other_anemia"] = comp2_comp1_anemia(param["p_mat_sepsis_other"], param["or_anemia_sepsis"])

    param["RDS_T"] = P_RDS(param)

    if "S_oxytocin_l45" in param:
        param["S_oxytocin"] = np.array([0, 0 * 0.3157 + 0.33 * (1 - 0.3157), param["S_oxytocin_l45"], param["S_oxytocin_l45"]])
    if "S_preterm_treat_l45" in param:
        param["S_preterm_treat"] = np.array([0, 0, param["S_preterm_treat_l45"], param["S_preterm_treat_l45"]])
    return param


def reset_inputs(param: dict[str, Any], n_months: int) -> dict[str, Any]:
    """County-aware version of your reset function."""
    LB = np.asarray(param["base_LB"], dtype=float)
    ANC = float(param["p_ANC_base"])
    CLASS = float(param["class"])
    highrisk = float(param["p_highrisk"])
    n_chv = int(param.get("n_CHV", 0))

    track = {
        "LB": LB,
        "ANC": ANC,
        "CLASS": CLASS,
        "LB_Track": np.zeros((n_months, 4)),
        "ANC_Track": np.zeros((n_months, 4)),
        "HighRisk_Track": np.zeros((n_months, 4)),
        "Facility_Capacity_Track": np.zeros((n_months, 1)),
        "Referral_Capacity_Track": np.zeros((n_months, 1)),
        "Num_Exp_L45_Track": np.zeros((n_months, 1)),
        "Constraint_Ratio_Track": np.zeros((n_months, 1)),
        "CS_Capacity_Track": np.zeros((n_months, 1)),
        "CHV_negative_Track": np.zeros((n_months, n_chv), dtype=int),
        "CHV_memory_Track": np.zeros((n_months, n_chv), dtype=int),
    }

    track["LB_Track"][0, :] = track["LB"]
    track["ANC_Track"][0, :] = np.repeat(ANC, 4)
    track["HighRisk_Track"][0, :] = np.repeat(highrisk, 4)
    track["Facility_Capacity_Track"][0, 0] = float(param["Capacity"])
    track["Referral_Capacity_Track"][0, 0] = 0
    track["Num_Exp_L45_Track"][0, 0] = float(LB[2] + LB[3])
    track["Constraint_Ratio_Track"][0, 0] = 1
    track["CS_Capacity_Track"][0, 0] = float(np.asarray(param["p_cs_capacity"])[3])
    return track
