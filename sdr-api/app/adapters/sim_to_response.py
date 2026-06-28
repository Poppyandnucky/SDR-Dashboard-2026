"""Translate sim DataFrame outputs into typed API response."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from app.adapters.scenario_to_sim import build_applied_interventions
from app.schemas.results import (
    AppliedIntervention,
    BaselineInterventionSeries,
    CostBreakdownItem,
    CostPerDalySeries,
    DeathByCause,
    DeliveryLocationSeries,
    DeliveryLocationTimeseries,
    FacilityCapacitySeries,
    IndicatorAvailable,
    IndicatorTimeseriesBundle,
    MaternalMortalitySeries,
    MetaResult,
    NarrativeResult,
    ResourceAdequacy,
    ScenarioResult,
    SummaryResult,
    TimeseriesResult,
)

WHO_KENYA_THRESHOLD_USD = 1042.0
CAUSE_COLUMNS = [
    ("Postpartum haemorrhage", "pph"),
    ("Eclampsia", "eclampsia"),
    ("Obstructed labour", "OL"),
    ("Sepsis", "mat_sepsis"),
    ("Other severe complications", "severe_comps"),
]


def _vec_sum(series_values) -> float:
    if series_values is None:
        return 0.0
    arr = np.concatenate(series_values) if hasattr(series_values.iloc[0], "__len__") else series_values.values
    return float(np.nansum(arr))


def _monthly_lb(df: pd.DataFrame) -> np.ndarray:
    lbs = []
    for val in df["Live Births Final"].values:
        lbs.append(float(np.sum(val)))
    return np.array(lbs)


def _cumulative_mmr(df: pd.DataFrame, lbs: np.ndarray) -> np.ndarray:
    deaths = df["Deaths"].apply(lambda x: float(np.sum(x))).values
    cum_deaths = np.cumsum(deaths)
    cum_lbs = np.cumsum(lbs)
    with np.errstate(divide="ignore", invalid="ignore"):
        mmr = np.where(cum_lbs > 0, cum_deaths / cum_lbs * 100_000, 0.0)
    return mmr


def _cell_total(val) -> float:
    if val is None:
        return 0.0
    arr = np.asarray(val, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.nansum(arr))


def _monthly_rate_per_lb(
    df: pd.DataFrame, col: str, lbs: np.ndarray, n_months: int, multiplier: float = 100.0
) -> list[float]:
    rates: list[float] = []
    for i in range(n_months):
        num = _cell_total(df.loc[i, col]) if col in df.columns else 0.0
        den = float(lbs[i])
        rate = num / den * multiplier if den > 0 else 0.0
        rates.append(round(rate, 4))
    return rates


def _monthly_mean_array_col(df: pd.DataFrame, col: str, n_months: int) -> list[float]:
    rates: list[float] = []
    for i in range(n_months):
        if col not in df.columns:
            rates.append(0.0)
            continue
        arr = np.asarray(df.loc[i, col], dtype=float)
        rates.append(round(float(np.nanmean(arr)), 4))
    return rates


def _pair_series(
    b_df: pd.DataFrame,
    i_df: pd.DataFrame,
    col: str,
    lbs_b: np.ndarray,
    lbs_i: np.ndarray,
    n_months: int,
    *,
    as_rate_per_lb: bool = True,
    multiplier: float = 100.0,
) -> BaselineInterventionSeries:
    if as_rate_per_lb:
        return BaselineInterventionSeries(
            baseline=_monthly_rate_per_lb(b_df, col, lbs_b, n_months, multiplier),
            intervention=_monthly_rate_per_lb(i_df, col, lbs_i, n_months, multiplier),
        )
    return BaselineInterventionSeries(
        baseline=_monthly_mean_array_col(b_df, col, n_months),
        intervention=_monthly_mean_array_col(i_df, col, n_months),
    )


def _build_indicator_series(
    b_df: pd.DataFrame,
    i_df: pd.DataFrame,
    lbs_b: np.ndarray,
    lbs_i: np.ndarray,
    n_months: int,
) -> IndicatorTimeseriesBundle:
    return IndicatorTimeseriesBundle(
        anc_rate_per_100_lb=_pair_series(b_df, i_df, "ANC", lbs_b, lbs_i, n_months),
        cs_rate_per_100_lb=_pair_series(b_df, i_df, "CS", lbs_b, lbs_i, n_months),
        normal_referral_per_100_lb=_pair_series(
            b_df, i_df, "Normal_referrals", lbs_b, lbs_i, n_months
        ),
        emergency_transfer_per_100_lb=_pair_series(
            b_df, i_df, "Emergency transfers", lbs_b, lbs_i, n_months
        ),
        high_risk_per_100_lb=_pair_series(b_df, i_df, "High risk", lbs_b, lbs_i, n_months),
        maternal_complication_rate_per_100_lb=_pair_series(
            b_df, i_df, "Comps after transfer", lbs_b, lbs_i, n_months
        ),
        severe_maternal_outcomes_per_100_lb=_pair_series(
            b_df, i_df, "severe_comps", lbs_b, lbs_i, n_months
        ),
        doppler_equipment_ratio=_pair_series(
            b_df, i_df, "Doppler_Ratio", lbs_b, lbs_i, n_months, as_rate_per_lb=False
        ),
        ctg_equipment_ratio=_pair_series(
            b_df, i_df, "CTG_Ratio", lbs_b, lbs_i, n_months, as_rate_per_lb=False
        ),
        nurse_staff_ratio=_pair_series(
            b_df, i_df, "Nurse_ratio", lbs_b, lbs_i, n_months, as_rate_per_lb=False
        ),
        surgical_staff_ratio=_pair_series(
            b_df, i_df, "Surgical_ratio", lbs_b, lbs_i, n_months, as_rate_per_lb=False
        ),
    )


def _delivery_shares(df: pd.DataFrame, n_months: int) -> DeliveryLocationSeries:
    home, l23, l4, l5 = [], [], [], []
    for i in range(n_months):
        lb = df.loc[i, "Live Births Final"]
        total = float(np.sum(lb))
        if total <= 0:
            home.append(0.0)
            l23.append(0.0)
            l4.append(0.0)
            l5.append(0.0)
        else:
            home.append(float(lb[0]) / total * 100)
            l23.append(float(lb[1]) / total * 100)
            l4.append(float(lb[2]) / total * 100)
            l5.append(float(lb[3]) / total * 100)
    return DeliveryLocationSeries(home=home, l23=l23, l4=l4, l5=l5)


def _compute_costs(
    b_df: pd.DataFrame,
    i_df: pd.DataFrame,
    i_flags: dict,
    i_param: dict,
    n_months: int,
    int_period: int,
) -> tuple[float, list[CostBreakdownItem], list[float]]:
    cost_dic = {k: v / i_param["USD_to_Ksh"] for k, v in i_param["cost_dict"].items()}
    months = list(range(1, n_months + 1))

    def acum_diff(indicator: str, n_cols: int, order: str = "baseline first") -> pd.DataFrame:
        def prep(df, scenario):
            indicator_vals = np.concatenate(df[indicator].values).reshape(-1, n_cols)
            counts = indicator_vals.sum(axis=1)
            out = pd.DataFrame({"Counts": counts, "Month": np.arange(1, n_months + 1), "Scenario": scenario})
            out["Cumulative_Counts"] = out["Counts"].cumsum()
            return out

        base = prep(b_df, "Baseline")
        inter = prep(i_df, "Intervention")
        diff = base[["Month"]].copy()
        if order == "baseline first":
            diff["Cum_Count_Diff"] = base["Cumulative_Counts"] - inter["Cumulative_Counts"]
            diff["Count_Diff"] = base["Counts"] - inter["Counts"]
        else:
            diff["Cum_Count_Diff"] = inter["Cumulative_Counts"] - base["Cumulative_Counts"]
            diff["Count_Diff"] = inter["Counts"] - base["Counts"]
        diff["Cum_Count_Int"] = inter["Cumulative_Counts"]
        diff["Count_Int"] = inter["Counts"]
        return diff

    daly_avt = acum_diff("DALYs", 4, "baseline first")
    daly_avt = daly_avt.rename(columns={"Cum_Count_Diff": "DALY averted"})
    total_dalys_averted = float(daly_avt["DALY averted"].iloc[-1]) if len(daly_avt) else 0.0

    breakdown: list[CostBreakdownItem] = []
    total_cost = 0.0

    if i_flags.get("flag_pph_bundle"):
        df_pph = acum_diff("Mothers with pph_bundle", 4, "intervention first")
        cost = float((df_pph["Cum_Count_Diff"].clip(lower=0) * cost_dic["pph_bundle"]).iloc[-1])
        breakdown.append(CostBreakdownItem(category="PPH bundle", amount_usd=cost, color_hint="#B5471F"))
        total_cost += cost

    if i_flags.get("flag_iv_iron"):
        df_iv = acum_diff("Mothers with iv_iron", 4, "intervention first")
        cost = float((df_iv["Cum_Count_Diff"].clip(lower=0) * cost_dic["iv_iron"]).iloc[-1])
        breakdown.append(CostBreakdownItem(category="IV iron", amount_usd=cost, color_hint="#2E5F5C"))
        total_cost += cost

    if i_flags.get("flag_MgSO4"):
        df_mg = acum_diff("Mothers with MgSO4", 4, "intervention first")
        cost = float((df_mg["Cum_Count_Diff"].clip(lower=0) * cost_dic["MgSO4"]).iloc[-1])
        breakdown.append(CostBreakdownItem(category="MgSO4", amount_usd=cost, color_hint="#5C4D3C"))
        total_cost += cost

    if i_flags.get("flag_antibiotics"):
        df_ab = acum_diff("Mothers with antibiotics", 4, "intervention first")
        cost = float((df_ab["Cum_Count_Diff"].clip(lower=0) * cost_dic["antibiotics"]).iloc[-1])
        breakdown.append(CostBreakdownItem(category="Antibiotics", amount_usd=cost))
        total_cost += cost

    if i_flags.get("flag_ANC") or i_flags.get("flag_SDR"):
        df_anc = acum_diff("ANC", 4, "intervention first")
        cost = float((df_anc["Cum_Count_Diff"].clip(lower=0) * cost_dic["SDR ANC"]).iloc[-1])
        breakdown.append(CostBreakdownItem(category="CHV / ANC", amount_usd=cost, color_hint="#2E5F5C"))
        total_cost += cost

    if i_flags.get("flag_capacity"):
        fixed = cost_dic.get("SDR Capacity", 0) * (int_period / max(int_period, 1))
        breakdown.append(CostBreakdownItem(category="Facility capacity", amount_usd=fixed, color_hint="#8B6914"))
        total_cost += fixed

    if not breakdown:
        breakdown.append(CostBreakdownItem(category="Minimal intervention cost", amount_usd=0.0))

    cpd_series: list[float] = []
    running_cost = 0.0
    for i, month in enumerate(months):
        month_frac = (i + 1) / n_months
        running_cost = total_cost * month_frac
        daly_at_month = float(daly_avt.loc[daly_avt["Month"] == month, "DALY averted"].iloc[0]) if month <= len(daly_avt) else total_dalys_averted
        cpd = running_cost / daly_at_month if daly_at_month > 0 else 0.0
        cpd_series.append(round(cpd, 2))

    return total_cost, breakdown, cpd_series


def sim_outputs_to_response(
    b_df: pd.DataFrame,
    i_df: pd.DataFrame,
    b_ind: pd.DataFrame,
    i_ind: pd.DataFrame,
    scenario,
    i_flags: dict,
    i_param: dict,
    n_months: int,
    int_period: int,
    runtime_seconds: float,
    n_runs: int = 1,
    seed: Optional[int] = None,
    warnings: Optional[list[str]] = None,
    b_df_runs: Optional[list[pd.DataFrame]] = None,
    i_df_runs: Optional[list[pd.DataFrame]] = None,
) -> ScenarioResult:
    warnings = warnings or []
    lbs_b = _monthly_lb(b_df)
    lbs_i = _monthly_lb(i_df)

    mmr_b = _cumulative_mmr(b_df, lbs_b)
    mmr_i = _cumulative_mmr(i_df, lbs_i)

    ci_lower, ci_upper = None, None
    if b_df_runs and i_df_runs and len(b_df_runs) > 1:
        mmr_b_runs, mmr_i_runs = [], []
        for b_run, i_run in zip(b_df_runs, i_df_runs):
            lb_b = _monthly_lb(b_run)
            lb_i = _monthly_lb(i_run)
            mmr_b_runs.append(_cumulative_mmr(b_run, lb_b))
            mmr_i_runs.append(_cumulative_mmr(i_run, lb_i))
        mmr_i_arr = np.array(mmr_i_runs)
        ci_lower = np.percentile(mmr_i_arr, 2.5, axis=0).tolist()
        ci_upper = np.percentile(mmr_i_arr, 97.5, axis=0).tolist()

    deaths_b = float(b_ind["i_mat_death"].sum()) if "i_mat_death" in b_ind.columns else _vec_sum(b_df["Deaths"])
    deaths_i = float(i_ind["i_mat_death"].sum()) if "i_mat_death" in i_ind.columns else _vec_sum(i_df["Deaths"])
    deaths_averted = deaths_b - deaths_i

    severe_b = _vec_sum(b_df["severe_comps"])
    severe_i = _vec_sum(i_df["severe_comps"])
    severe_averted = severe_b - severe_i

    dalys_b = _vec_sum(b_df["DALYs"])
    dalys_i = _vec_sum(i_df["DALYs"])
    dalys_averted = dalys_b - dalys_i

    total_cost, cost_breakdown, cpd_series = _compute_costs(
        b_df, i_df, i_flags, i_param, n_months, int_period
    )
    cost_per_daly = total_cost / dalys_averted if dalys_averted > 0 else 0.0
    cer_ratio = cost_per_daly / WHO_KENYA_THRESHOLD_USD if WHO_KENYA_THRESHOLD_USD else 0.0

    deaths_by_cause: list[DeathByCause] = []
    for label, col in CAUSE_COLUMNS:
        if col not in b_df.columns:
            continue
        base_c = _vec_sum(b_df[col])
        int_c = _vec_sum(i_df[col])
        av = base_c - int_c
        pct = (av / base_c * 100) if base_c > 0 else 0.0
        deaths_by_cause.append(
            DeathByCause(
                cause=label,
                baseline_count=base_c,
                intervention_count=int_c,
                averted=av,
                percent_reduction=round(pct, 1),
            )
        )

    cap_b = b_df["Capacity Ratio"].apply(lambda x: float(np.mean(x))).values.tolist()
    cap_i = i_df["Capacity Ratio"].apply(lambda x: float(np.mean(x))).values.tolist()

    nurse_ratio = float(i_df["Nurse_ratio"].apply(lambda x: float(np.mean(x))).iloc[-1])
    surgical_ratio = float(i_df["Surgical_ratio"].apply(lambda x: float(np.mean(x))).iloc[-1])

    def adequacy(name: str, pct: float) -> ResourceAdequacy:
        if pct >= 95:
            status = "positive"
        elif pct >= 80:
            status = "warning"
        else:
            status = "negative"
        return ResourceAdequacy(name=name, percent=round(pct * 100, 1), status=status)

    resource = [
        adequacy("Skilled birth attendants", nurse_ratio),
        adequacy("Surgical staff", surgical_ratio),
        adequacy("Facility capacity", cap_i[-1] if cap_i else 0.0),
    ]

    applied = [AppliedIntervention(**x) for x in build_applied_interventions(scenario)]

    narrative_text = (
        f"Over {scenario.run.implementation_years + scenario.run.maintenance_years} years, "
        f"this scenario is projected to avert approximately {deaths_averted:,.0f} maternal deaths "
        f"and {dalys_averted:,.0f} disability-adjusted life years (DALYs). "
    )
    if cost_per_daly > 0 and cost_per_daly < WHO_KENYA_THRESHOLD_USD:
        narrative_text += (
            f"At roughly ${cost_per_daly:,.0f} per DALY averted, the intervention appears "
            f"cost-effective against the WHO Kenya threshold of ${WHO_KENYA_THRESHOLD_USD:,.0f}."
        )
    elif cost_per_daly > 0:
        narrative_text += (
            f"The estimated cost of ${cost_per_daly:,.0f} per DALY averted exceeds the "
            f"WHO Kenya threshold — further refinement may improve value for money."
        )

    indicators = [
        IndicatorAvailable(
            id="facility_capacity",
            name="Facility capacity",
            domain="supply",
            pillar_source="hss",
            is_active=scenario.hss.enabled,
        ),
        IndicatorAvailable(
            id="equipment_capacity",
            name="Equipment capacity",
            domain="supply",
            pillar_source="hss",
            is_active=scenario.hss.enabled,
        ),
        IndicatorAvailable(
            id="supply_capacity",
            name="Supply capacity",
            domain="supply",
            pillar_source="hss",
            is_active=scenario.hss.enabled,
        ),
        IndicatorAvailable(
            id="delivery_location",
            name="Delivery location",
            domain="demand",
            pillar_source="hss",
            is_active=scenario.hss.enabled,
        ),
        IndicatorAvailable(
            id="anc_coverage",
            name="4+ ANC rate",
            domain="demand",
            pillar_source="hss",
            is_active=scenario.hss.enabled,
        ),
        IndicatorAvailable(
            id="anc_rate",
            name="ANC rate",
            domain="process",
            pillar_source="cross-cutting",
            is_active=True,
        ),
        IndicatorAvailable(
            id="cs_rate",
            name="C-section rate",
            domain="process",
            pillar_source="treatments",
            is_active=scenario.treatments.enabled,
        ),
        IndicatorAvailable(
            id="normal_referral",
            name="Normal referral",
            domain="process",
            pillar_source="hss",
            is_active=scenario.hss.enabled,
        ),
        IndicatorAvailable(
            id="emergency_transfer",
            name="Emergency transfer",
            domain="process",
            pillar_source="hss",
            is_active=scenario.hss.enabled,
        ),
        IndicatorAvailable(
            id="high_risk_pregnancy",
            name="High-risk pregnancy",
            domain="process",
            pillar_source="community",
            is_active=scenario.community.enabled,
        ),
        IndicatorAvailable(
            id="maternal_mortality",
            name="Maternal mortality",
            domain="outcomes",
            pillar_source="cross-cutting",
            is_active=True,
        ),
        IndicatorAvailable(
            id="cost_effectiveness",
            name="Cost-effectiveness",
            domain="outcomes",
            pillar_source="cross-cutting",
            is_active=True,
        ),
        IndicatorAvailable(
            id="dalys_averted",
            name="DALYs averted",
            domain="outcomes",
            pillar_source="cross-cutting",
            is_active=True,
        ),
        IndicatorAvailable(
            id="maternal_complication_rate",
            name="Maternal complication rate",
            domain="outcomes",
            pillar_source="cross-cutting",
            is_active=True,
        ),
        IndicatorAvailable(
            id="severe_maternal_outcomes",
            name="Severe maternal outcomes",
            domain="outcomes",
            pillar_source="cross-cutting",
            is_active=True,
        ),
    ]

    return ScenarioResult(
        summary=SummaryResult(
            maternal_deaths_averted=round(deaths_averted, 1),
            severe_maternal_outcomes_averted=round(severe_averted, 1),
            dalys_averted=round(dalys_averted, 1),
            cumulative_cost_usd=round(total_cost, 2),
            cost_per_daly_averted_usd=round(cost_per_daly, 2),
            cost_effectiveness_ratio_to_threshold=round(cer_ratio, 3),
        ),
        applied_interventions=applied,
        narrative=NarrativeResult(
            in_plain_english=narrative_text,
            key_numbers={
                "deaths_averted": f"{deaths_averted:,.0f}",
                "dalys_averted": f"{dalys_averted:,.0f}",
                "cost_per_daly": f"${cost_per_daly:,.0f}",
            },
        ),
        timeseries=TimeseriesResult(
            months=list(range(n_months)),
            maternal_mortality_rate=MaternalMortalitySeries(
                baseline=mmr_b.tolist(),
                intervention=mmr_i.tolist(),
                ci_lower=ci_lower,
                ci_upper=ci_upper,
            ),
            delivery_location=DeliveryLocationTimeseries(
                baseline=_delivery_shares(b_df, n_months),
                intervention=_delivery_shares(i_df, n_months),
            ),
            facility_capacity_ratio=FacilityCapacitySeries(
                baseline=cap_b,
                intervention=cap_i,
                capacity_limit=1.0,
            ),
            cost_per_daly=CostPerDalySeries(
                values=cpd_series,
                threshold_usd=WHO_KENYA_THRESHOLD_USD,
            ),
            indicator_series=_build_indicator_series(b_df, i_df, lbs_b, lbs_i, n_months),
        ),
        indicators_available=indicators,
        resource_adequacy_end_of_run=resource,
        cost_breakdown=cost_breakdown,
        deaths_by_cause=deaths_by_cause,
        meta=MetaResult(
            n_runs=n_runs,
            n_months=n_months,
            runtime_seconds=round(runtime_seconds, 2),
            seed=seed,
            warnings=warnings,
        ),
    )
