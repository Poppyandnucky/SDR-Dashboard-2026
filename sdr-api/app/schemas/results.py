from typing import Any, Literal, Optional

from pydantic import BaseModel

from app.schemas.scenario import ScenarioRequest


class SummaryResult(BaseModel):
    maternal_deaths_averted: float
    severe_maternal_outcomes_averted: float
    dalys_averted: float
    cumulative_cost_usd: float
    cost_per_daly_averted_usd: float
    cost_effectiveness_ratio_to_threshold: float


class AppliedIntervention(BaseModel):
    pillar: Literal["hss", "treatments", "community"]
    name: str
    intensity: Optional[str] = None
    is_wired_in_model: bool = True


class NarrativeResult(BaseModel):
    in_plain_english: str
    key_numbers: dict[str, str]


class MaternalMortalitySeries(BaseModel):
    baseline: list[float]
    intervention: list[float]
    ci_lower: Optional[list[float]] = None
    ci_upper: Optional[list[float]] = None


class DeliveryLocationSeries(BaseModel):
    home: list[float]
    l23: list[float]
    l4: list[float]
    l5: list[float]


class DeliveryLocationTimeseries(BaseModel):
    baseline: DeliveryLocationSeries
    intervention: DeliveryLocationSeries


class FacilityCapacitySeries(BaseModel):
    baseline: list[float]
    intervention: list[float]
    capacity_limit: float = 1.0


class CostPerDalySeries(BaseModel):
    values: list[float]
    threshold_usd: float


class BaselineInterventionSeries(BaseModel):
    baseline: list[float]
    intervention: list[float]


class IndicatorTimeseriesBundle(BaseModel):
    """Monthly indicator series derived from sim DataFrames (display-only)."""

    anc_rate_per_100_lb: BaselineInterventionSeries
    cs_rate_per_100_lb: BaselineInterventionSeries
    normal_referral_per_100_lb: BaselineInterventionSeries
    emergency_transfer_per_100_lb: BaselineInterventionSeries
    high_risk_per_100_lb: BaselineInterventionSeries
    maternal_complication_rate_per_100_lb: BaselineInterventionSeries
    severe_maternal_outcomes_per_100_lb: BaselineInterventionSeries
    doppler_equipment_ratio: BaselineInterventionSeries
    ctg_equipment_ratio: BaselineInterventionSeries
    nurse_staff_ratio: BaselineInterventionSeries
    surgical_staff_ratio: BaselineInterventionSeries


class TimeseriesResult(BaseModel):
    months: list[int]
    maternal_mortality_rate: MaternalMortalitySeries
    delivery_location: DeliveryLocationTimeseries
    facility_capacity_ratio: FacilityCapacitySeries
    cost_per_daly: CostPerDalySeries
    indicator_series: IndicatorTimeseriesBundle


class IndicatorAvailable(BaseModel):
    id: str
    name: str
    domain: Literal["supply", "demand", "process", "outcomes"]
    pillar_source: Literal["hss", "treatments", "community", "cross-cutting"]
    is_active: bool


class ResourceAdequacy(BaseModel):
    name: str
    percent: float
    status: Literal["positive", "warning", "negative"]


class CostBreakdownItem(BaseModel):
    category: str
    amount_usd: float
    color_hint: Optional[str] = None


class DeathByCause(BaseModel):
    cause: str
    baseline_count: float
    intervention_count: float
    averted: float
    percent_reduction: float


class MetaResult(BaseModel):
    n_runs: int
    n_months: int
    runtime_seconds: float
    seed: Optional[int] = None
    warnings: list[str] = []


class ScenarioResult(BaseModel):
    summary: SummaryResult
    applied_interventions: list[AppliedIntervention]
    narrative: NarrativeResult
    timeseries: TimeseriesResult
    indicators_available: list[IndicatorAvailable]
    resource_adequacy_end_of_run: list[ResourceAdequacy]
    cost_breakdown: list[CostBreakdownItem]
    deaths_by_cause: list[DeathByCause]
    meta: MetaResult


class RunResponse(BaseModel):
    run_id: str
    status: Literal["complete", "pending", "failed"]
    scenario: ScenarioRequest
    result: Optional[ScenarioResult] = None
    completed_at: Optional[str] = None
    estimated_seconds_remaining: Optional[int] = None
    error_message: Optional[str] = None


class DeltaMetric(BaseModel):
    a: float
    b: float
    diff: float
    pct: Optional[float] = None


class CompareDeltas(BaseModel):
    maternal_deaths_averted: DeltaMetric
    severe_maternal_outcomes_averted: DeltaMetric
    dalys_averted: DeltaMetric
    cost_per_daly_averted_usd: DeltaMetric
    cumulative_cost_usd: DeltaMetric


class CompareResponse(BaseModel):
    comparison_id: str
    status: Literal["complete", "pending"]
    scenario_a: ScenarioRequest
    scenario_b: ScenarioRequest
    result_a: Optional[ScenarioResult] = None
    result_b: Optional[ScenarioResult] = None
    deltas: Optional[CompareDeltas] = None
    combined_narrative: Optional[str] = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[dict[str, Any]] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
