from typing import Literal, Optional

from pydantic import BaseModel, Field


class HSSConfig(BaseModel):
    enabled: bool = False
    intensity: Literal["off", "light", "moderate", "intensive"] = "off"
    p_anc: Optional[float] = Field(None, ge=0.0, le=1.0)
    p_l45: Optional[float] = Field(None, ge=0.0, le=1.0)
    capacity_added: Optional[float] = Field(None, ge=0.0, le=1.0)
    chv_memory: Optional[Literal["Always Forget", "Logistic Decay", "Always Remember"]] = None
    refer_enabled: Optional[bool] = None
    transfer_enabled: Optional[bool] = None


class TreatmentsConfig(BaseModel):
    enabled: bool = False
    pph_bundle: bool = False
    iv_iron: bool = False
    mgso4: bool = False
    antibiotics: bool = False
    oxytocin: bool = False
    ultrasound: bool = False


class PromptsConfig(BaseModel):
    enabled: bool = False
    adoption: Optional[float] = Field(None, ge=0.0, le=1.0)
    chv_engagement: Optional[float] = Field(None, ge=0.0, le=1.0)
    intervention_fidelity: Optional[float] = Field(None, ge=0.0, le=1.0)


class MentorsConfig(BaseModel):
    enabled: bool = False
    adoption: Optional[float] = Field(None, ge=0.0, le=1.0)
    attendance: Optional[float] = Field(None, ge=0.0, le=1.0)
    fidelity: Optional[float] = Field(None, ge=0.0, le=1.0)


class FQAConfig(BaseModel):
    enabled: bool = False
    implementation: Literal["low", "high"] = "low"
    influence_on_pulse: Literal["low", "high"] = "low"


class PulseConfig(BaseModel):
    enabled: bool = False
    implementation: Literal["low", "high"] = "low"


class ReferralEMTConfig(BaseModel):
    enabled: bool = False
    emt_participation: Optional[float] = Field(None, ge=0.0, le=1.0)


class CommunityConfig(BaseModel):
    enabled: bool = False
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    mentors: MentorsConfig = Field(default_factory=MentorsConfig)
    fqa: FQAConfig = Field(default_factory=FQAConfig)
    pulse: PulseConfig = Field(default_factory=PulseConfig)
    referral_emt: ReferralEMTConfig = Field(default_factory=ReferralEMTConfig)


class RunConfig(BaseModel):
    implementation_years: int = Field(3, ge=1, le=10)
    maintenance_years: int = Field(1, ge=0, le=10)
    mode: Literal["quick", "robust"] = "quick"


class ScenarioRequest(BaseModel):
    name: str = "My scenario"
    county: Literal["kakamega"] = "kakamega"
    hss: HSSConfig = Field(default_factory=HSSConfig)
    treatments: TreatmentsConfig = Field(default_factory=TreatmentsConfig)
    community: CommunityConfig = Field(default_factory=CommunityConfig)
    run: RunConfig = Field(default_factory=RunConfig)


class RunScenarioRequest(BaseModel):
    scenario: ScenarioRequest


class CompareRequest(BaseModel):
    scenario_a: ScenarioRequest
    scenario_b: ScenarioRequest
    mode: Literal["quick", "robust"] = "quick"
