export type HSSIntensity = "off" | "light" | "moderate" | "intensive";
export type RunMode = "quick" | "robust";

export interface Scenario {
  name: string;
  county: "kakamega";
  hss: {
    enabled: boolean;
    intensity: HSSIntensity;
    p_anc?: number;
    p_l45?: number;
    capacity_added?: number;
    chv_memory?: string;
    refer_enabled?: boolean;
    transfer_enabled?: boolean;
  };
  treatments: {
    enabled: boolean;
    pph_bundle?: boolean;
    iv_iron?: boolean;
    mgso4?: boolean;
    antibiotics?: boolean;
    oxytocin?: boolean;
    ultrasound?: boolean;
  };
  community: {
    enabled: boolean;
    prompts: {
      enabled: boolean;
      adoption?: number;
      chv_engagement?: number;
      intervention_fidelity?: number;
    };
    mentors: {
      enabled: boolean;
      adoption?: number;
      attendance?: number;
      fidelity?: number;
    };
    fqa: {
      enabled: boolean;
      implementation: "low" | "high";
      influence_on_pulse: "low" | "high";
    };
    pulse: {
      enabled: boolean;
      implementation: "low" | "high";
    };
    referral_emt: {
      enabled: boolean;
      emt_participation?: number;
    };
  };
  run: {
    implementation_years: number;
    maintenance_years: number;
    mode: RunMode;
  };
}

export const DEFAULT_SCENARIO: Scenario = {
  name: "My scenario",
  county: "kakamega",
  hss: { enabled: false, intensity: "off" },
  treatments: { enabled: false },
  community: {
    enabled: false,
    prompts: { enabled: false },
    mentors: { enabled: false },
    fqa: { enabled: false, implementation: "low", influence_on_pulse: "low" },
    pulse: { enabled: false, implementation: "low" },
    referral_emt: { enabled: false },
  },
  run: { implementation_years: 3, maintenance_years: 1, mode: "quick" },
};

export interface Preset {
  id: string;
  name: string;
  subtitle: string;
  description: string;
  is_recommended?: boolean;
  scenario: Scenario;
}

export interface ScenarioSummary {
  maternal_deaths_averted: number;
  severe_maternal_outcomes_averted: number;
  dalys_averted: number;
  cumulative_cost_usd: number;
  cost_per_daly_averted_usd: number;
  cost_effectiveness_ratio_to_threshold: number;
}

export interface ScenarioResult {
  summary: ScenarioSummary;
  applied_interventions: Array<{
    pillar: string;
    name: string;
    intensity?: string;
    is_wired_in_model: boolean;
  }>;
  narrative: { in_plain_english: string; key_numbers: Record<string, string> };
  timeseries: {
    months: number[];
    maternal_mortality_rate: {
      baseline: number[];
      intervention: number[];
      ci_lower?: number[];
      ci_upper?: number[];
    };
    delivery_location: {
      baseline: { home: number[]; l23: number[]; l4: number[]; l5: number[] };
      intervention: { home: number[]; l23: number[]; l4: number[]; l5: number[] };
    };
    facility_capacity_ratio: {
      baseline: number[];
      intervention: number[];
      capacity_limit: number;
    };
    cost_per_daly: { values: number[]; threshold_usd: number };
    indicator_series?: {
      anc_rate_per_100_lb: { baseline: number[]; intervention: number[] };
      cs_rate_per_100_lb: { baseline: number[]; intervention: number[] };
      normal_referral_per_100_lb: { baseline: number[]; intervention: number[] };
      emergency_transfer_per_100_lb: { baseline: number[]; intervention: number[] };
      high_risk_per_100_lb: { baseline: number[]; intervention: number[] };
      maternal_complication_rate_per_100_lb: { baseline: number[]; intervention: number[] };
      severe_maternal_outcomes_per_100_lb: { baseline: number[]; intervention: number[] };
      doppler_equipment_ratio: { baseline: number[]; intervention: number[] };
      ctg_equipment_ratio: { baseline: number[]; intervention: number[] };
      nurse_staff_ratio: { baseline: number[]; intervention: number[] };
      surgical_staff_ratio: { baseline: number[]; intervention: number[] };
    };
  };
  indicators_available: Array<{
    id: string;
    name: string;
    domain: string;
    pillar_source: string;
    is_active: boolean;
  }>;
  resource_adequacy_end_of_run: Array<{
    name: string;
    percent: number;
    status: "positive" | "warning" | "negative";
  }>;
  cost_breakdown: Array<{ category: string; amount_usd: number; color_hint?: string }>;
  deaths_by_cause: Array<{
    cause: string;
    baseline_count: number;
    intervention_count: number;
    averted: number;
    percent_reduction: number;
  }>;
  meta: {
    n_runs: number;
    n_months: number;
    runtime_seconds: number;
    warnings: string[];
  };
}

export interface RunResponse {
  run_id: string;
  status: "complete" | "pending" | "failed";
  scenario: Scenario;
  result?: ScenarioResult;
  completed_at?: string;
  estimated_seconds_remaining?: number;
  error_message?: string;
}

export interface CompareResponse {
  comparison_id: string;
  status: string;
  scenario_a: Scenario;
  scenario_b: Scenario;
  result_a?: ScenarioResult;
  result_b?: ScenarioResult;
  deltas?: Record<string, { a: number; b: number; diff: number; pct?: number }>;
  combined_narrative?: string;
}
