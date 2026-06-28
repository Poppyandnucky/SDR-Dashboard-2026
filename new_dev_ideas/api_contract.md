# API Contract — `sdr-api`

All endpoints under base path `/api/v1`. JSON request/response. CORS-restricted to the frontend domain.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/meta/counties` | List supported counties |
| `GET` | `/api/v1/meta/parameters` | Parameter schema (intensity ranges, etc.) |
| `GET` | `/api/v1/presets` | The four starter presets |
| `POST` | `/api/v1/scenarios/run` | Run a single scenario |
| `GET` | `/api/v1/scenarios/{run_id}` | Get a previously-run result |
| `POST` | `/api/v1/scenarios/compare` | Run two scenarios as a comparison |
| `GET` | `/api/v1/scenarios/compare/{comparison_id}` | Get a comparison result |
| `GET` | `/health` | Liveness probe |

---

## `GET /api/v1/meta/counties`

```json
{
  "counties": [
    { "id": "kakamega", "name": "Kakamega", "calibrated": true, "population": 1872000 },
    { "id": "kisumu",   "name": "Kisumu",   "calibrated": false, "available": "Q3 2026" },
    { "id": "nairobi",  "name": "Nairobi",  "calibrated": false, "available": "Q4 2026" },
    { "id": "bungoma",  "name": "Bungoma",  "calibrated": false, "available": "Q4 2026" }
  ]
}
```

Only `kakamega` returns real data when used in `POST /scenarios/run`. Others return `404 Not Found`.

---

## `GET /api/v1/meta/parameters`

Returns the schema for intensity buckets and parameter ranges. The frontend uses this to render the "Light · 60–69%" labels.

```json
{
  "intensity_ranges": {
    "hss": {
      "off":       { "label": "Off",       "range": [0, 0],    "description": "Baseline / no HSS intervention" },
      "light":     { "label": "Light",     "range": [60, 69],  "description": "Modest demand-side investment" },
      "moderate":  { "label": "Moderate",  "range": [70, 79],  "description": "Stronger demand + matching supply" },
      "intensive": { "label": "Intensive", "range": [80, 95],  "description": "Aggressive demand + matched supply" }
    },
    "fqa":   { "low": { "label": "Low" }, "high": { "label": "High" } },
    "pulse": { "low": { "label": "Low" }, "high": { "label": "High" } }
  },
  "timeline_constraints": {
    "min_total_years": 1,
    "max_total_years": 10,
    "default_implementation_years": 3,
    "default_maintenance_years": 1
  }
}
```

---

## `GET /api/v1/presets`

```json
{
  "presets": [
    {
      "id": "status-quo",
      "name": "Status quo",
      "subtitle": "Baseline only",
      "description": "Today's conditions projected forward.",
      "scenario": { "...": "see Scenario schema below" }
    },
    {
      "id": "hss-intensive",
      "name": "Health system strengthening",
      "subtitle": "HSS · Intensive",
      "description": "Aggressive demand + matched supply.",
      "scenario": { "...": "..." }
    },
    {
      "id": "momish",
      "name": "Community engagement (MOMISH)",
      "subtitle": "PROMPTS · Full · CHVs strong",
      "description": "PROMPTS full rollout, CHV strongly engaged.",
      "scenario": { "...": "..." }
    },
    {
      "id": "combined",
      "name": "Combined strategy",
      "subtitle": "Recommended",
      "description": "HSS Moderate + drugs + MOMISH partial.",
      "scenario": { "...": "..." },
      "is_recommended": true
    }
  ]
}
```

Preset values are hardcoded in `sdr-api/app/data/presets.py` against the actual values from `sim/parameters.py`. **The frontend must use these — don't hardcode preset values in the Next.js code.**

---

## `POST /api/v1/scenarios/run`

### Request body — `ScenarioRequest`

```typescript
{
  // Identity
  name: string;                      // user-editable scenario name
  county: "kakamega";                // only kakamega supported in v1

  // Pillar 1: Health System Strengthening
  hss: {
    enabled: boolean;
    intensity: "off" | "light" | "moderate" | "intensive";
    // Optional fine-tuning within the bucket:
    p_anc?: number;                  // 0.0–1.0
    p_l45?: number;                  // 0.0–1.0
    capacity_added?: number;         // 0.0–1.0
    chv_memory?: "Always Forget" | "Logistic Decay" | "Always Remember";
    refer_enabled?: boolean;
    transfer_enabled?: boolean;
  };

  // Pillar 2: Treatments
  treatments: {
    enabled: boolean;
    pph_bundle?: boolean;
    iv_iron?: boolean;
    mgso4?: boolean;
    antibiotics?: boolean;
    oxytocin?: boolean;
    ultrasound?: boolean;            // flag_us in sim
  };

  // Pillar 3: Community (MOMISH)
  community: {
    enabled: boolean;

    prompts: {                       // wired in sim
      enabled: boolean;
      adoption?: number;             // 0.0–1.0
      chv_engagement?: number;       // 0.0–1.0
      intervention_fidelity?: number; // 0.0–1.0
    };

    mentors: {                       // wired in sim
      enabled: boolean;
      adoption?: number;
      attendance?: number;
      fidelity?: number;
    };

    fqa: {                           // UI only — model wiring pending
      enabled: boolean;
      implementation: "low" | "high";
      influence_on_pulse: "low" | "high";
    };

    pulse: {                         // UI only — model wiring pending
      enabled: boolean;
      implementation: "low" | "high";
    };

    referral_emt: {                  // partial wiring in sim
      enabled: boolean;
      emt_participation?: number;    // 0.0–1.0
    };
  };

  // Run settings
  run: {
    implementation_years: number;    // default 3
    maintenance_years: number;       // default 1
    mode: "quick" | "robust";        // quick = 1 sim, robust = 50 sims
  };
}
```

### Response — Quick mode (synchronous)

`200 OK` with the result inline:

```typescript
{
  run_id: string;                    // UUID
  status: "complete";
  scenario: ScenarioRequest;         // echoed back
  result: ScenarioResult;            // see below
  completed_at: string;              // ISO timestamp
}
```

### Response — Robust mode (async)

`202 Accepted`:

```typescript
{
  run_id: string;
  status: "pending";
  estimated_seconds_remaining: number;
}
```

Then the frontend polls `GET /scenarios/{run_id}` until `status === "complete"`.

---

## `ScenarioResult` schema

This is the response shape. Designed to be directly consumable by the Results page components.

```typescript
{
  // Top-line summary numbers
  summary: {
    maternal_deaths_averted: number;
    severe_maternal_outcomes_averted: number;
    dalys_averted: number;
    cumulative_cost_usd: number;
    cost_per_daly_averted_usd: number;
    cost_effectiveness_ratio_to_threshold: number;  // < 1 = cost-effective
  };

  // For the assumptions callout
  applied_interventions: Array<{
    pillar: "hss" | "treatments" | "community";
    name: string;                    // "PPH bundle"
    intensity?: string;              // "Intensive" | "Light" etc.
    is_wired_in_model: boolean;      // false for FQA, PULSE
  }>;

  // For the narrative summary
  narrative: {
    in_plain_english: string;        // pre-rendered editorial text
    key_numbers: Record<string, string>;
  };

  // Per-story time series
  timeseries: {
    months: number[];                // [0, 1, 2, ..., 48]

    maternal_mortality_rate: {
      baseline: number[];            // per 100k LB
      intervention: number[];
      ci_lower?: number[];           // Robust mode only
      ci_upper?: number[];
    };

    delivery_location: {
      // Share of births at each level over time
      baseline: { home: number[], l23: number[], l4: number[], l5: number[] };
      intervention: { home: number[], l23: number[], l4: number[], l5: number[] };
    };

    facility_capacity_ratio: {
      baseline: number[];
      intervention: number[];
      capacity_limit: number;        // typically 1.0
    };

    cost_per_daly: {
      values: number[];              // monthly running cost-per-DALY
      threshold_usd: number;         // WHO Kenya threshold
    };
  };

  // For the indicator drawer
  indicators_available: Array<{
    id: string;                      // "facility_capacity"
    name: string;                    // "Facility capacity"
    domain: "supply" | "demand" | "process" | "outcomes";
    pillar_source: "hss" | "treatments" | "community" | "cross-cutting";
    is_active: boolean;              // false if its pillar is off
  }>;

  // For Story 04 (System pressure)
  resource_adequacy_end_of_run: Array<{
    name: string;                    // "Skilled birth attendants"
    percent: number;                 // 102 = 102% adequate
    status: "positive" | "warning" | "negative";
  }>;

  // For Story 01 (Cost)
  cost_breakdown: Array<{
    category: string;                // "CHV / ANC"
    amount_usd: number;
    color_hint?: string;             // for chart consistency
  }>;

  // For Story 02 (Mothers)
  deaths_by_cause: Array<{
    cause: string;                   // "Postpartum haemorrhage"
    baseline_count: number;
    intervention_count: number;
    averted: number;
    percent_reduction: number;
  }>;

  // Diagnostic metadata
  meta: {
    n_runs: number;                  // 1 for Quick, 50 for Robust
    n_months: number;                // 48 typically
    runtime_seconds: number;
    seed?: number;
    warnings: string[];              // e.g. "FQA was set but not wired into model"
  };
}
```

---

## `POST /api/v1/scenarios/compare`

### Request

```typescript
{
  scenario_a: ScenarioRequest;
  scenario_b: ScenarioRequest;
  mode: "quick" | "robust";          // applies to both
}
```

### Response

```typescript
{
  comparison_id: string;
  status: "complete" | "pending";
  scenario_a: ScenarioRequest;
  scenario_b: ScenarioRequest;
  result_a: ScenarioResult;
  result_b: ScenarioResult;

  // Pre-computed deltas for the Compare Results KPI tiles
  deltas: {
    maternal_deaths_averted: { a: number, b: number, diff: number, pct: number };
    severe_maternal_outcomes_averted: { a: number, b: number, diff: number, pct: number };
    dalys_averted: { a: number, b: number, diff: number, pct: number };
    cost_per_daly_averted_usd: { a: number, b: number, diff: number };
    cumulative_cost_usd: { a: number, b: number, diff: number };
  };

  // Pre-rendered combined narrative
  combined_narrative: string;
}
```

---

## Error responses

All errors follow this shape:

```typescript
{
  error: {
    code: string;          // "invalid_scenario", "sim_error", "run_not_found", ...
    message: string;       // human-readable, safe to display to users
    details?: object;      // optional structured details
  }
}
```

Common error codes:

| HTTP | Code | When |
|---|---|---|
| 400 | `invalid_scenario` | Pydantic validation failed |
| 404 | `run_not_found` | `run_id` doesn't exist or expired |
| 404 | `county_not_supported` | County other than Kakamega requested |
| 422 | `sim_error` | Simulation raised an exception during run |
| 429 | `rate_limited` | Too many runs from one IP |
| 503 | `sim_unavailable` | Simulation module failed to import |

**Never expose Python tracebacks in error responses.** Log them server-side, return a sanitized message.

---

## Adapter implementation notes

The adapter layer (`sdr-api/app/adapters/`) is the most fragile part of the API. Here's the contract:

### `scenario_to_sim.py`

```python
def scenario_to_sim_inputs(req: ScenarioRequest) -> tuple[dict, dict, dict]:
    """
    Translates a clean API request into the three dicts the sim expects:
      - i_flags: Dict[str, int]   — feature flags ('flag_SDR', 'flag_PROMPTS', etc.)
      - i_HSS:   Dict[str, Any]   — HSS pillar parameters
      - i_param: Dict[str, Any]   — top-level model parameters
    """
```

Use `sim/SDR_Dash.py` as the reference for what these dicts should contain. Search for `i_flags = ` and `i_HSS = ` to see how the existing code builds them.

### `sim_to_response.py`

```python
def sim_outputs_to_response(
    baseline_df: pd.DataFrame,
    intervention_df: pd.DataFrame,
    individual_outcomes: pd.DataFrame,
    scenario: ScenarioRequest,
) -> ScenarioResult:
    """
    Translates the sim's DataFrame outputs into the typed JSON shape.
    """
```

The DataFrames have specific column names defined in `sim/global_func.py` (look for `Track` dicts and the `prepare_df_ce` function in `sim/SDR_Dash.py`). Match those exactly.

### Warnings

When the request includes a field for a UI-only intervention (FQA, PULSE), the adapter should:
1. Accept the value.
2. Not pass it to the sim (since the sim doesn't know about it).
3. Append a warning to `result.meta.warnings` like:
   `"FQA was set to {implementation: 'high'} but is not yet wired into the simulation model. Result reflects PROMPTS/MENTORS settings only."`

The frontend displays warnings in a small footnote on Results.
