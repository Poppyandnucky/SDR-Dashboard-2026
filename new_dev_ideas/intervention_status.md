# Intervention Status

This is the canonical reference for which interventions are **fully wired** into the simulation versus **UI only** (toggles exist but model logic is incomplete or absent). Based on a code audit of `sim/SDR_Dash.py`, `sim/LB_effect.py`, `sim/global_func.py`, and supporting files as of the v0.7.1 mockup.

**Use this when:**
- Building the FastAPI adapter — you'll know which fields to forward to the sim vs which to silently drop
- Setting `result.meta.warnings` — UI-only interventions trigger a warning in the result
- Rendering the Compare scenarios Intervention Library — the "● wired" / "● UI only" badges come from here

---

## Quick reference

| Pillar | Intervention | Status | Sim flag | Sim parameters |
|---|---|---|---|---|
| **HSS** | Demand: ANC | ✓ wired | `flag_ANC` | `i_HSS["P_ANC"]` |
| HSS | Demand: CHV outreach | ✓ wired | `flag_CHV` | `i_HSS["CHV_memory"]`, `tau_decay` |
| HSS | Demand: LB shift to L4/5 | ✓ wired | `flag_LB` | `i_HSS["P_L45"]` |
| HSS | Supply: facility capacity | ✓ wired | `flag_capacity`, `flag_SDR` | `i_HSS["capacity_added"]` |
| HSS | Supply: performance | ✓ wired | `flag_performance` | `i_HSS["knowledge"]` |
| HSS | Supply: labor | ✓ wired | `flag_labor` | `i_HSS["labor_ratio"]` |
| HSS | Supply: equipment | ✓ wired | `flag_equipment` | — |
| HSS | Supply: referral | ✓ wired | `flag_refer` | `i_HSS["P_refer"]` |
| HSS | Supply: emergency transfer | ✓ wired | `flag_transfer` | `i_HSS["P_transfer"]` |
| **Treatments** | PPH bundle | ✓ wired | `flag_pph_bundle` | — |
| Treatments | IV iron | ✓ wired | `flag_iv_iron` | — |
| Treatments | MgSO4 | ✓ wired | `flag_MgSO4` | — |
| Treatments | Antibiotics | ✓ wired | `flag_antibiotics` | — |
| Treatments | Oxytocin | ✓ wired | `flag_oxytocin` | — |
| Treatments | POCUS / ultrasound | ✓ wired | `flag_us` | — |
| **Community** | PROMPTS | ✓ wired | `flag_PROMPTS` | `i_HSS["adoption_prompts"]`, `chv_engagement`, `intervention_fidelity`, `OR_anc4p` |
| Community | MENTORS | ✓ wired | `flag_MENTOR` | `i_HSS["mentor_adoption"]`, `mentor_attendance`, `mentor_fidelity` |
| Community | **FQA** | ⚠ UI only | (none yet) | (none yet) |
| Community | **PULSE** | ⚠ UI only | `flag_pulse` exists | (no model logic reads it) |
| Community | Referral / EMT | ⚠ partial | `flag_emt` | `i_HSS["emt_participation"]` (slider exists, not fully consumed) |

---

## What "wired" means

**Wired** (✓): The simulation model (`LB_effect.py`, `intrapartum.py`, `mortality.py`, `global_func.py`) reads the flag and/or parameters and uses them to alter outcomes. Changing the value in the UI affects the result numbers.

**UI only** (⚠): A toggle or parameter exists in the current Streamlit dashboard, and may even be in the `i_flags` or `i_HSS` dict, but the simulation logic does not read it. Changing the value in the UI does nothing.

**Partial** (⚠ partial): Some hooks exist in the simulation but the implementation is incomplete or conditional. Treat as UI-only for now.

---

## FQA and PULSE — UI controls only

Both interventions need UI controls in the new design, per spec from the project lead. The simulation team is wiring them in a future sprint. The new UI should:

1. Render the controls fully (sliders, toggles, etc.) as designed in the mockup.
2. Accept the values in the API request.
3. Store them on the scenario for round-tripping (URL, share link).
4. **Not pass them to the simulation** — the adapter drops them silently.
5. Add a warning to `result.meta.warnings` when the user has set these:
   - `"FQA was configured (implementation: high, influence_on_pulse: low) but is not yet wired into the simulation model. Result reflects PROMPTS / MENTORS / HSS / Treatments settings only."`
   - `"PULSE was configured (implementation: high) but is not yet wired into the simulation model."`

### FQA controls

Per the project lead's spec:

| Control | Type | Values |
|---|---|---|
| Implementation | Two-level pill | `low`, `high` |
| Influence on PULSE | Two-level pill | `low`, `high` |

The "influence on PULSE" is a **cross-intervention dependency** — when the model gets wired, this control will modulate how strongly PULSE behaves. For now, both controls are stored on the scenario but ignored.

### PULSE controls

| Control | Type | Values |
|---|---|---|
| Implementation | Two-level pill | `low`, `high` |

---

## Referral / EMT — partial

The simulation has `flag_emt` and `i_HSS["emt_participation"]` (0.0–1.0 slider for emergency vehicle capacity). However:
- It's not consistently consumed throughout the model
- The mockup shows it in the Compare library with a "● partial" badge

**Recommendation for the new UI:**
- Render the toggle and participation slider
- Forward the value to the sim
- The sim may or may not act on it depending on which file is consuming it
- Add a milder warning if set: `"Referral / EMT model wiring is partial. Some effects may not reflect in outcomes."`

---

## Reference: HSS preset values

These come straight from `sim/global_func.py` (`reset_HSS`) and the `Demand_scenarios` / `Capacity_match` dicts in `sim/SDR_Dash.py`. The API's `/presets` endpoint must use these exact values.

### Status quo

```python
i_flags = { "flag_SDR": 0, "flag_PROMPTS": 0, "flag_CHV": 0, ... all flags 0 ... }
i_HSS = {
  "P_ANC": 0.56,
  "P_L45": 0.381,
  "knowledge": 0.60,
  "capacity_added": 0,
  "CHV_memory": "Always Forget",
  ...
}
```

Drug supply (from `parameters.py`): `iv_iron: 0.44`, `MgSO4: 0.77`, `antibiotics: 0.48`, `oxytocin: 0.78`.

### HSS · Intensive (Aggressive + Match Demand)

```python
i_flags = {
  "flag_SDR": 1, "flag_CHV": 1, "flag_ANC": 1, "flag_LB": 1,
  "flag_performance": 1, "flag_capacity": 1, "flag_labor": 1,
  "flag_equipment": 1, "flag_refer": 1, "flag_transfer": 1,
}
i_HSS = {
  "P_ANC": 0.90,
  "P_L45": 0.90,
  "knowledge": 1.0,
  "capacity_added": 0.85,        # Capacity_match["Aggressive"] / 100
  "labor_ratio": 1.0,
  "P_refer": 1.0,
  "P_transfer": 1.0,
  "CHV_memory": "Logistic Decay",
  "tau_decay": 6,
}
```

### MOMISH (PROMPTS Full · CHVs strong)

```python
i_flags = { "flag_PROMPTS": 1, ... }
i_HSS = {
  "adoption_prompts": 1.0,
  "chv_engagement": 1.0,
  "intervention_fidelity": 0.87,
  # MENTORS:
  "mentor_adoption": 0.8,  # check actual default
  "mentor_attendance": 0.8,
  "mentor_fidelity": 0.8,
}
```

(Verify mentor defaults against the actual Streamlit code's `render_prompts` function.)

### Combined (HSS Moderate + drugs + MOMISH partial)

```python
i_flags = {
  "flag_SDR": 1, "flag_capacity": 1, "flag_PROMPTS": 1,
  "flag_pph_bundle": 1, "flag_MgSO4": 1,
}
i_HSS = {
  "P_ANC": 0.80,
  "P_L45": 0.68,
  "capacity_added": 0.50,        # Capacity_match["Moderate"] / 100
  "adoption_prompts": 0.6,
  "chv_engagement": 0.6,
  "knowledge": 0.85,
}
```

**Verify all these values against the actual code before hardcoding them in the API.** The exact numbers may have shifted since this audit. Use `sim/SDR_Dash.py`'s `render_hss(preset_demand_scenario="Aggressive", preset_supply_scenario="Match Demand")` flow as the reference.

---

## How to update this document

When the simulation team wires a new intervention:

1. Find the new `flag_X` in the sim files.
2. Trace where it's read (search for `flags["flag_X"]`).
3. Confirm the flag actually changes outcomes (not just a placeholder).
4. Update the table above: status goes from ⚠ to ✓.
5. Remove the corresponding warning from the API's adapter.
6. Update the badge in the Compare scenarios Intervention Library in the frontend.
