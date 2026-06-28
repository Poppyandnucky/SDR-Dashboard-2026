# SDR Decision Tool — Implementation Guide

> **Audience.** This document is written for a developer (human, Cursor, or Claude Code) implementing the redesigned **Kenya Maternal Health Decision Tool** from the v0.7.1 HTML mockup. It assumes familiarity with Python, FastAPI, Next.js, TypeScript, and Tailwind. It does **not** assume familiarity with the SDR simulation; that's explained as needed.

> **Goal.** Replace the current Streamlit dashboard (`SDR_Dash.py`) with a new Next.js frontend backed by a FastAPI wrapper around the existing calibrated simulation. **The Python simulation core is not to be modified** — we wrap, we don't rewrite.

---

## Quick context

The current system is a Streamlit application (`SDR_Dash.py`, ~3000 lines) that wraps an agent-based maternal-health simulation. Stakeholders find it powerful but unusable without repeat training. We are not rewriting the simulation — only replacing the UI layer and adding a thin API between them.

**What you'll build:**

1. **`sdr-api/`** — A FastAPI service that exposes the existing simulation behind clean HTTP endpoints. Wraps `model_run.py`, `parameters.py`, `LB_effect.py`, `intrapartum.py`, `mortality.py`, `global_func.py` without touching them.
2. **`sdr-web/`** — A Next.js (App Router) + TypeScript + Tailwind frontend matching `SDR_Dashboard_Mockup.html` (v0.7.1).
3. **Deployment** — Both services on Railway.

**What you won't build:**
- Any change to the simulation logic (`LB_effect.py`, `intrapartum.py`, `mortality.py`, etc.). If you find yourself editing these, stop and ask.
- The analyst view (out of scope for v1).
- Authentication, user accounts, or persistence beyond URL-encoded scenarios.

---

## Repo structure

Recommended monorepo layout:

```
sdr-decision-tool/
├── README.md
├── sim/                          # Existing Python simulation — DO NOT MODIFY
│   ├── SDR_Dash.py               # Old Streamlit app (reference only after migration)
│   ├── model_run.py
│   ├── parameters.py
│   ├── LB_effect.py
│   ├── intrapartum.py
│   ├── mortality.py
│   ├── parameter_calculation.py
│   ├── global_func.py
│   ├── ANC_LB_effect_slider.py
│   └── config.json
│
├── sdr-api/                      # NEW — FastAPI wrapper
│   ├── app/
│   │   ├── main.py               # FastAPI entry point + CORS
│   │   ├── routes/
│   │   │   ├── scenarios.py      # POST /scenarios/run, /scenarios/compare
│   │   │   ├── presets.py        # GET /presets
│   │   │   └── meta.py           # GET /meta/parameters, /meta/counties
│   │   ├── adapters/
│   │   │   ├── scenario_to_sim.py    # Pydantic Scenario → sim's i_flags/i_HSS/i_param
│   │   │   └── sim_to_response.py    # sim DataFrames → typed JSON
│   │   └── schemas/
│   │       ├── scenario.py       # Pydantic models for request bodies
│   │       └── results.py        # Pydantic models for response shapes
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
│
├── sdr-web/                      # NEW — Next.js frontend
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx              # Start screen
│   │   ├── design/page.tsx       # Design & Run screen
│   │   ├── compare/page.tsx      # Compare scenarios builder
│   │   ├── compare/results/page.tsx
│   │   ├── results/page.tsx      # Single-scenario results
│   │   └── about/page.tsx        # About the Model
│   ├── components/
│   │   ├── TopNav.tsx
│   │   ├── PillSelector.tsx
│   │   ├── InterventionCard.tsx
│   │   ├── KPITile.tsx
│   │   ├── stories/              # Per-story components
│   │   └── modals/
│   ├── lib/
│   │   ├── api.ts                # Typed fetch client for sdr-api
│   │   ├── scenarios.ts          # Scenario state types + serialization
│   │   └── url-state.ts          # Encode/decode scenarios in URL
│   ├── styles/globals.css
│   ├── tailwind.config.ts
│   ├── package.json
│   └── tsconfig.json
│
└── new_dev_direction/            # This folder
    ├── SDR_Dashboard_Mockup.html
    ├── IMPLEMENTATION_GUIDE.md       (this file)
    ├── architecture.md
    ├── api_contract.md
    ├── ui_components.md
    └── intervention_status.md
```

**Important:** `sim/` is read-only. If you must change something there, it's a sign you need an adapter layer between sim and API instead.

---

## Phased build

Each phase produces something testable. Don't try to do them in parallel.

### Phase 1 — FastAPI wrapper (3–5 days)

**Goal:** Expose the existing simulation behind HTTP without changing it. By the end of this phase, you should be able to `curl` an endpoint and get the same numbers `SDR_Dash.py` would produce.

**Steps:**

1. **Set up `sdr-api/`** with FastAPI, uvicorn, pydantic, and the sim dependencies. Pin the same Python version as `sim/` (check `sim/requirements.txt`).
2. **Read the simulation entry point.** Open `sim/model_run.py` and `sim/SDR_Dash.py`. Identify the function that takes parameters and returns results. The pattern is roughly:
   ```python
   # In SDR_Dash.py around the "Run simulation" button:
   i_flags, i_HSS, i_param = build_inputs(...)
   b_df, i_df, individual_outcomes = run_simulation(...)
   ```
   Your job: reproduce `build_inputs` cleanly from Pydantic input, call the existing `run_simulation`, and serialize the DataFrames to JSON.
3. **Define Pydantic schemas** — see `api_contract.md` for the full contract. Start with:
   - `ScenarioRequest` — what comes in
   - `ScenarioResult` — what goes out
4. **Implement `POST /scenarios/run`** — accepts a `ScenarioRequest`, calls the sim, returns `ScenarioResult`.
5. **Implement `POST /scenarios/compare`** — accepts two scenarios, returns paired results.
6. **Implement `GET /presets`** — returns the four preset configurations (Status quo, HSS, MOMISH, Combined) hardcoded against actual values from `parameters.py`. See `intervention_status.md` for the canonical preset values.
7. **Add CORS** for the frontend's dev port (3000) and production domain.
8. **Verify parity.** Run the existing Streamlit dashboard with HSS Intensive preset. Note the maternal-deaths number. Hit your new API with the same inputs. Numbers must match exactly.

**Don't:**
- Don't touch `LB_effect.py`, `intrapartum.py`, `mortality.py`, `parameters.py`. Those are the calibrated model.
- Don't try to "improve" the simulation interface. Adapt to what's there.

**Output:** `sdr-api/` running locally on port 8000, returning JSON that matches Streamlit's numbers.

---

### Phase 2 — Next.js scaffold + design system (2–3 days)

**Goal:** Stand up the Next.js project with the design tokens, typography, and shared components matching the HTML mockup. No real screens yet — just the foundation.

**Steps:**

1. **Create `sdr-web/`** with `npx create-next-app@latest` — App Router, TypeScript, Tailwind, ESLint.
2. **Copy the design tokens** from `SDR_Dashboard_Mockup.html` into `tailwind.config.ts`. The palette is defined inline in the HTML's `<script>` block at the top:
   ```ts
   colors: {
     paper: '#F6F3ED',
     'paper-deep': '#EDE7DA',
     ink: '#1C1A15',
     'ink-soft': '#4A4339',
     'ink-muted': '#7E7464',
     border: '#E2DAC8',
     'border-soft': '#EDE7DA',
     accent: '#B5471F',
     'accent-soft': '#F4E5DC',
     baseline: '#9C9082',
     intervention: '#2E5F5C',
     'intervention-soft': '#DCE9E8',
     warning: '#B68B3E',
     'warning-soft': '#F0E2C5',
     positive: '#2E5F5C',
     negative: '#A03A2A',
     card: '#FFFFFF',
   }
   ```
3. **Load fonts** (Fraunces for display, Instrument Sans for body) via `next/font/google`. Map them to Tailwind utilities `font-display` and `font-sans`.
4. **Build the TopNav component** — see `ui_components.md` for the exact spec.
5. **Build the shared primitives:** `PillSelector`, `KPITile`, `Card`, `Drawer`, `Modal`. These appear on multiple screens.
6. **Set up the API client** (`lib/api.ts`) with typed fetch wrappers against `sdr-api`.

**Output:** Empty Next.js app with the design system applied. TopNav renders correctly. Visit `/` and see the same fonts/colors as the HTML mockup.

---

### Phase 3 — Stakeholder screens (5–7 days)

**Goal:** All four primary screens (Start, About, Design & Run, Results) working end-to-end against the API.

Build in this order:

1. **About the Model (`/about`)** — Pure static content from the HTML mockup. No API calls. Get this done first; it's the simplest and lets you confirm the design system works end-to-end.
2. **Start (`/`)** — Renders the four preset cards. Clicking a preset routes to `/design?preset=hss-intensive` (or similar). Preset list comes from `GET /presets`.
3. **Design & Run (`/design`)** — The biggest single screen. See `ui_components.md` for the breakdown. Key behaviors:
   - URL-encodes the current scenario state so the page is reload-safe and shareable.
   - Three pillars (HSS, Treatments, Community) with progressive disclosure.
   - Inline Run settings panel (timeline + Quick/Robust).
   - Editable scenario name.
   - "Run simulation" button POSTs to `/scenarios/run` and navigates to `/results?run_id=...`.
4. **Results (`/results?run_id=...`)** — Reads the run result from the API. Renders the assumptions callout, KPI tiles, four stories (Cost → Mothers → Where → System), and the indicators drawer.

**Don't:**
- Don't try to make the indicator drawer fully interactive yet. Render the checkboxes — but treat "select essentials / defaults" as visual mockups until Phase 5.
- Don't build the modals (onboarding, share, county-switch) yet. They go in Phase 5.

**Output:** A user can pick a preset, see the Design screen, click Run, and land on Results with real numbers from the simulation.

---

### Phase 4 — Compare flow (3–4 days)

**Goal:** The Scenario Comparison workflow end-to-end.

1. **Compare (`/compare`)** — Build the left Intervention Library sidebar and the two scenario columns. Adding from the library mutates URL state for A or B.
2. **Compare Results (`/compare/results`)** — Side-by-side KPIs with deltas, combined narrative, overlay charts. POSTs both scenarios to `/scenarios/compare`.

**Output:** User can build A vs B scenarios, run the comparison, and view results.

---

### Phase 5 — Polish, modals, and deployment (3–4 days)

1. **Modals:** onboarding (first-visit), share (URL copy + export), county-switch confirmation.
2. **Indicator drawer interactivity:** wire the checkboxes to actually filter what's displayed in stories below.
3. **localStorage** for onboarding dismissal.
4. **Deployment to Railway** — two services in one project. See `architecture.md` for env vars and deployment config.
5. **README** in the repo root explaining how to run locally.
6. **Brief team handoff** — 30-minute walkthrough.

**Output:** Production deployment at `sdr-tool.railway.app` (or your custom domain).

---

## Important conventions

**On intervention status.** The current Streamlit code has UI toggles for some interventions whose model wiring is incomplete (FQA, PULSE, parts of Referral/EMT). The new UI shows these with a small "● UI only" badge. See `intervention_status.md` for the table. When a user changes one of these "UI only" controls, the API should accept the value, store it on the scenario, and ignore it during simulation (with a console warning). Do **not** silently drop the field.

**On preset values.** Don't hardcode preset values in the frontend. The frontend asks `/presets` and trusts the answer. The API hardcodes them against what's in `parameters.py`. This way, when the simulation team updates a calibration, the API is the single source of truth.

**On parameter naming.** The simulation uses specific dict keys (`i_HSS["P_ANC"]`, `i_flags["flag_PROMPTS"]`, etc.). The API's Pydantic models should use clean snake_case names (`p_anc`, `prompts_enabled`) and the adapter (`scenario_to_sim.py`) handles the translation. **Don't leak sim's internal naming into the API or frontend.**

**On the "Robust" run mode.** It runs 50 simulations and can take 10 minutes. The API should expose this as a long-running job — return `202 Accepted` with a `run_id`, and let the frontend poll `GET /scenarios/{run_id}` until ready. Quick runs (≤2 minutes) can stay synchronous.

**On error handling.** When the sim raises an exception, do not return a 500 with the Python traceback. Return a 422 with a sanitized error message that the frontend can display.

---

## Where to look when you're stuck

- **What does the simulation expect?** → Read `sim/SDR_Dash.py` lines around the "Run simulation" button click handler. That's where the existing UI builds the dicts.
- **What parameters mean what?** → `sim/parameters.py` has docstrings on the defaults.
- **What's the visual target?** → Open `new_dev_direction/SDR_Dashboard_Mockup.html` in a browser.
- **What's the data shape?** → `new_dev_direction/api_contract.md`.
- **What's wired in the sim, what isn't?** → `new_dev_direction/intervention_status.md`.

---

## Out of scope

Don't build any of this:

- Analyst view (the 30+ outcome browser, raw data export, custom chart builder)
- User accounts, authentication, save-to-account
- Multi-county data (the API should accept a `county` parameter but only Kakamega returns real data; others return 404)
- AI features (chat, summarization, recommendations)
- Mobile-first responsive design (the tool is for desktop policy work; basic responsive is fine but optimize for desktop)
- Internationalization

If you finish phases 1–5 and want more to do, file a ticket and ask. Don't add scope.

---

## Done definition

You're done when:

- [ ] A stakeholder can land on `/`, pick a preset, see Design, click Run, and see Results with real simulation numbers.
- [ ] They can build A vs B scenarios on `/compare` and see paired results.
- [ ] The About page renders correctly with anchor navigation.
- [ ] The county dropdown shows "coming soon" for non-Kakamega options and triggers the confirm modal.
- [ ] First-visit users see the onboarding overlay; it doesn't reappear after dismissal.
- [ ] Share button produces a copyable URL that, when opened in a new browser, loads the same scenario.
- [ ] Both services are deployed to Railway and accessible via HTTPS.
- [ ] README exists explaining how to run locally and how to deploy.
- [ ] Numbers from the new tool match the old Streamlit dashboard for the same inputs (parity test).

---

## See also

- [`architecture.md`](./architecture.md) — system diagram, deployment, env vars
- [`api_contract.md`](./api_contract.md) — full API spec with request/response shapes
- [`ui_components.md`](./ui_components.md) — per-screen component breakdown
- [`intervention_status.md`](./intervention_status.md) — which interventions are wired vs UI-only
- [`SDR_Dashboard_Mockup.html`](./SDR_Dashboard_Mockup.html) — the visual source of truth
