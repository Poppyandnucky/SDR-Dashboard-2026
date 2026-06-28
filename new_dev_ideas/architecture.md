# Architecture

## System diagram

```
┌─────────────────────────────────────────────────────────┐
│                       Stakeholder                        │
│                    (browser, desktop)                    │
└────────────────────────┬────────────────────────────────┘
                         │ HTTPS
                         ▼
┌─────────────────────────────────────────────────────────┐
│  sdr-web  (Next.js · TypeScript · Tailwind)             │
│  - App Router pages: /, /design, /compare, /results      │
│  - Renders the v0.7.1 mockup design                      │
│  - Encodes scenario state in URL                         │
│  - Calls sdr-api over HTTPS                              │
│  Deployed: Railway · sdr-tool.railway.app                │
└────────────────────────┬────────────────────────────────┘
                         │ JSON over HTTPS
                         ▼
┌─────────────────────────────────────────────────────────┐
│  sdr-api  (FastAPI · Python 3.11)                        │
│  - Endpoints: /presets, /scenarios/run, /scenarios/...   │
│  - Adapters between API schemas and sim's internal dicts │
│  - In-memory cache for run results (15 min TTL)          │
│  - Async run queue for Robust mode (background tasks)    │
│  Deployed: Railway · sdr-api.railway.app                 │
└────────────────────────┬────────────────────────────────┘
                         │ Python function calls
                         ▼
┌─────────────────────────────────────────────────────────┐
│  sim/  (Existing — DO NOT MODIFY)                        │
│  - model_run.py · LB_effect.py · intrapartum.py          │
│  - parameters.py · mortality.py · global_func.py         │
│  - Calibrated ABM for Kakamega County                    │
└─────────────────────────────────────────────────────────┘
```

## Data flow — a single scenario run

```
User clicks "Run simulation" on /design
         │
         ▼
Next.js builds Scenario object from current form state
         │
         ▼
POST /scenarios/run  { Scenario JSON }
         │
         ▼
FastAPI: ScenarioRequest (Pydantic validation)
         │
         ▼
adapters/scenario_to_sim.py:
   Scenario → (i_flags dict, i_HSS dict, i_param dict)
         │
         ▼
sim.model_run.run_simulation(i_flags, i_HSS, i_param, ...)
         │
         ▼
Returns: baseline_df, intervention_df, individual_outcomes
         │
         ▼
adapters/sim_to_response.py:
   DataFrames → ScenarioResult Pydantic model
         │
         ▼
Response: 200 OK  { ScenarioResult JSON, run_id }
         │
         ▼
Next.js stores run_id, navigates to /results?run_id=...
         │
         ▼
/results fetches GET /scenarios/{run_id} and renders
```

## Data flow — Robust mode (50 simulations, ~10 min)

```
POST /scenarios/run  { Scenario JSON, mode: "robust" }
         │
         ▼
FastAPI generates run_id, enqueues background task
         │
         ▼
Response: 202 Accepted  { run_id, status: "pending" }
         │
         ▼
Next.js polls GET /scenarios/{run_id} every 5s
         │
         ▼
Once status === "complete", fetches result
```

Implement the background task with `BackgroundTasks` from FastAPI plus a simple in-memory dict for status tracking. No need for Celery/Redis at this scale — a single Railway container handles it fine.

## Repo and branch strategy

- **One monorepo** with `sim/`, `sdr-api/`, `sdr-web/`, `new_dev_direction/` at the top level.
- **Branches:** `main` (auto-deploys to Railway prod), `dev` (auto-deploys to Railway staging if you set one up), feature branches off `dev`.
- **Don't merge to main until parity test passes** — the new API's numbers must match the old Streamlit dashboard's numbers for the HSS Intensive preset.

## Deployment — Railway

### Setting up the project

1. Create a Railway project (`sdr-decision-tool`).
2. Add two services from the same GitHub repo:
   - **`sdr-api`** — Root directory `sdr-api/`, Dockerfile build.
   - **`sdr-web`** — Root directory `sdr-web/`, auto-detected Next.js.
3. Set environment variables (see below).
4. Connect a custom domain if needed.

### `sdr-api` environment variables

```env
# Required
PYTHON_VERSION=3.11
PORT=8000
ALLOWED_ORIGINS=https://sdr-tool.railway.app,http://localhost:3000

# Optional
LOG_LEVEL=info
RUN_CACHE_TTL_SECONDS=900    # 15 min
ROBUST_RUN_TIMEOUT_SECONDS=900
```

### `sdr-api` Dockerfile

```dockerfile
FROM python:3.11-slim

# Install sim dependencies (numpy, pandas, etc.)
COPY sim/requirements.txt /tmp/sim-requirements.txt
COPY sdr-api/requirements.txt /tmp/api-requirements.txt
RUN pip install --no-cache-dir -r /tmp/sim-requirements.txt -r /tmp/api-requirements.txt

# Copy sim (read-only) and api code
COPY sim/ /app/sim/
COPY sdr-api/ /app/sdr-api/

WORKDIR /app
ENV PYTHONPATH=/app

CMD ["uvicorn", "sdr-api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

The sim is imported as a Python module. Make sure `sim/__init__.py` exists (create an empty one if not).

### `sdr-web` environment variables

```env
NEXT_PUBLIC_API_BASE=https://sdr-api.railway.app
```

That's the only one. The frontend builds against the API base URL at build time.

### `sdr-web` build command

Railway auto-detects Next.js. Default build (`npm run build`) and start (`npm start`) commands work.

## Local development

In two terminals:

```bash
# Terminal 1 — API
cd sdr-api
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Web
cd sdr-web
npm install
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

Open `http://localhost:3000`.

## Performance budget

| Operation | Target | Mechanism |
|---|---|---|
| Page load (any screen) | < 1.5 s | Next.js SSR + static where possible |
| `POST /scenarios/run` (Quick) | < 90 s | Synchronous; warn user > 30s |
| `POST /scenarios/run` (Robust) | < 10 min | Async with polling |
| `POST /scenarios/compare` | < 3 min | Two parallel Quick runs |
| `GET /presets` | < 100 ms | Static dict in memory |

If the Quick simulation takes longer than 90s, that's a sim issue — flag it back to the simulation team, don't try to speed up the API.

## Monitoring

Railway's built-in logs are fine for v1. Watch for:

- 5xx errors on `/scenarios/run` — usually means a sim crash, look at the Python traceback in logs
- Timeouts on Robust mode — may indicate sim performance regression
- 4xx spikes — frontend is sending malformed scenarios; check the adapter

For v1 we don't need Sentry/Datadog. Add if you start handling real user traffic.

## Caching strategy

- **In-memory cache** in `sdr-api` for run results keyed by `run_id`. TTL 15 minutes.
- **No caching of `/presets`** at the API level — they're already static dicts, called rarely.
- **Browser caching** on the frontend for static assets (Next.js handles this).
- **No CDN** for v1.

When the cache evicts a run, the URL `/results?run_id=...` 404s. Frontend should detect this and offer a "re-run scenario" CTA.

## Security notes

- **CORS** restricted to the production frontend domain + localhost for dev.
- **No authentication** in v1. The tool is public; numbers are not sensitive.
- **No PII** is stored. Scenarios only contain configuration values.
- **Rate limiting** — implement a 10-runs-per-minute-per-IP cap on the API to prevent abuse, since each run consumes CPU.

## Cost estimate

Railway billing, rough monthly cost at expected stakeholder traffic (~50 daily active users):

- `sdr-api` container (1 CPU, 1 GB RAM): **~$10–15/mo**
- `sdr-web` container (0.5 CPU, 512 MB RAM): **~$5–8/mo**
- Total: **~$15–25/mo**

If Robust runs become heavy, the API container may need bumping to 2 CPU. Don't over-provision early.
