# Kenya Maternal Health Decision Tool (SDR Dashboard)

Agent-based maternal health simulation for Kakamega County, Kenya — with a modern Next.js UI and FastAPI wrapper around the calibrated Python model.

## Architecture

```
sdr-web (Next.js)  →  sdr-api (FastAPI)  →  sim/ (Python ABM)
```

Legacy Streamlit dashboard remains in `sim/SDR_Dash.py` for reference and parity testing.

## Local development

### Legacy Streamlit

```bash
pip install -r sim/requirements.txt
streamlit run sim/SDR_Dash.py
```

Open http://localhost:8501

### New stack (recommended)

**Terminal 1 — API**

```bash
cd sdr-api
pip install -r requirements.txt
PYTHONPATH=../sim uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — Web**

```bash
cd sdr-web
npm install
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

Open http://localhost:3000

### Docker Compose

```bash
docker compose up --build
```

## Railway deployment

1. Create a Railway project and connect this GitHub repo.
2. Add **sdr-api** service:
   - Dockerfile path: `sdr-api/Dockerfile`
   - Build context: repository root
   - Env: `ALLOWED_ORIGINS=https://<web-domain>,http://localhost:3000`
3. Add **sdr-web** service:
   - Root directory: `sdr-web`
   - Env: `NEXT_PUBLIC_API_BASE=https://<api-domain>`
4. Redeploy web after API URL is known (Next.js bakes `NEXT_PUBLIC_*` at build time).

See [new_dev_ideas/architecture.md](new_dev_ideas/architecture.md) for full deployment details.

## Tests

```bash
cd sdr-api
PYTHONPATH=../sim pytest tests/ -v
```

Parity tests compare API output against Streamlit-equivalent inputs.

## Project layout

| Path | Purpose |
|------|---------|
| `sim/` | Python simulation (read-only core) + legacy Streamlit |
| `sdr-api/` | FastAPI wrapper |
| `sdr-web/` | Next.js frontend |
| `new_dev_ideas/` | Design docs and HTML mockup |
| `docs/` | Stakeholder materials |
