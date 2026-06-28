import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure sim/ is on PYTHONPATH before any sim imports
SIM_PATH = Path(__file__).resolve().parents[2] / "sim"
if str(SIM_PATH) not in sys.path:
    sys.path.insert(0, str(SIM_PATH))

from app.routes.health import router as health_router
from app.routes.meta import router as meta_router
from app.routes.scenarios import router as scenarios_router

app = FastAPI(
    title="SDR Decision Tool API",
    description="FastAPI wrapper around the Kakamega maternal health simulation",
    version="1.0.0",
)

origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(meta_router)
app.include_router(scenarios_router)
