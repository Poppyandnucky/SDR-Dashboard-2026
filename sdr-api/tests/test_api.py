import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.data.presets import HSS_INTENSIVE, STATUS_QUO

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_presets():
    r = client.get("/api/v1/presets")
    assert r.status_code == 200
    presets = r.json()["presets"]
    assert len(presets) == 4
    assert all(not p["is_recommended"] for p in presets)


def test_meta_counties():
    r = client.get("/api/v1/meta/counties")
    assert r.status_code == 200
    kak = [c for c in r.json()["counties"] if c["id"] == "kakamega"][0]
    assert kak["calibrated"] is True


@pytest.mark.slow
def test_run_status_quo_quick():
    scenario = STATUS_QUO.model_copy(deep=True)
    scenario.run.mode = "quick"
    r = client.post("/api/v1/scenarios/run", json=scenario.model_dump())
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "complete"
    assert "run_id" in data
    assert data["result"]["summary"]["maternal_deaths_averted"] >= 0
    assert "indicator_series" in data["result"]["timeseries"]
    assert len(data["result"]["timeseries"]["indicator_series"]["anc_rate_per_100_lb"]["baseline"]) > 0


@pytest.mark.slow
def test_run_hss_intensive():
    scenario = HSS_INTENSIVE.model_copy(deep=True)
    scenario.run.mode = "quick"
    r = client.post("/api/v1/scenarios/run", json=scenario.model_dump())
    assert r.status_code == 200
    result = r.json()["result"]
    assert result["summary"]["dalys_averted"] >= 0


@pytest.mark.slow
def test_hss_beats_status_quo():
    """HSS Intensive should avert more deaths than status quo baseline comparison."""
    sq = STATUS_QUO.model_copy(deep=True)
    sq.run.mode = "quick"
    hss = HSS_INTENSIVE.model_copy(deep=True)
    hss.run.mode = "quick"

    r_sq = client.post("/api/v1/scenarios/run", json=sq.model_dump())
    r_hss = client.post("/api/v1/scenarios/run", json=hss.model_dump())
    assert r_sq.status_code == 200 and r_hss.status_code == 200
    hss_deaths = r_hss.json()["result"]["summary"]["maternal_deaths_averted"]
    sq_deaths = r_sq.json()["result"]["summary"]["maternal_deaths_averted"]
    assert hss_deaths >= sq_deaths
