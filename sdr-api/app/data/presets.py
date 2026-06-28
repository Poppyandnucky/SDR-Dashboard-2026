from copy import deepcopy

from app.schemas.scenario import ScenarioRequest

STATUS_QUO = ScenarioRequest(
    name="Status quo",
    hss={"enabled": False, "intensity": "off"},
    treatments={"enabled": False},
    community={"enabled": False},
)

HSS_INTENSIVE = ScenarioRequest(
    name="Health system strengthening",
    hss={"enabled": True, "intensity": "intensive"},
    treatments={"enabled": False},
    community={"enabled": False},
)

MOMISH = ScenarioRequest(
    name="Community engagement (MOMISH)",
    hss={"enabled": False, "intensity": "off"},
    treatments={"enabled": False},
    community={
        "enabled": True,
        "prompts": {
            "enabled": True,
            "adoption": 1.0,
            "chv_engagement": 1.0,
            "intervention_fidelity": 0.87,
        },
        "mentors": {
            "enabled": True,
            "adoption": 0.8,
            "attendance": 0.8,
            "fidelity": 0.8,
        },
    },
)

COMBINED = ScenarioRequest(
    name="Combined strategy",
    hss={"enabled": True, "intensity": "moderate"},
    treatments={
        "enabled": True,
        "pph_bundle": True,
        "mgso4": True,
    },
    community={
        "enabled": True,
        "prompts": {
            "enabled": True,
            "adoption": 0.6,
            "chv_engagement": 0.6,
            "intervention_fidelity": 0.75,
        },
    },
)

PRESETS = [
    {
        "id": "status-quo",
        "name": "Status quo",
        "subtitle": "Baseline only",
        "description": "Today's conditions projected forward.",
        "scenario": STATUS_QUO,
        "is_recommended": False,
    },
    {
        "id": "hss-intensive",
        "name": "Health system strengthening",
        "subtitle": "HSS · Intensive",
        "description": "Aggressive demand-side investment with matched supply.",
        "scenario": HSS_INTENSIVE,
        "is_recommended": False,
    },
    {
        "id": "momish",
        "name": "Community engagement (MOMISH)",
        "subtitle": "PROMPTS · Full · CHVs strong",
        "description": "PROMPTS full rollout with strong CHV engagement.",
        "scenario": MOMISH,
        "is_recommended": False,
    },
    {
        "id": "combined",
        "name": "Combined strategy",
        "subtitle": "HSS · Treatments · MOMISH",
        "description": "HSS Moderate + key treatments + partial MOMISH.",
        "scenario": COMBINED,
        "is_recommended": False,
    },
]


def get_presets_response() -> dict:
    return {
        "presets": [
            {
                "id": p["id"],
                "name": p["name"],
                "subtitle": p["subtitle"],
                "description": p["description"],
                "is_recommended": p.get("is_recommended", False),
                "scenario": p["scenario"].model_dump(),
            }
            for p in PRESETS
        ]
    }


def get_preset_scenario(preset_id: str) -> ScenarioRequest | None:
    for p in PRESETS:
        if p["id"] == preset_id:
            return deepcopy(p["scenario"])
    return None
