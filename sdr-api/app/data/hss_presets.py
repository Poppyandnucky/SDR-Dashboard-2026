"""Map API intensity buckets to Streamlit demand/supply presets."""

DEMAND_SCENARIOS = {
    "light": {"P_ANC": 70, "P_L45": 53, "key": "Conservative"},
    "moderate": {"P_ANC": 80, "P_L45": 68, "key": "Moderate"},
    "intensive": {"P_ANC": 90, "P_L45": 90, "key": "Aggressive"},
}

CAPACITY_MATCH = {
    "Conservative": 25.0,
    "Moderate": 50.0,
    "Aggressive": 85.0,
}

INTENSITY_LABELS = {
    "off": "Off",
    "light": "Light",
    "moderate": "Moderate",
    "intensive": "Intensive",
}
