"""Run simulation with same logic as SDR_Dash.py single-run mode."""

from __future__ import annotations

import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np

from app.adapters.scenario_to_sim import scenario_to_sim_inputs, sync_param_momish_from_hss
from app.adapters.sim_to_response import sim_outputs_to_response
from app.schemas.results import ScenarioResult
from app.schemas.scenario import ScenarioRequest

SIM_PATH = Path(__file__).resolve().parents[3] / "sim"
if str(SIM_PATH) not in sys.path:
    sys.path.insert(0, str(SIM_PATH))


def _run_single(
    req: ScenarioRequest,
    base_seed: Optional[int] = None,
) -> tuple[ScenarioResult, dict]:
    from global_func import reset_E, reset_HSS, reset_S, reset_flags
    from model_run import run_model_dash
    from parameters import calculate_derived_parameters, get_parameters, get_slider_params

    i_flags, i_e, i_s, i_hss, warnings = scenario_to_sim_inputs(req)
    slider_params = get_slider_params()

    n_months = req.run.implementation_years * 12 + req.run.maintenance_years * 12
    int_period = req.run.implementation_years * 12

    if base_seed is None:
        base_seed = int(np.random.default_rng().integers(0, 1_000_000))

    master_rng = np.random.default_rng(base_seed)
    base_seeds = master_rng.integers(low=0, high=1_000_000, size=n_months)

    b_param = get_parameters(rng=np.random.default_rng(base_seed))
    b_param = calculate_derived_parameters(b_param)
    i_param = get_parameters(rng=np.random.default_rng(base_seed))
    i_param = calculate_derived_parameters(i_param)

    b_flags = reset_flags()
    b_hss = reset_HSS(slider_params)
    b_s = reset_S(slider_params)
    b_e = reset_E()
    b_param.update({"E": b_e, "S": b_s, "HSS": b_hss})

    i_param.update({"E": i_e, "S": i_s, "HSS": i_hss})
    sync_param_momish_from_hss(i_param, i_hss)

    start = time.time()
    b_df, b_ind, _ = run_model_dash(b_param, b_flags, n_months, int_period, base_seed=base_seeds)
    i_df, i_ind, _ = run_model_dash(i_param, i_flags, n_months, int_period, base_seed=base_seeds)
    runtime = time.time() - start

    result = sim_outputs_to_response(
        b_df=b_df,
        i_df=i_df,
        b_ind=b_ind,
        i_ind=i_ind,
        scenario=req,
        i_flags=i_flags,
        i_param=i_param,
        n_months=n_months,
        int_period=int_period,
        runtime_seconds=runtime,
        n_runs=1,
        seed=base_seed,
        warnings=warnings,
    )
    return result, {"i_flags": i_flags, "i_param": i_param}


def run_scenario(req: ScenarioRequest, base_seed: Optional[int] = None) -> ScenarioResult:
    if req.run.mode == "quick":
        result, _ = _run_single(req, base_seed=base_seed)
        return result

    n_robust = int(os.environ.get("ROBUST_RUN_COUNT", "10"))
    results = []
    b_dfs, i_dfs = [], []
    master = np.random.default_rng(base_seed or 2025)
    seeds = master.integers(0, 1_000_000, size=n_robust)

    from global_func import reset_E, reset_HSS, reset_S, reset_flags
    from model_run import run_model_dash
    from parameters import calculate_derived_parameters, get_parameters, get_slider_params

    i_flags, i_e, i_s, i_hss, warnings = scenario_to_sim_inputs(req)
    slider_params = get_slider_params()
    n_months = req.run.implementation_years * 12 + req.run.maintenance_years * 12
    int_period = req.run.implementation_years * 12

    start = time.time()
    b_df_last, i_df_last, b_ind_last, i_ind_last = None, None, None, None
    i_param = None

    for seed in seeds:
        master_rng = np.random.default_rng(int(seed))
        base_seeds = master_rng.integers(low=0, high=1_000_000, size=n_months)

        b_param = get_parameters(rng=np.random.default_rng(int(seed)))
        b_param = calculate_derived_parameters(b_param)
        i_param = get_parameters(rng=np.random.default_rng(int(seed)))
        i_param = calculate_derived_parameters(i_param)

        b_flags = reset_flags()
        b_hss = reset_HSS(slider_params)
        b_s = reset_S(slider_params)
        b_e = reset_E()
        b_param.update({"E": b_e, "S": b_s, "HSS": b_hss})
        i_param.update({"E": deepcopy(i_e), "S": deepcopy(i_s), "HSS": deepcopy(i_hss)})
        sync_param_momish_from_hss(i_param, i_hss)

        b_df, b_ind, _ = run_model_dash(b_param, b_flags, n_months, int_period, base_seed=base_seeds)
        i_df, i_ind, _ = run_model_dash(i_param, i_flags, n_months, int_period, base_seed=base_seeds)
        b_dfs.append(b_df)
        i_dfs.append(i_df)
        b_df_last, i_df_last, b_ind_last, i_ind_last = b_df, i_df, b_ind, i_ind

    runtime = time.time() - start
    return sim_outputs_to_response(
        b_df=b_df_last,
        i_df=i_df_last,
        b_ind=b_ind_last,
        i_ind=i_ind_last,
        scenario=req,
        i_flags=i_flags,
        i_param=i_param,
        n_months=n_months,
        int_period=int_period,
        runtime_seconds=runtime,
        n_runs=n_robust,
        seed=int(seeds[0]) if len(seeds) else None,
        warnings=warnings,
        b_df_runs=b_dfs,
        i_df_runs=i_dfs,
    )
