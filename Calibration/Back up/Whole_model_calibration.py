import numpy as np
import pandas as pd
import altair as alt
import math
import scipy.stats as stats
import time
import matplotlib.pyplot as plt
import seaborn as sns
import cProfile
import pstats
import io
from parameters import get_parameters, get_slider_params
from model_run import run_model_dash
from global_func import reset_flags, reset_E, reset_HSS, reset_S, get_P_l45, odds_prob, sample_from_ci

MODEL = {
    "imple_time": 3,
    "main_time": 0,
    "int_period": 0,
    "n_months": 36,
    "multiple_run": False,
    "n_runs": 1,
}
slider_params = get_slider_params()
base_seed = np.random.default_rng(2025).integers(low=0, high=1e6, size=1)[0]
rng_param = np.random.default_rng(base_seed)
b_param = get_parameters(rng = rng_param)
b_flags = reset_flags()
b_HSS = reset_HSS(slider_params)
b_S = reset_S(slider_params)
b_E = reset_E()
b_param.update({"E": b_E, "S": b_S, "HSS": b_HSS
})
n_months = MODEL["n_months"]
int_period = MODEL["int_period"]
b_df, b_ind_outcomes = run_model_dash(b_param, b_flags, n_months, int_period, base_seed = base_seed)
b_ind_outcomes["Run"] = 1
b_ind_outcomes["Scenario"] = "Baseline"

print(b_df)