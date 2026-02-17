import random
import numpy as np
import streamlit as st
import math

from numba.core.typing.builtins import Print

from parameters import get_parameters
from global_func import odds_prob

# Initialize parameters
P = {}  # dict to restore probabilities
n = {}  # dict to restore counts
E = {}  # dict to restore effects
S = {}  # dict to restore supplies and capacities
OR = {}  # dict to restore odds ratio
M = {}  # dict to restore maternal outcomes

# Calibration criteria
param = get_parameters()
P["preterm_fac"] = param["preterm_fac_target"] #0.134

random.seed(0)
def f_ANC_LB_effect(PT_scale):
    # Initialize parameters
    LB_tot = np.array([23729, 18196, 20709, 5126])
    P["highrisk_all"] = 0.26  # high risk pregnancies among all live births
    P["ANC"] = 0.56
    # Six parameters - calibrated using beyesian optimization
    P_home_noANC, P_l45_fac, P_home_lowrisk, P_L23_highrisk, Sen_traditional, Spec_traditional = \
        0.706, 0.11, 0.1692, 0.3255, 0.7641, 0.6306 #(newly calibrated)
        #0.7056735167353498, 0.21039299793502853, 0.17051451153731043, 0.36751767996993084, 0.7939486174518111, 0.6311811663318615

    GA = param['GA_sequence']
    GA_n = len(GA)
    n["GA"] = param["GA_distribution"]
    P["GA"] = n["GA"] / np.sum(n["GA"])
    PT_mask = np.array([1] * 10 + [0] * 8, dtype=bool)
    FT_mask = ~PT_mask
    P["GA"][PT_mask] = P["GA"][PT_mask] * PT_scale  # scale P[GA|PT] to match kenya level
    P_mult = (1 - np.sum(P["GA"][PT_mask])) / np.sum(P["GA"][FT_mask])
    P["GA"][FT_mask] *= P_mult

    OR["ANC"] = param["OR_preterm_ANC"]
    Preterm_rate = np.sum(P["GA"][PT_mask])  # overall preterm rate

    preterm_anc_noanc = odds_prob(OR["ANC"], Preterm_rate, P["ANC"])
    P["GA_anc"] = P["GA"].copy()
    P_scale_anc = preterm_anc_noanc[0] / Preterm_rate
    P["GA_anc"][PT_mask] *= P_scale_anc
    P_mult = (1 - preterm_anc_noanc[0]) / np.sum(P["GA_anc"][FT_mask])
    P["GA_anc"][FT_mask] *= P_mult

    P["GA_noanc"] = P["GA"].copy()
    P_scale_noanc = preterm_anc_noanc[1] / Preterm_rate
    P["GA_noanc"][PT_mask] *= P_scale_noanc
    P_mult = (1 - preterm_anc_noanc[1]) / np.sum(P["GA_noanc"][FT_mask])
    P["GA_noanc"][FT_mask] *= P_mult

    #E["Preterm_LMP"] = param["E_Preterm_LMP"]
    #E["Postterm_LMP"] = param["E_Postterm_LMP"]
    E_Preterm_LMP = np.zeros(3)
    E_Preterm_LMP[0], E_Preterm_LMP[1] = param['E_Preterm_LMP'][0], param['E_Preterm_LMP'][1]
    E_Preterm_LMP[2] = 1 - (E_Preterm_LMP[0] + E_Preterm_LMP[1])
    E_Postterm_LMP = np.zeros(3)
    E_Postterm_LMP[0], E_Postterm_LMP[1] = param['E_Postterm_LMP'][0], param['E_Postterm_LMP'][1]
    E_Postterm_LMP[2] = 1 - (E_Postterm_LMP[0] + E_Postterm_LMP[1])

    P_fac_noANC = 1 - P_home_noANC
    P_L45_noANC = P_fac_noANC * P_l45_fac
    P_L23_noANC = P_fac_noANC - P_L45_noANC

    # Initialize counters
    n["LB_L"] = np.zeros(4)                                               # number of delivery location by facility levels
    M["PT"] = np.zeros(4)                                                 # number of preterm deliveries by facility levels

    for k_LB in range(np.sum(LB_tot)):
        i_risk = np.random.binomial(1, P["highrisk_all"])
        i_ANC = np.random.binomial(1, P["ANC"])
        if i_ANC:
            P_GA = P["GA_anc"]
        else:
            P_GA = P["GA_noanc"]
        i_jGA = np.searchsorted(np.cumsum(P_GA), np.random.rand())  # % actual index of GA
        i_GA = GA[i_jGA]  # % actual GA

        if i_GA < 37:
            i_term_status = 0   #% 0 = preterm, 1 = full term, 2 = postterm
        elif i_GA >= 43:
            i_term_status = 2
        else:
            i_term_status = 1

        if i_ANC == 0:
            i_loc = np.random.choice([0, 1, 2], p=[P_home_noANC, P_L23_noANC, P_L45_noANC])
        else:
            # Risk stratification
            if i_risk:
                i_risk_pred = 1 if np.random.random() < Sen_traditional else 0
            else:
                i_risk_pred = 0 if np.random.random() < Spec_traditional else 1

            # Gestational age estimation
            i_preterm_pred = 1 if np.random.random() < E_Preterm_LMP[i_term_status] else 0
            i_postterm_pred = 1 if np.random.random() < E_Postterm_LMP[i_term_status] else 0

            if i_risk_pred or i_preterm_pred == 1 or i_postterm_pred == 1:
                i_loc = np.random.choice([0, 1, 2], p=[0, P_L23_highrisk, 1 - P_L23_highrisk])
            else:
                i_loc = np.random.choice([0, 1, 2], p=[P_home_lowrisk, (1 - P_home_lowrisk) * 0.89, (1 - P_home_lowrisk) * 0.11])

        # assume among all live births at L45, 5126/(20709 + 5126) going to L5
        P_l5_l45 = 5126 / (20709 + 5126)
        if i_loc == 2 and np.random.random() < P_l5_l45:
            i_loc = 3

        # restore results
        n["LB_L"][i_loc]     += 1
        M["PT"][i_loc] += i_term_status == 0

    # Predicted outcomes
    Preterm_fac = np.sum(M["PT"][1:]) / np.sum(n["LB_L"][1:])

    return Preterm_fac

#Test the function for 10 runs and taking average
PT_scale = 0.8250540888309176
results = [f_ANC_LB_effect(PT_scale) for _ in range(10)]

Preterm_fac_mean = np.mean(results)
print("Predicted outcomes:", Preterm_fac_mean)  # Debug print


# ##Calibrate pre-transfer live births
# def calibration_loss(params):
#     PT_scale = params
#
#     print("Testing params:", params)  # Debug print
#
#     results = [f_ANC_LB_effect(PT_scale) for _ in range(10)]
#
#     Preterm_fac = np.mean(results)
#     print("Predicted outcomes:", Preterm_fac)  # Debug print
#
#     loss = abs(Preterm_fac - P["preterm_fac"])
#     print("Loss:", loss)  # Debug print
#     return loss
#
# from skopt import gp_minimize
# from skopt.space import Real
# import numpy as np
# import math
#
# # Define the parameter space
# param_space = [
#     Real(0.8, 0.9, name='PT_scale')
# ]
#
# # Bayesian Optimization function
# def optimize_model():
#     result = gp_minimize(
#         func=calibration_loss,       # Objective function
#         dimensions=param_space,      # Parameter space
#         acq_func='EI',               # Acquisition function (Expected Improvement)
#         n_calls=100,                 # Number of iterations
#         n_random_starts=10,          # Initial random samples
#         random_state=42              # For reproducibility
#     )
#
#     print("Best Parameters:", result.x)
#     print("Best Loss:", result.fun)
#     return result
#
# # Run the optimization
# if __name__ == "__main__":
#     optimize_model()

