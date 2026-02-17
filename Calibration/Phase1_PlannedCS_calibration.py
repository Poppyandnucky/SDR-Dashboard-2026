import random
import numpy as np
import streamlit as st
import math
from parameters import get_parameters
from global_func import odds_prob

# Initialize parameters
P = {}  # dict to restore probabilities
n = {}  # dict to restore counts
E = {}  # dict to restore effects
S = {}  # dict to restore supplies and capacities
OR = {}  # dict to restore odds ratio
M = {}  # dict to restore maternal outcomes

param = get_parameters()

# Calibration criteria
P["elective_CS"] = param["elective_CS_target"]  #0.06
P["elective_CS_preterm"] = param["elective_CS_preterm_target"]  #0.265

random.seed(0)
def f_ANC_LB_effect(p_elec_CS_highrisk, p_elec_CS_preterm):
    # Initialize parameters
    LB_tot = np.array([23729, 18196, 20709, 5126])
    P["highrisk_all"] = 0.26  # high risk pregnancies among all live births
    P["ANC"] = 0.56
    # Six parameters - calibrated using beyesian optimization
    P_home_noANC, P_l45_fac, P_home_lowrisk, P_L23_highrisk, Sen_traditional, Spec_traditional = \
        0.706, 0.11, 0.1692, 0.3255, 0.7641, 0.6306 #(newly calibrated - also works)
        #0.7056735167353498, 0.21039299793502853, 0.17051451153731043, 0.36751767996993084, 0.7939486174518111, 0.6311811663318615


    GA = param['GA_sequence']
    GA_n = len(GA)
    n["GA"] = param["GA_distribution"]
    P["GA"] = n["GA"] / np.sum(n["GA"])
    PT_mask = np.array([1] * 10 + [0] * 8, dtype=bool)
    FT_mask = ~PT_mask
    P["GA"][PT_mask] = P["GA"][PT_mask] * param["PT_scale"]             # scale P[GA|PT] to match kenya level
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

    E["Preterm_LMP"] = np.zeros(3)
    E["Preterm_LMP"][0], E["Preterm_LMP"][1] = param['E_Preterm_LMP'][0], param['E_Preterm_LMP'][1]
    E["Preterm_LMP"][2] = 1 - (E["Preterm_LMP"][0] + E["Preterm_LMP"][1])
    E["Postterm_LMP"] = np.zeros(3)
    E["Postterm_LMP"][0], E["Postterm_LMP"][1] = param['E_Postterm_LMP'][0], param['E_Postterm_LMP'][1]
    E["Postterm_LMP"][2] = 1 - (E["Postterm_LMP"][0] + E["Postterm_LMP"][1])

    P_fac_noANC = 1 - P_home_noANC
    P_L45_noANC = P_fac_noANC * P_l45_fac
    P_L23_noANC = P_fac_noANC - P_L45_noANC

    # Initialize counters
    n["LB_L"] = np.zeros(4)                                               # number of delivery location by facility levels
    n["elective_CS"] = np.zeros(4)                                         #number of elective CS by facility levels
    n["PT"] = np.zeros(4)                                                 # number of preterm deliveries by facility levels

    for k_LB in range(np.sum(LB_tot)):
        i_risk = np.random.binomial(1, P["highrisk_all"])
        i_ANC = np.random.binomial(1, P["ANC"])
        i_elec_CS = 0
        i_risk_pred = 0
        i_preterm_pred = 0
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
            i_preterm_pred = 1 if np.random.random() < E["Preterm_LMP"][i_term_status] else 0
            i_postterm_pred = 1 if np.random.random() < E["Postterm_LMP"][i_term_status] else 0

            # Normal referrals
            if i_risk_pred or i_preterm_pred or i_postterm_pred:
                i_loc = np.random.choice([0, 1, 2], p=[0, P_L23_highrisk, 1 - P_L23_highrisk])
            else:
                i_loc = np.random.choice([0, 1, 2], p=[P_home_lowrisk, (1 - P_home_lowrisk) * 0.89, (1 - P_home_lowrisk) * 0.11])

            # elective CS
            if i_loc > 1 and (
                    i_risk_pred or i_preterm_pred):  # elective c-section often planned for reducing post-term risk, so it often happens before 39 weeks
                if i_risk_pred:
                    i_elec_CS = np.random.binomial(1, p_elec_CS_highrisk)
                elif i_preterm_pred:
                    i_elec_CS = np.random.binomial(1, p_elec_CS_preterm)

        # assume among all live births at L45, 5126/(20709 + 5126) going to L5
        P_l5_l45 = 5126 / (20709 + 5126)
        if i_loc == 2 and np.random.random() < P_l5_l45:
            i_loc = 3

        # restore results
        n["LB_L"][i_loc]     += 1
        n["PT"][i_loc] += i_term_status == 0
        n["elective_CS"][i_loc] += i_elec_CS

    # Predicted outcomes
    P_elec_CS = np.sum(n["elective_CS"][1:]) / np.sum(n["LB_L"][1:])
    P_elec_CS_PT = np.sum(n["elective_CS"][1:]) / np.sum(n["PT"])

    return P_elec_CS, P_elec_CS_PT

#Test the function for 10 runs and taking average
p_elec_CS_highrisk, p_elec_CS_preterm = 0.23741230625326665, 0.00025119198807865844
results = [f_ANC_LB_effect(p_elec_CS_highrisk, p_elec_CS_preterm) for _ in range(10)]

P_elec_CS = np.mean([x[0] for x in results])
P_elec_CS_preterm = np.mean([x[1] for x in results])
print("Predicted outcomes:", P_elec_CS, P_elec_CS_preterm)

# # ##Calibration
# def calibration_loss(params):
#     p_elec_CS_highrisk, p_elec_CS_preterm = params
#
#     print("Testing params:", params)  # Debug print
#
#     results = [f_ANC_LB_effect(p_elec_CS_highrisk, p_elec_CS_preterm) for _ in range(10)]
#
#     P_elec_CS, P_elec_CS_preterm = np.mean([x[0] for x in results]), np.mean([x[1] for x in results])
#
#     loss = math.sqrt((P_elec_CS - P["elective_CS"])**2 + (P_elec_CS_preterm - P["elective_CS_preterm"])**2)
#     print("Loss:", loss)  # Debug print
#     return loss
#
# from skopt import gp_minimize
# from skopt.space import Real
# import numpy as np
#
# # Define the parameter space
# param_space = [
#     Real(0, 0.5, name='p_elec_CS_highrisk'),
#     Real(0, 0.5, name='p_elec_CS_preterm')
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