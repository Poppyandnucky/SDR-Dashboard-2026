import random
import numpy as np
from parameters import get_parameters
from global_func import odds_prob
import streamlit as st

# Initialize parameters
P = {}  # dict to restore probabilities
n = {}  # dict to restore counts
E = {}  # dict to restore effects
S = {}  # dict to restore supplies and capacities
OR = {}  # dict to restore odds ratio
M = {}  # dict to restore maternal outcomes

#known parameters
LB_tot = np.array([23729, 18196, 20709, 5126])
P["highrisk_all"] = 0.26                                              # high risk pregnancies among all live births
P["ANC"] = 0.56

param = get_parameters()
P["GA_anc"], P["GA_noanc"] = param["GA_anc"], param["GA_noanc"]

E["Preterm_LMP"] = param["E_Preterm_LMP"]
E["Postterm_LMP"] = param["E_Posterm_LMP"]

#Six parameters - calibrated using simulating annealing
P_home_noANC, P_l45_fac, P_home_lowrisk, P_L23_highrisk, Sen_traditional, Spec_traditional = \
0.7056735167353498, 0.21039299793502853, 0.17051451153731043, 0.36751767996993084, 0.7939486174518111, 0.6311811663318615

random.seed(10)
def f_ANC_LB_slider(P_ANC_slider):
    P_fac_noANC = 1 - P_home_noANC
    P_L45_noANC = P_fac_noANC * P_l45_fac
    P_L23_noANC = P_fac_noANC - P_L45_noANC

    # Initialize counters
    n["highrisk"] = np.zeros(4)  # number of high-risk pregnancies by facility levels
    n["ANC"] = np.zeros(4)  # number of 4+ANC by facility levels
    n["LB_L"] = np.zeros(4)  # number of delivery location by facility levels
    M["PT"] = np.zeros(4)
    M["Postterm"] = np.zeros(4)
    M["GA"] = np.zeros((4, len(param['GA_sequence'])))  # GA distributions by facility levels

    for k_LB in range(np.sum(LB_tot)):
        i_risk = np.random.binomial(1, P["highrisk_all"])
        i_ANC = np.random.binomial(1, P_ANC_slider)
        if i_ANC:
            P_GA = P["GA_anc"]
        else:
            P_GA = P["GA_noanc"]
        i_jGA = np.searchsorted(np.cumsum(P_GA), np.random.rand())  # % actual index of GA
        i_GA = param['GA_sequence'][i_jGA]  # % actual GA

        if i_GA < 37:
            i_term_status = 0  # % 0 = preterm, 1 = full term, 2 = postterm
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

            if i_risk_pred or i_preterm_pred == 1 or i_postterm_pred == 1:
                i_loc = np.random.choice([0, 1, 2], p=[0, P_L23_highrisk, 1 - P_L23_highrisk])
            else:
                i_loc = np.random.choice([0, 1, 2],
                                         p=[P_home_lowrisk, (1 - P_home_lowrisk) * 0.89, (1 - P_home_lowrisk) * 0.11])

        # assume among all live births at L45, 5126/(20709 + 5126) going to L5
        P_l5_l45 = 5126 / (20709 + 5126)
        if i_loc == 2 and np.random.random() < P_l5_l45:
            i_loc = 3

        # restore results
        n["LB_L"][i_loc]     += 1
    # Predicted outcomes
    P_l45_all = (n["LB_L"][2] + n["LB_L"][3]) / np.sum(LB_tot) + 0.122

    return P_l45_all