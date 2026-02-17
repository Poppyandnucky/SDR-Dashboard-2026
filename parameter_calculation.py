from global_func import GA_assign_kenya, GA_by_ANC
from parameters import get_parameters
import numpy as np
import pandas as pd
import random

P = {}  # dict to restore probabilities
n = {}  # dict to restore counts
OR = {}  # dict to restore odds ratio

param = get_parameters()

#Gestational age assignment to Kenya level
P["GA"] = GA_assign_kenya(param, n, P)
#Adjust GA distribution and preterm rate by ANC status
P["GA_anc"], P["GA_noanc"] = GA_by_ANC(param, OR, P)

# f_ANC_LB_slider with probability clipping

def f_ANC_LB_slider(P_ANC_slider):
    # Initialize parameters
    P = {}  # dict to restore probabilities
    n = {}  # dict to restore counts
    E = {}  # dict to restore effects
    S = {}  # dict to restore supplies and capacities
    OR = {}  # dict to restore odds ratio
    M = {}  # dict to restore maternal outcomes

    # known parameters
    LB_tot = np.array([23729, 18196, 20709, 5126])
    P["highrisk_all"] = 0.26  # high risk pregnancies among all live births

    param = get_parameters()
    P["GA_anc"], P["GA_noanc"] = param["GA_anc"], param["GA_noanc"]

    E["Preterm_LMP"] = param["E_Preterm_LMP"]
    E["Postterm_LMP"] = param["E_Posterm_LMP"]

    # Six parameters - calibrated using simulating annealing
    P_home_noANC, P_l45_fac, P_home_lowrisk, P_L23_highrisk, Sen_traditional, Spec_traditional = \
        0.7056735167353498, 0.21039299793502853, 0.17051451153731043, 0.36751767996993084, 0.7939486174518111, 0.6311811663318615

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
        i_ANC = np.random.binomial(1, P_ANC_slider) if P_ANC_slider < 1 else 1
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

        # restore results
        n["LB_L"][i_loc]     += 1
    # Predicted outcomes
    P_l45_all = n["LB_L"][2] / np.sum(LB_tot)

    return P_l45_all


# Function to compute mean P_l45 over 10 runs for each P_ANC value (optimized)
def compute_mean_P_l45(P_ANC_values, num_runs=10):
    P_l45_results = {P_ANC: np.mean([f_ANC_LB_slider(P_ANC) for _ in range(num_runs)]) for P_ANC in P_ANC_values}
    return P_l45_results

# Define range of P_ANC values from 0.56 to 1.00 with step 0.02
P_ANC_values = np.arange(0.56, 1.02, 0.02)

# Run the function with optimizations
P_l45_results = compute_mean_P_l45(P_ANC_values, num_runs=10)

# Convert results to DataFrame for better visualization
df_P_l45 = pd.DataFrame(list(P_l45_results.items()), columns=["P_ANC", "Mean_P_l45"])

# Display results
print(df_P_l45)




