import numpy as np
import math
import random
from global_func import sample_from_ci, odds_prob, comps_riskstatus_vs_lowrisk, comp2_comp1_anemia, P_RDS
import streamlit as st

def get_slider_params():
    slider_params = {
        'p_l45_anc_slider': np.array([
            [0.56, 0.255565],
            [0.58, 0.26353],
            [0.60, 0.270298],
            [0.62, 0.277736],
            [0.64, 0.284199],
            [0.66, 0.291049],
            [0.68, 0.296492],
            [0.70, 0.304284],
            [0.72, 0.311035],
            [0.74, 0.318588],
            [0.76, 0.325821],
            [0.78, 0.333545],
            [0.80, 0.339764],
            [0.82, 0.346684],
            [0.84, 0.353901],
            [0.86, 0.360546],
            [0.88, 0.367761],
            [0.90, 0.374581],
            [0.92, 0.38115],
            [0.94, 0.389032],
            [0.96, 0.394433],
            [0.98, 0.401752],
            [1.00, 0.407648]
        ]),                                                     #Predicted probability of delivering at L4/5 by ANC coverage - controlled by slider
        'base_knowledge_L45_slider': 0.60,                      #Baseline knowledge of healthcare providers at L4/5 - default value of slider
        'base_p_45_slider': (20709 + 5126) / np.sum(np.array([23729, 18196, 20709, 5126])), #Baseline probability of delivering at L4/5 - default value of slider
        'p_ANC_base_slider': 0.56,                              #Baseline probability of 4+ ANC - default value of slider
        'S_pph_bundle_slider': np.array([0, 0, 0, 0]),          #PPH bundle implementation - default value of slider
        'S_iv_iron_slider': 0.44,                               #IV iron implementation - default value of slider
        'S_MgSO4_slider': np.array([0, 0, 0.77, 0.77]),         #MgSO4 implementation - default value of slider
        'S_antibiotics_slider': np.array([0, 0, 0.48, 0.48]),   #Antibiotics implementation - default value of slider
        'S_oxytocin_slider': np.array([0, 0, 0.78, 0.78]),      #Oxytocin implementation - default value of slider
        't_l23_l45_notsevere_slider': 76,                       #Probability of referral from L2/3 to L4/5 for not severe cases - default value of slider
    }
    return slider_params

def get_parameters(rng = None):
    if rng is None:
        rng = np.random.default_rng()

    param = {
        ##CHV##
        'n_CHV': 420 * 12,                     #around 420 CHVs per subcounty, 12 subcounties
        'mothers_per_CHV_permonth': 13 / 12,   #13 mothers can be reached by one CHV per year
        'RR_l45_poorQOC': 0.54,                #RR for facility delivery if poor quality of care
        'p_CHV_soften_spread': 0.9,            #probability of CHV softening the spread of negative information

        ##mothers' conditions##
        'p_highrisk': sample_from_ci(0.26, 0.247, 0.273, 4419, 'proportion', 1, rng = rng)[0],                          # probability of high risk pregnancy
        'p_ANC_base': sample_from_ci(0.56, 0.494, 0.626, 216, 'proportion', 1, rng = rng)[0],                           # probability of 4+ ANCs
        'class': sample_from_ci(0.572, 0.506, 0.638, 216, 'proportion', 1, rng = rng)[0],                               # calculated as if above 3 for wealth, education, and has at least a motor bike or car, then at least 1 of the following
        'close_to_L23': sample_from_ci(0.89, 0.886, 0.894, 19127, 'proportion', 1, rng = rng)[0],                       # probability of mothers having cloest distance to L2/3 facility

        ##facility properties##
        "base_LB": np.array([23729, 18196, 20709, 5126]) / 12,                                              # number of live births by facility level at baseline by month
        'base_p_45': (20709 + 5126) / np.sum(np.array([23729, 18196, 20709, 5126])),                        # probability of delivering at L4/5 at baseline
        'base_knowledge_L23': sample_from_ci(0.54, 0.419, 0.661, n=65, kind='proportion', size=1, rng = rng)[0],       # baseline knowledge of healthcare providers at l23
        'base_knowledge_L45': sample_from_ci(0.60, 0.496, 0.704, n=86, kind='proportion', size=1, rng = rng)[0],       # baseline knowledge of healthcare providers at l45
        'Capacity': 34777 / 12,                                                                             # baseline facility capacity of L4/5
        'p_l5_l45': 5126 / (20709 + 5126),                                                                  # probability of delivering at l5 if delivering at L45
        'base_surgical': [0, 25, 7],                                                                        #number of surgical staff in L2/3, l4 and l5 at baseline
        'base_nurse': [(223+1038-147-57), 147, 57],                                                         #number of nurses in L2/3, l4 and l5 at baseline
        'base_anesthetist': [0, 20, 11],                                                                    #number of anesthetists in L2/3, l4 and l5 at baseline
        'num_L2/3': 187,                                                                                    # number of L2/3 facilities for redesign
        'num_L4': 17,                                                                                       #number of L4 facilities for redesign
        'num_L5': 1,                                                                                        #number of L5 facilities for redesign
        'num_dopplers_L2/3': math.floor(52 / 58 * 187),                                                     #number of dopplers in L2/3 facilities
        'num_dopplers_L4': math.floor(10 / 5 * 17),                                                         #number of dopplers in L4 facilities
        'num_dopplers_L5': 2,                                                                               #number of dopplers in L5 facilities
        'num_CTGs_L2/3': 0,                                                                                 #number of CTG machines in L2/3 facilities
        'num_CTGs_L4': 0,                                                                                   #number of CTG machines in L4 facilities
        'num_CTGs_L5': 0,                                                                                   #number of CTG machines in L5 facilities

        ##parameters in LB_effect.py##
        # calibrated in model
        "home_noANC": 0.706, #calculated based on P(home)=0.35 and P(home|anc)=0.07 #0.7056735167353498,    # probability of home delivery without ANC
        "l45_fac": 0.11, #assumed based on cloest distance to L4/5 if no ANC and based on mothers' decision #0.21039299793502853,                                                                     # probability of delivery at L4/5 facility if chosing deliver at facilities and without ANC
        "home_lowrisk": 0.1692, #0.17051451153731043(old version),                                          # probability of home delivery if predicted as low risk
        "L23_highrisk": 0.3255, #0.36751767996993084(old version),                                          # probability of delivery at L2/3 facility if predicted as high risk
        "sen_risk_trad": 0.7641, #0.7939486174518111(old version),                                          # sensitivity of traditional ANC monitoring in predicting high risk
        "spec_risk_trad": 0.6306, #0.6311811663318615(old version),                                         # specificity of traditional ANC monitoring in predicting low risk
        "PT_scale": 0.8250540888309176,                                                                     # scaling factor for reducing preterm birth rate to Kenya level
        "p_elec_CS|highrisk": 0.0642,                                                                       # probability of elective CS if predicted as high risk
        "p_elec_CS|preterm": 0.7799,                                                                        # probability of elective CS if predicted as preterm
        "p_elec_CS|highrisk_us": 0.35,                                                                      # probability of elective CS if predicted as high risk by ultrasound
        "p_elec_CS|preterm_us": 0.7799,                                                                     # probability of elective CS if predicted as preterm by ultrasound

        # known parameters
        "p_lb_l23_45": sample_from_ci(0.122, 0.119, 0.125, n=50981, kind='proportion', size=1, rng = rng)[0],           # probability of live births at L4/5 if incoming maternal referrals
        "GA_sequence": np.arange(27, 45),                                                                                                  # gestational age sequence
        'OR_preterm_ANC': sample_from_ci(0.48, 0.479, 0.481, kind='OR', size=1, rng = rng)[0],                          # odds ratio of preterm birth given adequate ANC
        "GA_distribution":                                                                                                                 # Pooled (Hazel et al.) gestational age distribution
            np.array([
                    1104,  # GA(0) = 27 weeks completed (similar to < 28 weeks completed)
                    625,   # GA(1) = 28 weeks completed
                    853,   # GA(2) = 29 weeks completed
                    1269,  # GA(3) = 30 weeks completed
                    1707,  # GA(4) = 31 weeks completed
                    2479,  # GA(5) = 32 weeks completed
                    3584,  # GA(6) = 33 weeks completed
                    5790,  # GA(7) = 34 weeks completed
                    10109, # GA(8) = 35 weeks completed
                    15665, # GA(9) = 36 weeks completed
                    23644, # GA(10) = 37 weeks completed
                    33812, # GA(11) = 38 weeks completed
                    45770, # GA(12) = 39 weeks completed
                    40159, # GA(13) = 40 weeks completed
                    23973, # GA(14) = 41 weeks completed
                    12425, # GA(15) = 42 weeks completed
                    6117, # GA(16) = 43 weeks completed
                    3378  # GA(17) = 44 weeks completed
                    ]),
        "E_GA_US": np.array([.057, .729]),                      #% Normal distribution parameters for GA error by ultrasound - sd and mean
        'E_Preterm_LMP': np.array([                             #probability of predicting as preterm birth by LMP by actual preterm vs at-term
            sample_from_ci(0.643, 0.641, 0.645, n=165908, kind='proportion', size=1, rng = rng)[0],  # if preterm
            sample_from_ci(0.039, 0.038, 0.040, n=165908, kind='proportion', size=1, rng = rng)[0],  # if at-term
        ]),
        'E_Postterm_LMP': np.array([                           #probability of predicting as postterm birth by LMP by actual preterm vs at-term
            sample_from_ci(0.030, 0.029, 0.031, n=165908, kind='proportion', size=1, rng = rng)[0],  # if preterm
            sample_from_ci(0.105, 0.104, 0.106, n=165908, kind='proportion', size=1, rng = rng)[0],  # if at-term
        ]),

        #parameters after calculation
        'GA_anc': np.array([0.00284327, 0.00160964, 0.00219684, 0.00326822, 0.00439625,
       0.00638448, 0.00923033, 0.01491172, 0.02603499, 0.04034406,
       0.11102357, 0.15876878, 0.21491917, 0.18857196, 0.11256843,
       0.05834325, 0.02872319, 0.01586185]), # P(GA|ANC)
        'GA_noanc': np.array([0.00528652, 0.00299282, 0.0040846, 0.00607662, 0.00817399,
                           0.01187072, 0.01716203, 0.02772549, 0.04840708, 0.07501206,
                           0.099085, 0.14169608, 0.19180852, 0.16829448, 0.10046375,
                           0.0520695, 0.02563454, 0.0141562]), # P(GA|no ANC)

        ##parameters in intrapatum.py##
        # calibrated
        "p_aph": 0.0176,                                        # probability of antepartum hemorrhage
        "p_eclampsia": 0.0164,                                  # probability of eclampsia - calibrated to match the rate
        'p_ruptured_uterus': 0.0191,                            # probability of getting other complications - calibrated to match the rate
        "p_cs_capacity": np.array([0, 0.0568, 0.1215, 0.1215]), # probability of EmCS if predicted with complications - reflecting the capacity of CS - calibrated to match CS rate
        "p_OL_scale": 0.7982,                                   # scaling factor for reducing obstructed labor from Ethiopia to Kenya level
        "p_pph_other": 0.0100,                                  # probability of PPH by not OL and not CS, but other risk factors - calibrated to match PPH in Kenya
        "p_mat_sepsis_other": 0.0356,                           # probability of maternal sepsis by not OL, not CS, but other risk factors - calibrated to match matnernal sepsis in Kenya

        # assumption
        "p_cs_capacity_sdr": np.array([0, 0, 0.1215, 0.1215]),  # no surgical capacity for L2/3 - assumption
        "p_cs_capacity_sensor": np.array([0, 0.0568, 0.1215, 0.1215]),# probability of EmCS if predicted with complications by intrapartum sensors - assumption
        "p_cs_capacity_sdr_sensor": np.array([0, 0, 0.1215, 0.1215]),
        "sen_comp_trad": 0.70,                                  # sensitivity in predicting complications of traditional intrapartum monitoring - assumption
        "spec_comp_trad": 0.75,                                 # specificity in predicting complications of traditional intrapartum monitoring - assumptiom

        # known parameters
        'RR_comp_highrisk_vs_lowrisk': sample_from_ci(4.2, 1.79, 10.86, kind='RR', size=1, rng = rng)[0], # relative risk of complications if high risk compared to low-risk pregancies
        'p_PL_GA': np.array([
            sample_from_ci(0.073, 0.067, 0.079, n=6168, kind='proportion', size=1, rng = rng)[0],  #GA = 37
            sample_from_ci(0.079, 0.075, 0.083, n=15773, kind='proportion', size=1, rng = rng)[0], #GA = 38
            sample_from_ci(0.092, 0.089, 0.095, n=30685, kind='proportion', size=1, rng = rng)[0], #GA = 39
            sample_from_ci(0.112, 0.109, 0.115, n=44722, kind='proportion', size=1, rng = rng)[0], #GA = 40
            sample_from_ci(0.149, 0.144, 0.154, n=18638, kind='proportion', size=1, rng = rng)[0], #GA = 41
            sample_from_ci(0.197, 0.183, 0.211, n=3268, kind='proportion', size=1, rng = rng)[0],  #GA = 42+
        ]),                                                  # probability of prolonged labor by gestational age
        "p_OL":  np.array([
            sample_from_ci(0.07, 0.042, 0.098, n=318, kind='proportion', size=1, rng = rng)[0], # probability of obstructed labor if not prolonged
            sample_from_ci(0.185, 0.142, 0.228, n=318, kind='proportion', size=1, rng = rng)[0], # probability of obstructed labor if prolonged
        ]),
        "p_hypoxia": sample_from_ci(0.105, 0.083, 0.127, n=779, kind='proportion', size=1, rng = rng)[0],                              # probability of hypoxia
        "p_mat_sepsis_OL": sample_from_ci(0.357, 0.227, 0.474, n=None, kind='proportion', size=1, rng = rng)[0],                       # probability of maternal sepsis by OL
        "p_mat_sepsis_elective_CS": sample_from_ci(0.07, 0.032, 0.108, n=170, kind='proportion', size=1, rng = rng)[0],                # probability of maternal sepsis by elective CS
        "p_mat_sepsis_emergency_CS": sample_from_ci(0.335, 0.264, 0.406, n=170, kind='proportion', size=1, rng = rng)[0],              # probability of maternal sepsis by emergency CS
        "p_pph_OL": sample_from_ci(0.148, 0.088, 0.208, n=135, kind='proportion', size=1, rng = rng)[0],                               # probability of PPH by OL
        "p_pph_elective_CS": sample_from_ci(0.064, 0.027, 0.101, n=170, kind='proportion', size=1, rng = rng)[0],                      # probability of PPH by elective CS
        "p_pph_emergency_CS": sample_from_ci(0.188, 0.129, 0.247, n=170, kind='proportion', size=1, rng = rng)[0],                     # probability of PPH by emergency CS
        "p_stillbirth_OL": sample_from_ci(0.185, 0.119, 0.251, n=135, kind='proportion', size=1, rng = rng)[0],                        # probability of stillbirth by OL
        "p_asphyxia_OL": sample_from_ci(0.692, 0.515, 0.869, n=26, kind='proportion', size=1, rng = rng)[0],                           # probability of asphyxia by OL
        "p_neo_sepsis_OL": sample_from_ci(0.165, 0.102, 0.228, n=135, kind='proportion', size=1, rng = rng)[0],                        # probability of neonatal sepsis by OL
        "E_stillbirth_CS": 0.55,                                                                                                                          # efficacy of timely CS in preventing stillbirth by hypoxia
        "p_stillbirth_hypoxia": 0.27,                                                                                                                     # probability of stillbirths by hypoxia
        'sen_prolonged_IS': sample_from_ci(0.74, 0.65, 0.82, n=None, kind='proportion', size=1, rng = rng)[0],                          # sensitivity of predicting prolonged labor by intrapartum sensors
        'spec_prolonged_IS': sample_from_ci(0.77, 0.69, 0.84, n=None, kind='proportion', size=1, rng = rng)[0],                        # specificity of predicting prolonged labor by intrapartum sensors
        'sen_ol_IS': sample_from_ci(0.87, 0.817, 0.923, n=154, kind='proportion', size=1, rng = rng)[0],                               # sensitivity of predicting obstructed labor by intrapartum sensors
        "spec_ol_IS": 0.95,                                                                                                                               # specificity of predicting obstructed labor by intrapartum sensors
        'sen_hypoxia_IS': sample_from_ci(0.774, 0.739, 0.809, n=552, kind='proportion', size=1, rng = rng)[0],                         # sensitivity of predicting hypoxia by intrapartum sensors
        'spec_hypoxia_IS': sample_from_ci(0.939, 0.919, 0.959, n=552, kind='proportion', size=1, rng = rng)[0],                        # specificity of predicting hypoxia by intrapartum sensors
        'CS_AVD_ratio': sample_from_ci(0.985, 0.984, 0.986, n=44031, kind='proportion', size=1, rng = rng)[0],                         # ratio of CS to AVD at L4/5
        'p_comp_anemia': sample_from_ci(0.25, 0.250, 0.250, n=3544672, kind='proportion', size=1, rng = rng)[0],                       # probability of getting anemia
        'or_anc_anemia': sample_from_ci(2.26, 1.05, 4.89, kind='OR', size=1, rng = rng)[0],                                            # odds ratio of getting anemia without ANC
        'scale_comps': 4.48,                                                                                                                              # scaling factor for increasing maternal complications to Kenya level
        'or_anemia_pph': sample_from_ci(3.54, 1.20, 10.4, kind='OR', size=1, rng = rng)[0],                                            # odds ratio of getting PPH given anemia
        'or_anemia_sepsis': sample_from_ci(5.68, 4.38, 7.36, kind='OR', size=1, rng = rng)[0],                                          # odds ratio of getting sepsis given anemia
        'or_anemia_eclampsia': sample_from_ci(3.74, 2.79, 4.81, kind='OR', size=1, rng = rng)[0],                                       # odds ratio of getting eclampsia given anemia
        "p_RDS_noT": np.array([90] + [60] * 4 + [40] * 3 + [20] * 2 + [0] * 8) / 100,                                                                      # probability of RDS by GA without treatment
        'RR_comp_severe_highrisk_vs_lowrisk': sample_from_ci(4.2, 1.68, 10.5, kind='RR', size=1, rng = rng)[0],                         # relative risk of severe complications for high-risk mothers compared to low-risk mothers

        ##parameters in mortality.py##
        # calibrated
        "p_MM_others": 0.0053,                         # probability of maternal mortality by other causes - calibrated to match MMR due to other causes
        "MM_weight_pph": 2.0000,                       # weight of PPH in calculating MMR if getting SMO - calibrated to match MMR due to PPH
        "MM_weight_sepsis": 0.4760,                    # weight of maternal sepsis in calculating MMR if getting SMO - calibrated to match MMR due to maternal sepsis
        "MM_weight_eclampsia": 2.0000,                 # weight of eclampsia in calculating MMR if getting SMO - calibrated to match MMR due to eclampsia
        "MM_weight_ol": 0.1855,                        # weight of obstructed labor in calculating MMR if getting SMO - calibrated to match MMR due to obstructed labor
        "MM_weight_aph": 2.0000,                       # weight of antepartum hemorrhage in calculating MMR if getting SMO - calibrated to match MMR due to antepartum hemorrhage
        'p_MM_home': 0.1456,                           # maternal mortality rate due to SMO by home - calibrated to match MMR at home
        'weight_facility_mat': 2.6661,                 # weight of facility in calculating MMR due to healthcare worker density if getting SMO
        'p_comp_severe_lowrisk': 0.0501,               # probability of severe if complications for low-risk mothers - calibrated to match severe maternal outcome rate at L4/5
        'p_NM_home': 0.235,                            # neonatal mortality rate by home - calibrated to match NMR at home (need revision)
        'weight_facility_neo': 3.15,                   # weight of facility in calculating NMR due to healthcare worker density (need revision)

        # known parameters
        'OR_MM_CSvsSVD': sample_from_ci(2.28, 1.87, 2.79, kind='OR', size=1, rng = rng)[0],                                 # odds ratio of maternal mortality by CS vs SVD
        'OR_MM_EmCSvsELCS': sample_from_ci(3.17, 2.48, 4.04, kind='OR', size=1, rng = rng)[0],                              # odds ratio of maternal mortality by emergency CS vs elective CS
        'OR_MM_transfer': sample_from_ci(1.59, 1.30, 1.93, kind='OR', size=1, rng = rng)[0],                                # odds ratio of maternal mortality by transfer
        'OR_NM_transfer': sample_from_ci(2.5, 1.1, 5.6, kind='OR', size=1, rng = rng)[0],                                   # odds ratio of neonatal mortality by transfer
        "D_RDS": np.array([39.7, 20.0]) / 100,              # % P[death|RDS] for GA < 32 or >= 32
        "D_IVH": 11.0 / 100,                                # % P[death|IVH]
        "D_NEC": 21.2 / 100,                                # % P[death|NEC]
        "D_Sepsis": 32.5 / 100,                             # % P[death|Sepsis]
        "D_asphyxia": 0.239,                                # probability of death by asphyxia

        ##parameters for emergency transfer rate
        # calibrated
        't_l23_l45_preterm': 80.9485,                       # probability of transfer from L2/3 to L4/5 if preterm - calibrated to match preterm rate by facility level
        't_l23_l45_notsevere': 74.2358,                     # probability of transfer from L2/3 to L4/5 if not severe - calibrate to match live birth distribution by facility level
        't_l23_l45_severe': 90.0000,                        # probability of transfer from L2/3 to L4 - calibrated to match severe maternal outcome rate at L4/5
        # known parameters
        't_l4_l4_severe': 16.6,                             # probability of transfer from L4 to L4
        't_l4_l5_severe': 44.7,                             # probability of transfer from L4 to L5

        ##parameters for single interventions

        #ANC
        'S_iv_iron': 0.44,                                                                                                            # the supply level of IV iron at facilities
        #Intrapartum + PNC
        'S_oxytocin_l45': sample_from_ci(0.78, 0.706, 0.854, n=120, kind='proportion', size=1, rng = rng)[0],      # the supply level of oxytocin at L4/5
        'S_preterm_treat_l45': sample_from_ci(0.35, 0.285, 0.415, n=206, kind='proportion', size=1, rng = rng)[0], # the supply level of preterm treatment at L4/5
        'S_pph_bundle': np.array([0, 0, 0, 0]),                                                                                       # the supply level of obstetric drape at L4/5
        'S_MgSO4': np.array([0, 0, 0.77, 0.77]),                                                                                      # the supply level of MgSO4 at L4/5
        'S_antibiotics': np.array([0, 0, 0.48, 0.48]),                                                                                # the supply level of antibiotics at L4/5
        "OR_RDS_treat": 0.53,                                                                                                         # odds ratio of preterm having RDS given treatment
        'OR_IVH_treat': 0.38, 
        "OR_knowledge": 1.99,                                                                                                        # odds ratio of preterm having IVH given treatment
        'OR_NEC_treat': sample_from_ci(0.28, 0.14, 0.56, n=580, kind='OR', size=1, rng = rng)[0],                  # odds ratio of preterm having NEC given treatment
        'RR_Sepsis_treat': sample_from_ci(0.24, 0.13, 0.44, n=2063, kind='RR', size=1, rng = rng)[0],              # relative risk of preterm having sepsis given treatment
        'E_pph_bundle': sample_from_ci(0.51, 0.44, 0.60, kind='RR', size=1, rng = rng)[0],                         # efficacy of obstetric drape in reducing PPH
        'E_iv_iron': sample_from_ci(0.30, 0.203, 0.397, n=86, kind='proportion', size=1, rng = rng)[0],            # efficacy of IV iron infusion in reducing anemia
        'E_MgSO4': sample_from_ci(0.41, 0.29, 0.58, n=11444, kind='RR', size=1, rng = rng)[0],                     # efficacy of MGSO4 in reducing eclampsia
        'E_antibiotics': sample_from_ci(0.67, 0.56, 0.79, n=14590, kind='RR', size=1, rng = rng)[0],               # efficacy of antibiotics in reducing maternal sepsis
        'E_oxytocin': sample_from_ci(0.91, 0.827, 0.993, n=46, kind='proportion', size=1, rng = rng)[0],           # efficacy of oxytocin
        # PROMPTS mechanism parameters (NOT user inputs)
        'phone_ownership': 0.89,
        'intervention_fidelity': 0.87,
        'OR_anc4p': 1.38,
        'sen_risk_trad_target': 0.95, ## under case where mothers recognize danger signs
        'spec_risk_trad_target': 0.631,
        'p_move_home_base': 0.3,
        ###  parameter 
        
        ##parameters in DALY calculation##
        'DW': { # disability weights for different conditions
            'anemia': sample_from_ci(0.052, 0.0338, 0.0757, kind='mean', size=1, rng = rng)[0],
            'low pph': sample_from_ci(0.114, 0.0779, 0.1587, kind='mean', size=1, rng = rng)[0],
            'high pph': sample_from_ci(0.324, 0.2197, 0.4418, kind='mean', size=1, rng = rng)[0],
            'maternal sepsis': sample_from_ci(0.133, 0.0884, 0.1895, kind='mean', size=1, rng = rng)[0],
            'eclampsia': sample_from_ci(0.602, 0.4266, 0.7527, kind='mean', size=1, rng = rng)[0],
            'obstructed labor': sample_from_ci(0.324, 0.2197, 0.4418, kind='mean', size=1, rng = rng)[0],
            'maternal death': 1,
            'neonatal death': 1,
            'preterm comp': 0.388,
            'asphyxia': 0.549,
            'neonatal sepsis': 0.459,
        },
        'Mother_life_expectancy': 69.2,                # life expectancy of mothers in Kenya
        'Neonate_life_expectancy': 66.8,               # life expectancy of neonates in Kenya
        'Childbearing_age': 28.6,                      # average childbearing age in Kenya

        ##parameters for labor calculation##
        'n_population': 1867283,                        # total population in Kakamega County
        'density_skilled_worker_kenya': 174.09,         # density of skilled healthcare workers in Kenya
        'Ave_LBs_thres': 167,                           # threshold for average number of live births at L4/5 - for calculating the number of surgical staff
        'surgical_needed_below_thres': 6,               # number of surgical staff needed if average number of live births at L4 is below threshold
        'surgical_needed_perLB_above_thres': 0.036,     # number of surgical staff needed per live births if average number of live births at L4 is above threshold
        'nurse_needed_perLB': 1/5.4,                    # number of nurses needed per live births
        'anesthetist_needed_perCS': 0.0435,             # number of anesthetists needed per live births
        'surgical_weight': 10,                          # weight of surgical staff in calculating quality of care
        'scaled_factor_density': 6,                     # scaling factor for density of skilled healthcare workers

        ##parameters for fetal sensors calculation##
        'check_time_doppler': 10,                       # time to check fetal heart rate by doppler in minutes
        '1st_stage_time_normal': [4*60+38, 80],         # duration of 1st stage of labor in minutes for normal delivery - mean and standard deviation
        '2nd_stage_time_normal': [37.26, 14],           # duration of 2nd stage of labor in minutes for normal delivery - mean and standard deviation
        '1st_stage_time_abnormal': [7*60+48, 2*60+47],  # duration of 1st stage of labor in minutes for abnormal labor delivery - mean and standard deviation
        '2nd_stage_time_abnormal': [86, 49],            # duration of 2nd stage of labor in minutes for abnormal labor delivery - mean and standard deviation
        'check_interval_1st_stage': 30,                 # time interval to check fetal heart rate by doppler in minutes during 1st stage of labor
        'check_interval_2nd_stage': 15,                 # time interval to check fetal heart rate by doppler in minutes during 2nd stage of labor
        'usage_time_sensor_perday': 20*60,              # time to use doppler/CTG per day in minutes, assuming 20 hours allowing for cleaning and downtime

        ##parameters for cost calculation##
        'USD_to_Ksh': 129.5,                            # exchange rate from USD to Ksh
        'cost_dict': {
                    'pph_bundle': 129.25*1.08*49101/4158, #(USD12.75 per pph case) #Additional cost per pph patient
                    'iv_iron': 2843.50,             #Ksh 2,843.50 per patient for IV iron infusion
                    'MgSO4': 1680.25,               #Ksh 1,680.25 per patient for MgSO4
                    'antibiotics': 125.37,          #Ksh 125.37 per patient for Antibiotics
                    'POCUS': 5000 * 129.5,          #USD 5000 per POCUS machine
                    'Doppler': 4000,                #Ksh 4000 per Doppler machine.
                    'CTG': 657900,                  #Ksh 657,900 per CTG machine
                    'SDR PM': 40207381,             #Fixed cost: program management
                    'SDR PM2': 5693559,             #Maintainance cost for program management per year
                    'SDR PM2 Conservative': 4556971, #Maintainance cost for program management per year under conservative secenario
                    'SDR PM2 Moderate': 5693559,     #Maintainance cost for program management per year under moderate secenario
                    'SDR PM2 Aggressive': 6833850,   #Maintainance cost for program management per year under aggressive secenario
                    'SDR Infra': 1031600000 / 457,  #Fixed cost: infrastructure cost per maternity bed and related infrastructure
                    'SDR Equip': 11359478,          #Fixed cost: Equipment
                    'SDR Taxi Setup': 19000,        #Setup cost for one free taxi/boda
                    'SDR Taxi Monthly': 250,        #Monthly cost for one free taxi/boda
                    'SDR Taxi Dispatch': 5500,      #Dispatch cost per dispatch
                    'SDR Ambulance Setup': 130500,  #Setup cost for one ambulance
                    'SDR Ambulance Monthly': 2250,  #Monthly cost for one ambulance
                    'SDR Ambulance Dispatch': 8500, #Dispatch cost per dispatch
                    'SDR ANC': 312,                 #ANC cost per patient
                    'SDR Fac Delivery': 6148,       #Facility normal delivery cost per patient
                    'SDR CS Delivery': 29804,       #C-section delivery cost per patient
                    'SDR surgical staff': 9958,     #monthly salary for one surgical staff
                    'SDR nurse staff': 4380,        #monthly salary for one nurse staff
                    'SDR anesthetist': 5098,        #monthly salary for one anesthetist
                },
        'dispatches_per_vehicle': 57,               # number of dispatches per vehicle per month

        ##Calibration criteria
        "home_all_target": 0.35,                               # % home births pre-transfers
        "l23_all_target": 0.27 + 0.122,                        # % l23 births pre-transfers, 12.2% of all births occurred after incoming maternal referrals in Uganda
        "l45_all_target": 0.38 - 0.122,                        # % l45 births pre-transfers
        "home_anc_target": 0.07,                               # % home births among mothers with 4+ ANCs
        "l23_anc_target": 0.490,                               # % l23 births among mothers with 4+ ANCs - calculated based on total probability theory given known live birth distribution and ANC coverage at home
        "l45_anc_target": 0.439,                               # % l45 births among mothers with 4+ ANCs - calculated based on total probability theory given known live birth distribution and ANC coverage at home
        "preterm_fac_target": 0.134,                           # preterm birth rate in facilities
        "elective_CS_target": 0.06,                            # % elective CS = 0.06 for matching 40% of all CSs (16.5%) are elective CS
        "elective_CS_preterm_target": 0.265,                   # % elective CS among preterm deliveries
    }

    return param

def calculate_derived_parameters(param):
    # caculate some parameters
    param['p_anemia_anc'] = odds_prob(param['or_anc_anemia'], param['p_comp_anemia'], (1 - param['p_ANC_base']))           # probability of anemia if ANC
    param["severe_highrisk"] = param['p_comp_severe_lowrisk'] * param['RR_comp_severe_highrisk_vs_lowrisk']                # probability of severe complications for high-risk pregnancies
    param["severe"] = np.array([param['p_comp_severe_lowrisk'], param["severe_highrisk"]])                                 # probability of severe complications by risk level
    param["OL"] = param["p_OL"] * param["p_OL_scale"]                                                                      # probability of obstructed labor in the model adjusted to Kenya level
    param["OL_highrisk"], param["OL_lowrisk"] = comps_riskstatus_vs_lowrisk(param["OL"][0], param['p_highrisk'],
                                                                            param["RR_comp_highrisk_vs_lowrisk"])           # probability of obstructed labor by risk status
    param["ruptured_uterus_highrisk"], param["ruptured_uterus_lowrisk"] = comps_riskstatus_vs_lowrisk(
        param["p_ruptured_uterus"], param['p_highrisk'], param["RR_comp_highrisk_vs_lowrisk"])                              # probability of ruptured uterus by risk status
    param["aph_highrisk"], param["aph_lowrisk"] = comps_riskstatus_vs_lowrisk(param["p_aph"], param['p_highrisk'],
                                                                              param["RR_comp_highrisk_vs_lowrisk"])        # probability of antepartum hemorrhage by risk status
    param["eclampsia_highrisk"], param["eclampsia_lowrisk"] = comps_riskstatus_vs_lowrisk(param["p_eclampsia"],
                                                                                          param['p_highrisk'], param[
                                                                                              "RR_comp_highrisk_vs_lowrisk"]) # probability of eclampsia by risk status
    param["eclampsia_highrisk_anemia"] = comp2_comp1_anemia(param["eclampsia_highrisk"], param['or_anemia_eclampsia'])      # probability of eclampsia if high-risk by anemia status
    param["eclampsia_lowrisk_anemia"] = comp2_comp1_anemia(param["eclampsia_lowrisk"], param['or_anemia_eclampsia'])        # probability of eclampsia if low-risk by anemia status

    # maternal complications with anemia
    param["pph_OL_anemia"] = comp2_comp1_anemia(param["p_pph_OL"], param['or_anemia_pph'])                                  # probability of PPH if OL by anemia status
    param["mat_sepsis_OL_anemia"] = comp2_comp1_anemia(param["p_mat_sepsis_OL"], param['or_anemia_sepsis'])                 # probability of maternal sepsis if OL by anemia status
    param["pph_elective_CS_anemia"] = comp2_comp1_anemia(param["p_pph_elective_CS"], param['or_anemia_pph'])                # probability of PPH if elective CS by anemia status
    param["mat_sepsis_elective_CS_anemia"] = comp2_comp1_anemia(param["p_mat_sepsis_elective_CS"],
                                                                param['or_anemia_sepsis'])                                  # probability of maternal sepsis if elective CS by anemia status
    param["pph_emergency_CS_anemia"] = comp2_comp1_anemia(param["p_pph_emergency_CS"], param['or_anemia_pph'])              # probability of PPH if emergency CS by anemia status
    param["mat_sepsis_emergency_CS_anemia"] = comp2_comp1_anemia(param["p_mat_sepsis_emergency_CS"],
                                                                 param['or_anemia_sepsis'])                                 # probability of maternal sepsis if emergency CS by anemia status
    param["pph_other_anemia"] = comp2_comp1_anemia(param["p_pph_other"], param['or_anemia_pph'])
    param["mat_sepsis_other_anemia"] = comp2_comp1_anemia(param["p_mat_sepsis_other"], param['or_anemia_sepsis'])           # probability of maternal sepsis if other complications by anemia status

    param["RDS_T"] = P_RDS(param) # probability of RDS by GA with treatment

    # supply
    param["S_oxytocin"] = np.array([0, 0, param["S_oxytocin_l45"], param["S_oxytocin_l45"]])                         # supply level of oxytocin at different facility levels
    param["S_preterm_treat"] = np.array([0, 0, param["S_preterm_treat_l45"], param["S_preterm_treat_l45"]])          # supply level of preterm treatment at different facility levels
    return param

def reset_inputs(param, n_months):
    '''reset of whole county characteristics, tracking of live births, and intervention-specific characteristics'''
    n_months = n_months
    LB = param['base_LB']
    ANC = param['p_ANC_base']
    CLASS = param['class']
    highrisk = param['p_highrisk']

    # tracking the whole county properties (i.e. number of live births, coverage of ANC, etc.)
    track = dict({'LB': LB,
               'ANC': ANC,
               'CLASS': CLASS,
               'LB_Track': np.zeros((n_months, 4)),
               'ANC_Track': np.zeros((n_months, 4)),
               'HighRisk_Track': np.zeros((n_months, 4)),
               'Facility_Capacity_Track': np.zeros((n_months, 1)),
               'Referral_Capacity_Track': np.zeros((n_months, 1)),
               'Num_Exp_L45_Track': np.zeros((n_months, 1)), #track the expected number of live births at L4/5
               'Constraint_Ratio_Track': np.zeros((n_months, 1)),
               'CS_Capacity_Track': np.zeros((n_months, 1)),
               'CHV_negative_Track': np.zeros((n_months, param["n_CHV"]), dtype=int),  # 0/1 negative experience each month
               'CHV_memory_Track': np.zeros((n_months, param["n_CHV"]), dtype=int),    # memory age in months
               })

    track['LB_Track'][0,:] = track['LB']
    track['ANC_Track'][0,:] = np.repeat(ANC, 4)
    track['HighRisk_Track'][0,:] = np.repeat(highrisk, 4)
    track['Facility_Capacity_Track'][0, 0] = 34777 / 12
    track['Referral_Capacity_Track'][0, 0] = 0
    track['Num_Exp_L45_Track'][0, 0] = (20709 + 5126) / 12
    track['Constraint_Ratio_Track'][0, 0] = 1
    track['CS_Capacity_Track'][0, 0] = param["p_cs_capacity"][3]

    return track