import numpy as np
from parameters import get_parameters as get_old_parameters
from parameter_loader import get_parameters as get_new_parameters

rng1 = np.random.default_rng(123)
rng2 = np.random.default_rng(123)

old = get_old_parameters(rng=rng1)
new = get_new_parameters("/Users/poppy/Library/CloudStorage/OneDrive-SharedLibraries-JohnsHopkins/Meibin Chen - MOMISH interventions.xlsx", county="kakamega", rng=rng2)

ignore = []  # add intentionally changed params here

import numpy as np

def compare_values(a, b, tol=0.01, path=""):

    # numpy arrays or array-like numeric values
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):

        arr_a = np.asarray(a)
        arr_b = np.asarray(b)

        if arr_a.shape != arr_b.shape:
            print(f"Shape mismatch at {path}: old {arr_a.shape}, new {arr_b.shape}")
            print("old:", arr_a)
            print("new:", arr_b)
            return False

        return np.allclose(arr_a, arr_b, rtol=tol, atol=tol)

    # dictionaries
    elif isinstance(a, dict) and isinstance(b, dict):

        if a.keys() != b.keys():
            print(f"Dict key mismatch at {path}")
            print("Only old:", set(a.keys()) - set(b.keys()))
            print("Only new:", set(b.keys()) - set(a.keys()))
            return False

        for k in a:
            new_path = f"{path}.{k}" if path else str(k)
            if not compare_values(a[k], b[k], tol=tol, path=new_path):
                return False

        return True

    # lists / tuples
    elif isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):

        if len(a) != len(b):
            print(f"Length mismatch at {path}: old {len(a)}, new {len(b)}")
            print("old:", a)
            print("new:", b)
            return False

        for i, (x, y) in enumerate(zip(a, b)):
            new_path = f"{path}[{i}]"
            if not compare_values(x, y, tol=tol, path=new_path):
                return False

        return True

    # scalars
    else:
        try:
            return bool(np.isclose(a, b, rtol=tol, atol=tol))
        except Exception:
            return a == b

sampled_or_uncertain = {
    "DW",
    "E_MgSO4",
    "E_antibiotics",
    "E_iv_iron",
    "E_oxytocin",
    "OR_MM_CSvsSVD",
    "OR_MM_EmCSvsELCS",
    "OR_MM_transfer",
    "OR_NM_transfer",
    "OR_NEC_treat",
    "RR_Sepsis_treat",
    "RR_comp_highrisk_vs_lowrisk",
    "RR_comp_severe_highrisk_vs_lowrisk",
    "S_oxytocin_l45",
    "S_preterm_treat_l45",
    "base_knowledge_L23",
    "base_knowledge_L45",
    "or_anc_anemia",
    "or_anemia_eclampsia",
    "or_anemia_pph",
    "or_anemia_sepsis",
    "p_OL",
    "p_asphyxia_OL",
    "p_hypoxia",
    "p_mat_sepsis_OL",
    "p_mat_sepsis_emergency_CS",
    "p_pph_elective_CS",
    "p_pph_emergency_CS",
    "sen_hypoxia_IS",
    "sen_prolonged_IS",
    "spec_prolonged_IS",
}

for key in sorted(set(old) | set(new)):
    if key in sampled_or_uncertain:
        continue
    
    if key not in old:
        print(f"Only in new: {key}")
        continue

    if key not in new:
        print(f"Only in old: {key}")
        continue

    same = compare_values(old[key], new[key], tol=0.01, path=key)

    if not same:
        print(f"\nMismatch: {key}")
        print("old:", old[key])
        print("new:", new[key])