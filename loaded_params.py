"""Print all parameters loaded from the Excel workbook via parameter_loader."""

import numpy as np
from parameter_loader import get_parameters, calculate_derived_parameters

COUNTY = "kakamega"
SEED = 42

param = get_parameters(county=COUNTY, seed=SEED)
param = calculate_derived_parameters(param)

print("=" * 80)
print(f"PARAMETERS  (county={COUNTY}, seed={SEED})")
print("=" * 80)
for key in sorted(param.keys()):
    val = param[key]
    if isinstance(val, np.ndarray):
        print(f"{key} = np.array({val.tolist()})")
    elif isinstance(val, dict):
        print(f"{key} = {{")
        for k, v in val.items():
            if isinstance(v, np.ndarray):
                print(f"    {k!r}: np.array({v.tolist()}),")
            else:
                print(f"    {k!r}: {v!r},")
        print("}")
    else:
        print(f"{key} = {val!r}")
