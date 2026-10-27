"""Kakamega runs the original model: its women-level outputs must match main.

The reference (data/kakamega_original_reference.json) is the column sums of one
3-month Kakamega baseline run (parameter seed 42, monthly seeds from 20261201) with
main's code and workbook (commit ba034d1). Columns added since (p_mat_death,
i_transfer_group) are not compared.
"""

import json
import math
from pathlib import Path

import numpy as np

from Calibration.calibration_common import simulate

REFERENCE = Path(__file__).parent / "data" / "kakamega_original_reference.json"


def test_kakamega_matches_main():
    reference = json.loads(REFERENCE.read_text())
    people = simulate("kakamega", {}, reference["runs"], reference["months"], 20261201)[0]
    assert len(people) == reference["women"]
    differing = {
        column: (expected, float(np.asarray(people[column], dtype=float).sum()))
        for column, expected in reference["column_sums"].items()
        if not math.isclose(float(np.asarray(people[column], dtype=float).sum()), expected, rel_tol=1e-9, abs_tol=1e-9)
    }
    assert not differing, f"Kakamega differs from main (expected, got): {differing}"
