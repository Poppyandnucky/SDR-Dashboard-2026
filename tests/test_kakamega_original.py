"""Kakamega runs the original model: its women-level outputs must match main.

Each reference (data/kakamega_original*_reference.json) is the column sums of one
3-month Kakamega run (parameter seed 42, monthly seeds from 20261201) with main's
code and workbook (commit ba034d1): the baseline, and a run with FQA on. PULSE and
PROMPTS (registration drawn for each mother) were replaced for every county in 2026,
so Kakamega's runs with them differ from main by design. Columns added since (p_mat_death, i_transfer_group) are not compared.
"""

import json
import math
from pathlib import Path

import numpy as np

from Calibration.calibration_common import simulate
from global_func import reset_flags

DATA = Path(__file__).parent / "data"


def assert_matches(reference, people):
    assert len(people) == reference["women"]
    differing = {
        column: (expected, float(np.asarray(people[column], dtype=float).sum()))
        for column, expected in reference["column_sums"].items()
        if not math.isclose(float(np.asarray(people[column], dtype=float).sum()), expected, rel_tol=1e-9, abs_tol=1e-9)
    }
    assert not differing, f"Kakamega differs from main (expected, got): {differing}"


def test_kakamega_matches_main():
    reference = json.loads((DATA / "kakamega_original_reference.json").read_text())
    people = simulate("kakamega", {}, reference["runs"], reference["months"], 20261201)[0]
    assert_matches(reference, people)


def test_kakamega_keeps_original_fqa():
    reference = json.loads((DATA / "kakamega_original_fqa_reference.json").read_text())
    flags = reset_flags()
    flags.update({flag: 1 for flag in reference["flags"]})
    values = {(name, None): value for name, value in reference["values"].items()}
    people = simulate("kakamega", values, reference["runs"], reference["months"], 20261201, flags)[0]
    assert_matches(reference, people)
