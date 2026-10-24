"""Entry point: validate the current workbook, or recalibrate Stages 1-6 and validate.

    python Calibration/run_calibration.py reproduce
    python Calibration/run_calibration.py recalibrate

Each run writes a timestamped folder under Calibration/results/. recalibrate
writes values.csv (every calibrated value next to its workbook value); the model
runs with these values applied in memory, so the workbook is never modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(script, *arguments):
    subprocess.run([sys.executable, str(REPO_ROOT / "Calibration" / script), *map(str, arguments)],
                   cwd=REPO_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("reproduce", "recalibrate"), nargs="?", default="reproduce")
    parser.add_argument("--output-root", default="Calibration/results")
    args = parser.parse_args()

    output = (REPO_ROOT / args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")).resolve()
    output.mkdir(parents=True)
    validation = ["--output", output / "target_validation.csv"]
    if args.mode == "recalibrate":
        run("calibrate_stages_1_to_5.py", "--output-dir", output)    # Stages 1-5: values.csv
        run("calibrate_stage6_mortality.py", "--stage-dir", output)  # Stage 6: adds the mortality anchors
        validation += ["--values", output / "values.csv"]
    run("validate_baseline_calibration.py", *validation)
    print(f"Calibration {args.mode} completed: {output}")


if __name__ == "__main__":
    main()
