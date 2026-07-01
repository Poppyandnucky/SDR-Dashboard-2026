#!/usr/bin/env bash
# Monitor and restart notebook execution on failure
REPO_DIR="/Users/poppy/Documents/GitHub/SDR-Dashboard-2026"
NOTEBOOK="$REPO_DIR/Calibration/Phase1_ANC_calibration_v7_kisii_executed.ipynb"
LOG_DIR="$REPO_DIR/Calibration/monitor_logs"
mkdir -p "$LOG_DIR"

echo "Monitor started for notebook: $NOTEBOOK"

while true; do
  TS="$(date +%Y%m%d_%H%M%S)"
  LOGFILE="$LOG_DIR/run_$TS.log"
  OUTNAME="executed_$TS.ipynb"

  echo "=== START: $TS ===" | tee -a "$LOGFILE"
  cd "$REPO_DIR" || exit 1

  # Execute notebook and capture output
  jupyter nbconvert --to notebook --execute "$NOTEBOOK" --ExecutePreprocessor.timeout=-1 --output "$OUTNAME" --output-dir "$LOG_DIR" >> "$LOGFILE" 2>&1
  EXIT=$?

  if [ $EXIT -eq 0 ]; then
    echo "Run completed successfully at $(date)" | tee -a "$LOGFILE"
    break
  else
    echo "Run failed with exit code $EXIT at $(date)" | tee -a "$LOGFILE"
    echo "Retrying in 10s..." | tee -a "$LOGFILE"
    sleep 10
  fi
done

echo "Monitor finished." >> "$LOGFILE"
