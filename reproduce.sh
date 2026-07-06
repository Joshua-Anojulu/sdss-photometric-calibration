#!/usr/bin/env bash
# One-command reproduction of the v2 results and figures.
# Requires data/sdss.csv (regenerate from data/query.sql via SDSS CasJobs; it is gitignored).
set -euo pipefail
pip install -r requirements.txt
python -m pytest -q
if [ ! -f data/sdss.csv ]; then
  echo "ERROR: data/sdss.csv missing. Regenerate it from data/query.sql (SDSS CasJobs)." >&2
  exit 1
fi
python src/calibration_experiment.py --data data/sdss.csv --seeds 20 --outdir results
echo "Done. See results/ and figures/."
