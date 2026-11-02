#!/bin/bash
# Stage 1, after the manifest's Jacobians exist: choose settings, score validation, run the checks.
# Nothing here reads test prompts or test windows. Safe to rerun: finished lenses are skipped.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -m src.experiments select
$PY -m src.experiments run 1 --split val
$PY -m src.experiments run 2 --split val
$PY -m analysis.summarize --split val
$PY -m analysis.validation_checks
echo "stage 1 finished"
