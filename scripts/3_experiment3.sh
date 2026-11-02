#!/bin/bash
# Stage 3, after the selected windows' Jacobians exist. Experiment 3 has fixed budgets, so nothing to choose.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -m src.experiments run 3 --split val
$PY -m src.experiments run 3 --split test
$PY -m analysis.summarize --split test
echo "stage 3 finished"
