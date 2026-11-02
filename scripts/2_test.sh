#!/bin/bash
# Stage 2, once settings are frozen and validation_checks has been read: score experiments 1 and 2 on test.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -m src.experiments run 1 --split test
$PY -m src.experiments run 2 --split test
$PY -m src.experiments sensitivity --split test
$PY -m analysis.summarize --split test
echo "stage 2 finished"
