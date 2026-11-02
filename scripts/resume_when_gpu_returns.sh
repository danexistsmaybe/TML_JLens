#!/bin/bash
# Resumes the interrupted work once a new process can use the GPU. Run it detached:
#     setsid nohup scripts/resume_when_gpu_returns.sh > logs/resume.log 2>&1 < /dev/null &
# Stops without resuming if the correctness checks fail on the current driver. Finished work
# is skipped by every job, so this is safe to run again. It changes nothing on the system.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
until $PY -c "import torch; torch.zeros(1).cuda()" >/dev/null 2>&1; do sleep 300; done
echo "GPU back at $(date)"
$PY -c "import torch; print(torch.cuda.get_device_name(0), 'torch CUDA', torch.version.cuda)" 2>/dev/null
cat /proc/driver/nvidia/version | head -1

# The Jacobians were computed on the old driver. They must still match fresh ones.
if ! $PY -m tests.check_jacobian 2>/dev/null | tee logs/check_after_driver.log | grep -qx PASS; then
    echo "CHECK FAILED: tests.check_jacobian. Not resuming."; exit 1
fi
if ! $PY -m tests.test_pipeline fitting_cached 2>/dev/null | tee -a logs/check_after_driver.log | grep -q "^all passed"; then
    echo "CHECK FAILED: cached Jacobians differ from fresh ones. Not resuming."; exit 1
fi
echo "checks passed at $(date)"

nohup $PY -m src.compute_jacobians selected > logs/jac_selected.log 2>&1 &
scripts/1_validation.sh > logs/stage1.log 2>&1 && echo "validation finished at $(date)" || echo "VALIDATION FAILED: see logs/stage1.log"
