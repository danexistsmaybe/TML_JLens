#!/bin/bash
# Waits until a new process can use the GPU, then restarts what the driver change interrupted.
# Both jobs skip finished work. Checks only; changes nothing on the system.
cd "$(dirname "$0")/.."
until .venv/bin/python -c "import torch; torch.zeros(1).cuda()" >/dev/null 2>&1; do sleep 300; done
echo "GPU back at $(date)"
.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0), torch.version.cuda)" 2>/dev/null
nohup .venv/bin/python -m src.compute_jacobians selected > logs/jac_selected.log 2>&1 &
scripts/1_validation.sh > logs/stage1.log 2>&1
echo "resume script finished"
