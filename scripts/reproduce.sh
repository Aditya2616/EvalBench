#!/usr/bin/env bash
# Reproduce every number in the README and the recorded demo in docs/.
# Needs about 3 GB of model downloads and several hours of CPU on a busy machine.
# Generation resumes from runs/main/generations if you stop and restart it.
set -euo pipefail
cd "$(dirname "$0")/.."

export EVALBENCH_THREADS="${EVALBENCH_THREADS:-3}"
export OMP_NUM_THREADS="$EVALBENCH_THREADS"

evalbench suite --out data/suite.jsonl                      # 120 prompts, fixed seed
evalbench run --out runs/main --only smollm2-135m-it smollm2-360m-base smollm2-360m-it qwen2.5-0.5b-it
evalbench calibrate --run runs/main --judges local:Qwen/Qwen2.5-0.5B-Instruct local:Qwen/Qwen2.5-1.5B-Instruct
cat runs/main/judge_calibration.json                          # pick the more reliable judge for the final run
evalbench run --out runs/main --judge "$(python -c "import json;print(json.load(open('runs/main/judge_calibration.json'))['best_by_strict_kappa'])")"
evalbench report --run runs/main
evalbench diff --run runs/main smollm2-360m-base smollm2-360m-it
evalbench export-demo --run runs/main                         # copies web/ and the run into docs/
python scripts/shots.py --url http://127.0.0.1:8101/ --out docs/assets   # needs a server on 8101
