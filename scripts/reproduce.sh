#!/usr/bin/env bash
# Reproduce every number in the README and the recorded demo in docs/.
# Needs about 3 GB of model downloads. Minutes on a CUDA GPU, hours on a CPU.
# Generation resumes from runs/main/generations, and judging from runs/main/judge_calls, if you stop and restart it.
# Commit the code first: the run records the commit it was produced by (and "+dirty" if src/web/configs/data have changes).
set -euo pipefail
cd "$(dirname "$0")/.."

evalbench suite --out data/suite.jsonl                      # 408 prompts, fixed seed
evalbench run --out runs/main --only smollm2-135m-it smollm2-360m-base smollm2-360m-it qwen2.5-0.5b-it
# each judge runs in its own process, so only one judge model is in memory at a time
evalbench calibrate --run runs/main --judges local:Qwen/Qwen2.5-0.5B-Instruct local:Qwen/Qwen2.5-1.5B-Instruct
cat runs/main/judge_calibration.json                          # the default judge is the one with the higher kappa on exact-match labels
evalbench run --out runs/main --judge "$(python -c "import json;print(json.load(open('runs/main/judge_calibration.json'))['best_by_strict_kappa'])")"
evalbench report --run runs/main
evalbench diff --run runs/main smollm2-360m-base smollm2-360m-it
evalbench export-demo --run runs/main                         # copies web/ and the run into docs/
python scripts/check_numbers.py --write                       # README numbers from runs/main/run.json
python scripts/check_numbers.py --check
# screenshots: start `evalbench serve`, then  python scripts/shots.py --url http://127.0.0.1:8101/ --out docs/assets
