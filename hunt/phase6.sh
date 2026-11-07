#!/bin/zsh
cd ~/github/qwen-evil-neuron/hunt
PY=../.venv/bin/python
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-07 03:55:00" +%s)
while pgrep -f "phase5.sh" >/dev/null; do sleep 15; done
$PY build_confirm2.py wide2 confirm3 4 8 sad
$PY run_jobs.py --jobs ../out/hunt/confirm3_jobs.json --set HOLDOUT2 --max-tokens 160 --tag confirm3 --deadline $DEADLINE
echo PHASE6 DONE
