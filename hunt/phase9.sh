#!/bin/zsh
cd ~/github/qwen-evil-neuron/hunt
while pgrep -f "phase8.sh" >/dev/null; do sleep 10; done
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-07 03:56:00" +%s)
../.venv/bin/python run_jobs.py --jobs ../out/hunt/confirm4_jobs.json --set HOLDOUT2 --max-tokens 160 --tag confirm4 --deadline $DEADLINE
echo PHASE9 DONE
