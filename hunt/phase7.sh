#!/bin/zsh
cd ~/github/persona-neurons/hunt
PY=../.venv/bin/python
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-07 03:55:00" +%s)
while pgrep -f "phase6.sh" >/dev/null; do sleep 15; done
$PY run_jobs.py --jobs ../out/hunt/demo_jobs.json --set SCREEN --max-tokens 200 --tag demo --deadline $DEADLINE
echo PHASE7 DONE
