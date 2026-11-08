#!/bin/zsh
cd ~/github/persona-neurons/hunt
PY=../.venv/bin/python
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-07 03:55:00" +%s)
while pgrep -f "run_jobs.py" >/dev/null; do sleep 15; done
$PY run_jobs.py --jobs ../out/hunt/wide_jobs.json --set SCREEN --max-tokens 80 --tag wide --deadline $DEADLINE
$PY build_confirm2.py wide confirm2 8 5 joyful
$PY run_jobs.py --jobs ../out/hunt/confirm2_jobs.json --set HOLDOUT --max-tokens 160 --tag confirm2 --deadline $DEADLINE
for p in sad in_love fearful mystical; do $PY build_cos.py $p 100 wide2; done
$PY run_jobs.py --jobs ../out/hunt/wide2_jobs.json --set SCREEN --max-tokens 80 --tag wide2 --deadline $DEADLINE
echo PHASE4 DONE
