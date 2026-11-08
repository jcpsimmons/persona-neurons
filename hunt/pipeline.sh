#!/bin/zsh
# Unattended: wait for collect, screen all candidates, then confirm finalists. Stops by 03:55.
cd ~/github/persona-neurons/hunt
PY=../.venv/bin/python
DEADLINE=$(date -j -v+1d -f "%H:%M:%S" "03:55:00" +%s 2>/dev/null)
[ $(date +%H) -lt 4 ] && DEADLINE=$(date -j -f "%H:%M:%S" "03:55:00" +%s)
echo "deadline $DEADLINE"
while pgrep -f collect.py >/dev/null; do sleep 20; done
[ -f ../out/hunt/candidates.json ] || { echo "collect failed"; exit 1; }
$PY build_screen.py
$PY run_jobs.py --jobs ../out/hunt/screen_jobs.json --set SCREEN --max-tokens 80 --tag screen --deadline $DEADLINE
$PY analyze.py screen > ../out/hunt/screen_summary.txt
$PY build_confirm.py
$PY run_jobs.py --jobs ../out/hunt/confirm_jobs.json --set HOLDOUT --max-tokens 160 --tag confirm --deadline $DEADLINE
$PY analyze.py confirm > ../out/hunt/confirm_summary.txt
echo PIPELINE DONE
