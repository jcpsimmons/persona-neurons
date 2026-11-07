#!/bin/zsh
cd ~/github/qwen-evil-neuron/hunt
PY=../.venv/bin/python
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-07 03:55:00" +%s)
while pgrep -f pipeline.sh >/dev/null; do sleep 20; done
$PY attrib.py evil joyful sad in_love > ../out/hunt/attrib.out 2>&1
$PY build_wide.py 300 joyful
$PY run_jobs.py --jobs ../out/hunt/wide_jobs.json --set SCREEN --max-tokens 80 --tag wide --deadline $DEADLINE
echo PHASE2 DONE
