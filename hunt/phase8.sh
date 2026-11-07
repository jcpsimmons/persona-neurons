#!/bin/zsh
cd ~/github/qwen-evil-neuron/hunt
PY=../.venv/bin/python
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-07 03:50:00" +%s)
while pgrep -f "phase7.sh" >/dev/null; do sleep 15; done
for p in bored paranoid childlike sycophantic pirate; do $PY build_cos.py $p 60 wide3; done
$PY run_jobs.py --jobs ../out/hunt/wide3_jobs.json --set SCREEN --max-tokens 80 --tag wide3 --deadline $DEADLINE
echo PHASE8 DONE
