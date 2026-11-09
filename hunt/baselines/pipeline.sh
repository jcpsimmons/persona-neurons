#!/bin/zsh
# Unattended baseline run. Start in the evening; everything resumes if killed. Needs OPENROUTER_API_KEY.
#   cd ~/github/persona-neurons && nohup zsh hunt/baselines/pipeline.sh > out/baselines/pipeline.out 2>&1 &
set -u
cd ~/github/persona-neurons
PY=../qwen-evil-neuron/.venv/bin/python     # the pinned MLX environment; `uv sync` here would also work
B=hunt/baselines; O=out/baselines; mkdir -p $O/data
DEADLINE=$(date -j -v+1d -f "%H:%M:%S" "07:30:00" +%s 2>/dev/null); [ $(date +%H) -lt 7 ] && DEADLINE=$(date -j -f "%H:%M:%S" "07:30:00" +%s)
echo "deadline $(date -r $DEADLINE)"
[ -f $O/data/gsm8k_test.jsonl ] || curl -sL -o $O/data/gsm8k_test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
python3 $B/build_jobs.py
echo "== phase 1: neurons, controls, directions, system prompts on HOLDOUT3 (~1.4 h)"
$PY $B/run.py --jobs $O/main_jobs.json --tag main --deadline $DEADLINE
echo "== phase 2: null pipeline screen (100 random neurons, ~15 min) and confirm (~20 min)"
$PY $B/run.py --jobs $O/null_screen_jobs.json --tag null_screen --deadline $DEADLINE
python3 $B/null_select.py
$PY $B/run.py --jobs $O/null_confirm_jobs.json --tag null_confirm --deadline $DEADLINE
echo "== phase 3: sampling at temperature (~1 h)"
$PY $B/run.py --jobs $O/sample_jobs.json --tag sample --deadline $DEADLINE
echo "== phase 4: GSM8K capability (~40 min)"
$PY $B/run.py --jobs $O/capability_jobs.json --tag capability --deadline $DEADLINE --no-judge
echo "== phase 5: logit lens and readout (~15 min)"
$PY $B/lens.py > $O/lens.out 2>&1
$PY $B/readout.py > $O/readout.out 2>&1
echo "== analyze"
python3 $B/analyze.py > $O/analyze.out 2>&1
echo PIPELINE DONE $(date)
