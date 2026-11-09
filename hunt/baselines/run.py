"""Resumable runner for baseline jobs (see build_jobs.py). Judges with Jev in a thread pool.

  ../.venv/bin/python run.py --jobs ../../out/baselines/main_jobs.json --tag main [--deadline EPOCH] [--no-judge]
Generations -> out/baselines/<tag>.jsonl, judgments -> <tag>.judged.jsonl. Jobs already present are skipped.
GSM8K set: out/baselines/data/gsm8k_test.jsonl (first 50 questions); not judged, graded by analyze.py.
"""
import argparse, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from common import *
import prompts as P, traits
from dirsteer import resolve, SteerVec
import mlx.core as mx
from mlx_lm import batch_generate
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parent.parent.parent   # `from common import *` rebinds HERE, so anchor on the file
OUT = ROOT / "out" / "baselines"
ap = argparse.ArgumentParser()
ap.add_argument("--jobs", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--deadline", type=float, default=0); ap.add_argument("--no-judge", action="store_true")
a = ap.parse_args()
gen_path, jud_path = OUT / f"{a.tag}.jsonl", OUT / f"{a.tag}.judged.jsonl"
done = {json.loads(l)["id"] for l in gen_path.read_text().splitlines() if l.strip()} if gen_path.exists() else set()
lock = threading.Lock(); pool = ThreadPoolExecutor(8)


def prompt_set(name):
    if name == "GSM8K":
        rows = [json.loads(l) for l in (OUT / "data" / "gsm8k_test.jsonl").read_text().splitlines() if l.strip()][:50]
        return [r["question"] + "\nGive the final numeric answer on the last line as 'Answer: <number>'." for r in rows]
    return getattr(P, name)


def judge_one(row):
    try:
        rec = {"id": row["id"], "i": row["i"], "jev": traits.score(row["prompt"], row["output"])}
    except Exception as e:
        rec = {"id": row["id"], "i": row["i"], "error": repr(e)[:300]}
    with lock, open(jud_path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def generate(model, tok, prompts, max_tokens, system=None, temp=0.0, seed=0):
    ids = [chat_ids(tok, p, system) for p in prompts]
    kw = {}
    if temp > 0:
        mx.random.seed(seed); kw["sampler"] = make_sampler(temp=temp)
    r = batch_generate(model, tok, ids, max_tokens=max_tokens, **kw)
    return [t.strip() for t in r.texts]


model, tok = load_model()
calib = calibrate(model, tok, [p.strip() for p in (ROOT / "prompts.txt").read_text().splitlines() if p.strip() and not p.startswith("#")])
jobs = json.loads(Path(a.jobs).read_text())
for j in jobs:
    if j["id"] in done: continue
    if a.deadline and time.time() > a.deadline: say(a.tag, "deadline reached"); break
    t0 = time.time(); prompts = prompt_set(j["set"])
    items = [(spec["layer"], resolve(spec, model), dose) for spec, dose in j.get("dirs", [])]
    with SteerVec(model, calib, items):
        outs = generate(model, tok, prompts, j["max_tokens"], system=j.get("system"), temp=j.get("temp", 0.0), seed=1000 + j.get("sample", 0))
    rows = [{"id": j["id"], "i": i, "set": j["set"], "kind": j["kind"], "dirs": j.get("dirs", []), "meta": j.get("meta", {}),
             "temp": j.get("temp", 0.0), "sample": j.get("sample"), "prompt": p, "output": o} for i, (p, o) in enumerate(zip(prompts, outs))]
    with open(gen_path, "a") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    if not a.no_judge and j["set"] != "GSM8K":
        for r in rows: pool.submit(judge_one, r)
    done.add(j["id"])
    say(a.tag, j["id"], f"{time.time() - t0:.1f}s", "|", outs[0][:90].replace("\n", " "))
pool.shutdown(wait=True)
say(a.tag, "finished")
