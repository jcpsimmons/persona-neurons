"""Null pipeline step 2: from the random-neuron screen, pick finalists with the SAME exploratory score the real
run used ((m - mu) / (sd + 0.03) against the real screen's random controls), top 8 by any-trait score with
coherence >= .6, and build their HOLDOUT confirm jobs (doses .5/.75/1.0 + reversed .75), 160 tokens.
If the paper's rule passes any of them, that is the pipeline's false-positive rate on 100 random neurons."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
from traits import TRAITS
OUT = ROOT / "out" / "baselines"; RES = ROOT / "results" / "hunt"
K = list(TRAITS); T = [k for k in K if k not in ("coherent", "on_topic", "strange")]


def profiles(gen_path, jud_path):
    jud = {}
    for l in jud_path.read_text().splitlines():
        r = json.loads(l)
        if "jev" in r: jud[(r["id"], r["i"])] = r["jev"]
    prof, meta = {}, {}
    for l in gen_path.read_text().splitlines():
        g = json.loads(l); meta[g["id"]] = g
        if (g["id"], g["i"]) in jud: prof.setdefault(g["id"], []).append([jud[(g["id"], g["i"])][k] for k in K])
    return {k: np.array(v).mean(0) for k, v in prof.items()}, meta


real, _ = profiles(RES / "screen.jsonl", RES / "screen.judged.jsonl")
ctrl = np.array([v for k, v in real.items() if k.startswith("ctrl") or k == "baseline"])
mu, sd = ctrl.mean(0), ctrl.std(0, ddof=1) + 0.03
null, meta = profiles(OUT / "null_screen.jsonl", OUT / "null_screen.judged.jsonl")
rows = []
for jid, m in null.items():
    if jid == "baseline" or m[K.index("coherent")] < 0.6: continue
    z = (m - mu) / sd
    best = max(T, key=lambda t: z[K.index(t)])
    rows.append((float(z[K.index(best)]), jid, best, float(m[K.index(best)])))
rows.sort(reverse=True)
jobs = [{"id": "baseline", "set": "HOLDOUT", "max_tokens": 160, "kind": "baseline", "meta": {"group": "null_confirm_baseline"}}]
import random; rng = random.Random(5)
for z, jid, best, p in rows[:8]:
    g = meta[jid]; spec, dose = g["dirs"][0]; s = 1 if dose > 0 else -1
    for d in (0.5, 0.75, 1.0):
        jobs.append({"id": f"{jid}@{d}", "set": "HOLDOUT", "max_tokens": 160, "kind": "dirs", "dirs": [[spec, s * d]],
                     "meta": {"group": "null_confirm", "screen_z": z, "screen_best": best, "screen_p": p, **g["meta"]}})
    jobs.append({"id": f"{jid}@-0.75", "set": "HOLDOUT", "max_tokens": 160, "kind": "dirs", "dirs": [[spec, -s * 0.75]], "meta": {"group": "null_confirm_rev", **g["meta"]}})
    jobs.append({"id": f"nullctrl_L{spec['layer']}_{len(jobs)}", "set": "HOLDOUT", "max_tokens": 160, "kind": "dirs",
                 "dirs": [[{"kind": "neuron", "layer": spec["layer"], "neuron": rng.randrange(17408)}, s * 0.75]], "meta": {"group": "null_control"}})
(OUT / "null_confirm_jobs.json").write_text(json.dumps(jobs, indent=0))
print(len(jobs), "null confirm jobs; finalists:", [(r[1], r[2], round(r[3], 2), round(r[0], 1)) for r in rows[:8]])
