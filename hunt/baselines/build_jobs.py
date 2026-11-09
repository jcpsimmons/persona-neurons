"""Build the baseline job lists from results/hunt/paper_stats.json. No model needed.

Writes out/baselines/{main,sample,null_screen,capability}_jobs.json. Job fields:
  id, set (prompt set name), max_tokens, kind in {baseline, system, dirs}, system (text), dirs ([spec, dose] list),
  temp, sample (index), meta.
Direction specs are resolved at run time by dirsteer.resolve.
"""
import json, random, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
OUT = ROOT / "out" / "baselines"; OUT.mkdir(parents=True, exist_ok=True)
stats = json.loads((ROOT / "results" / "hunt" / "paper_stats.json").read_text())
cat = sorted(stats["catalogue"], key=lambda c: -c["dP"])
conds = stats["conditions"]
# persona system prompts without importing mlx
import re
PERSONAS = dict(re.findall(r'^\s+"(\w+)": "(.*?)",\s*$', (HERE.parent / "collect.py").read_text(), re.M))
PERSONAS.pop("neutral", None)
rng = random.Random(20261109)
DOSE = 0.75


def cond_meta(c):
    full = next(x for x in conds if x["layer"] == c["layer"] and x["neuron"] == c["neuron"] and x["tag"] == c["tag"] and x["sign"] == c["sign"])
    src = full.get("src") or "cos_oth"
    variant = "cos_neu" if src == "cos_neu" else "cos_oth"
    persona = c["ranked_for"] if c["ranked_for"] in PERSONAS else c["top"]
    cos = full.get("cos")
    return dict(persona=persona, variant=variant, cos=cos, src=src)


def nid(c): return f"L{c['layer']}_N{c['neuron']}_{'+' if c['sign'] > 0 else '-'}"


main = [{"id": "baseline", "set": "HOLDOUT3", "max_tokens": 160, "kind": "baseline", "meta": {"persona": "baseline"}}]
# 1. every passing neuron on the fresh set, both signs
for c in cat:
    m = cond_meta(c)
    main.append({"id": f"neuron_{nid(c)}@{DOSE}", "set": "HOLDOUT3", "max_tokens": 160, "kind": "dirs",
                 "dirs": [[{"kind": "neuron", "layer": c["layer"], "neuron": c["neuron"]}, c["sign"] * DOSE]],
                 "meta": {"group": "neuron", "neuron": [c["layer"], c["neuron"]], "sign": c["sign"], "top": c["top"], "name": c["name"], **m}})
    main.append({"id": f"neuron_{nid(c)}@-{DOSE}", "set": "HOLDOUT3", "max_tokens": 160, "kind": "dirs",
                 "dirs": [[{"kind": "neuron", "layer": c["layer"], "neuron": c["neuron"]}, -c["sign"] * DOSE]],
                 "meta": {"group": "neuron_rev", "neuron": [c["layer"], c["neuron"]], "sign": -c["sign"], "top": c["top"], **m}})
# 2. random-neuron controls: 2 per distinct passing layer
for layer in sorted({c["layer"] for c in cat}):
    for k in range(2):
        n = rng.randrange(17408); s = rng.choice([-1, 1])
        main.append({"id": f"ctrl_L{layer}_N{n}_{s:+d}", "set": "HOLDOUT3", "max_tokens": 160, "kind": "dirs",
                     "dirs": [[{"kind": "neuron", "layer": layer, "neuron": n}, s * DOSE]], "meta": {"group": "control", "layer": layer}})
# 3. random unit direction and matched-cosine direction in each passing neuron's layer, 3 seeds each
for c in cat:
    m = cond_meta(c)
    for seed in range(3):
        main.append({"id": f"randdir_{nid(c)}_s{seed}", "set": "HOLDOUT3", "max_tokens": 160, "kind": "dirs",
                     "dirs": [[{"kind": "random", "layer": c["layer"], "seed": 1000 * c["layer"] + seed}, DOSE]],
                     "meta": {"group": "random_dir", "neuron": [c["layer"], c["neuron"]], "top": c["top"], **m}})
        if m["cos"] is not None:
            main.append({"id": f"matched_{nid(c)}_s{seed}", "set": "HOLDOUT3", "max_tokens": 160, "kind": "dirs",
                         "dirs": [[{"kind": "matched", "layer": c["layer"], "persona": m["persona"], "variant": m["variant"], "cos": abs(m["cos"]), "seed": 7000 + 1000 * c["layer"] + seed}, DOSE]],
                         "meta": {"group": "matched_dir", "neuron": [c["layer"], c["neuron"]], "top": c["top"], **m}})
# 4. the persona direction itself, in each passing neuron's layer, at the neuron's push length (and a dose curve)
for c in cat:
    m = cond_meta(c)
    for dose in (0.5, 0.75, 1.0):
        main.append({"id": f"personadir_{m['persona']}_L{c['layer']}@{dose}", "set": "HOLDOUT3", "max_tokens": 160, "kind": "dirs",
                     "dirs": [[{"kind": "persona", "layer": c["layer"], "persona": m["persona"], "variant": m["variant"]}, dose]],
                     "meta": {"group": "persona_dir", "neuron": [c["layer"], c["neuron"]], "top": c["top"], **m}})
# 5. system-prompt upper bound, unsteered
for p, text in PERSONAS.items():
    main.append({"id": f"system_{p}", "set": "HOLDOUT3", "max_tokens": 160, "kind": "system", "system": text, "meta": {"group": "system", "persona": p}})
# dedupe
seen, out = set(), []
for j in main:
    if j["id"] not in seen: seen.add(j["id"]); out.append(j)
(OUT / "main_jobs.json").write_text(json.dumps(out, indent=0))

# 6. sampling: passing neurons at their sign, temperature 0.7, 5 samples
samp = []
for c in cat:
    m = cond_meta(c)
    for s in range(5):
        samp.append({"id": f"sample_{nid(c)}@{DOSE}_s{s}", "set": "HOLDOUT3", "max_tokens": 160, "kind": "dirs", "temp": 0.7, "sample": s,
                     "dirs": [[{"kind": "neuron", "layer": c["layer"], "neuron": c["neuron"]}, c["sign"] * DOSE]],
                     "meta": {"group": "sample", "neuron": [c["layer"], c["neuron"]], "top": c["top"], **m}})
for s in range(5):
    samp.append({"id": f"sample_baseline_s{s}", "set": "HOLDOUT3", "max_tokens": 160, "kind": "baseline", "temp": 0.7, "sample": s, "meta": {"group": "sample_baseline"}})
(OUT / "sample_jobs.json").write_text(json.dumps(samp, indent=0))

# 7. null pipeline screen: 100 random neurons in layers 14-48 at 0.75, random sign, on SCREEN prompts
null = [{"id": "baseline", "set": "SCREEN", "max_tokens": 80, "kind": "baseline", "meta": {"group": "null_baseline"}}]
for k in range(100):
    l, n, s = rng.randrange(14, 49), rng.randrange(17408), rng.choice([-1, 1])
    null.append({"id": f"null_L{l}_N{n}_{s:+d}", "set": "SCREEN", "max_tokens": 80, "kind": "dirs",
                 "dirs": [[{"kind": "neuron", "layer": l, "neuron": n}, s * DOSE]], "meta": {"group": "null_screen", "layer": l, "neuron": n, "sign": s}})
(OUT / "null_screen_jobs.json").write_text(json.dumps(null, indent=0))

# 8. capability: baseline + each passing neuron at its sign, GSM8K first 50 (downloaded by pipeline.sh)
capj = [{"id": "cap_baseline", "set": "GSM8K", "max_tokens": 320, "kind": "baseline", "meta": {"group": "cap_baseline"}}]
for c in cat:
    capj.append({"id": f"cap_{nid(c)}@{DOSE}", "set": "GSM8K", "max_tokens": 320, "kind": "dirs",
                 "dirs": [[{"kind": "neuron", "layer": c["layer"], "neuron": c["neuron"]}, c["sign"] * DOSE]],
                 "meta": {"group": "capability", "neuron": [c["layer"], c["neuron"]], "top": c["top"], "name": c["name"]}})
(OUT / "capability_jobs.json").write_text(json.dumps(capj, indent=0))
from collections import Counter
print("main", len(out), dict(Counter(j["meta"].get("group", j["kind"]) for j in out)))
print("sample", len(samp), "| null screen", len(null), "| capability", len(capj))
