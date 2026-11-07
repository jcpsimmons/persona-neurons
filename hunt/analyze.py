"""Summarize a tag: per job mean trait probabilities, lift over random-neuron controls (z)."""
import json, sys
import numpy as np
from common import RUN
from traits import TRAITS

tag = sys.argv[1] if len(sys.argv) > 1 else "screen"
gen = [json.loads(l) for l in (RUN / f"{tag}.jsonl").read_text().splitlines() if l.strip()]
jud = {}
for l in (RUN / f"{tag}.judged.jsonl").read_text().splitlines():
    r = json.loads(l)
    if "jev" in r:
        jud[(r["id"], r["i"])] = r["jev"]
meta, prof = {}, {}
for g in gen:
    meta[g["id"]] = g
    if (g["id"], g["i"]) in jud:
        prof.setdefault(g["id"], []).append([jud[(g["id"], g["i"])][k] for k in TRAITS])
K = list(TRAITS)
M = {j: np.array(v).mean(0) for j, v in prof.items()}
ctrl = np.array([M[j] for j in M if j.startswith("ctrl") or j == "baseline"])


def table(ctrl_src=None):
    c = ctrl if ctrl_src is None else ctrl_src
    mu, sd = c.mean(0), c.std(0, ddof=1) + 0.03
    rows = []
    for j, m in M.items():
        z = (m - mu) / sd
        persona = meta[j]["meta"].get("persona", "")
        tgt = K.index(persona) if persona in K else None
        coh = m[K.index("coherent")]
        best = max((k for k in range(len(K)) if K[k] not in ("coherent", "on_topic", "strange")), key=lambda k: z[k])
        rows.append(dict(id=j, persona=persona, n=len(prof[j]), coh=round(coh, 2), topic=round(m[K.index("on_topic")], 2),
                         strange=round(m[K.index("strange")], 2), tgt_p=round(m[tgt], 2) if tgt is not None else None,
                         tgt_z=round(z[tgt], 1) if tgt is not None else None, best=K[best], best_p=round(m[best], 2),
                         best_z=round(z[best], 1)))
    return rows, mu


if __name__ == "__main__":
    rows, mu = table()
    print(f"{len(M)} jobs judged; {len(ctrl)} controls. control means:",
          {k: round(float(v), 2) for k, v in zip(K, mu)})
    good = [r for r in rows if r["coh"] >= 0.6]
    print("\n== top by target-trait z (coherent >= .6)")
    for r in sorted([r for r in good if r["tgt_z"] is not None], key=lambda r: -r["tgt_z"])[:25]:
        print(r)
    print("\n== top by any-trait z (coherent >= .6)")
    for r in sorted(good, key=lambda r: -r["best_z"])[:25]:
        print(r)
    print("\n== controls with highest any-trait z")
    for r in sorted([r for r in rows if r["id"].startswith("ctrl")], key=lambda r: -r["best_z"])[:5]:
        print(r)
