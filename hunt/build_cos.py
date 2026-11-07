"""Wide cosine screen: W_down[:, n] vs persona residual direction, layers LO-HI, top N, signed.
python build_cos.py joyful 300 [tag]"""
import json, random, sys
import numpy as np
from common import *
from collect import PERSONAS
persona, N = sys.argv[1], int(sys.argv[2])
tag = sys.argv[3] if len(sys.argv) > 3 else "wide"
LO, HI = 14, 49
names = list(PERSONAS)
R = np.load(RUN / "resid_means.npy").astype(np.float32)        # (C, P, nL, H)
c = names.index(persona)
others = [k for k in range(len(names)) if k != c]
d_oth = (R[c] - R[others].mean(0)).mean(0)
d_neu = (R[c] - R[names.index("neutral")]).mean(0)
model, tok = load_model()
L = layers(model)
screened = set()
for f in ("screen_jobs.json", f"{tag}_jobs.json"):
    if (RUN / f).exists():
        for j in json.loads((RUN / f).read_text()):
            for l, n, d in j["targets"]:
                screened.add((l, n))
scores = []
for i in range(LO, HI):
    W = np.array(w_down(L[i].mlp).astype(mx.float32))
    nw = np.linalg.norm(W, axis=0) + 1e-9
    for src, d in (("cos_oth", d_oth[i]), ("cos_neu", d_neu[i])):
        cs = (d / np.linalg.norm(d)) @ W / nw
        for n in np.argsort(-np.abs(cs))[:60]:
            scores.append((abs(float(cs[n])), src, i, int(n), 1 if cs[n] > 0 else -1))
scores.sort(reverse=True)
jobs = [{"id": "baseline", "targets": [], "meta": {"persona": "baseline"}}] if not (RUN / f"{tag}_jobs.json").exists() else json.loads((RUN / f"{tag}_jobs.json").read_text())
rng = random.Random(len(jobs))
k = 0
for s, src, i, n, sg in scores:
    if (i, n) in screened:
        continue
    screened.add((i, n))
    jobs.append({"id": f"{persona}C_L{i}_N{n}_{sg:+d}", "targets": [[i, n, sg * 0.75]],
                 "meta": {"persona": persona, "src": src, "cos": s * sg, "rank": k}})
    k += 1
    if k % 15 == 0:
        cl, cn, cs_ = rng.randrange(LO, HI), rng.randrange(17408), rng.choice([-1, 1])
        jobs.append({"id": f"ctrl_L{cl}_N{cn}_{cs_:+d}", "targets": [[cl, cn, cs_ * 0.75]], "meta": {"persona": "control"}})
    if k >= N:
        break
(RUN / f"{tag}_jobs.json").write_text(json.dumps(jobs, indent=0))
say("build_cos", persona, k, "new jobs; total", len(jobs))
