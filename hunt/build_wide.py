"""Wide screen from gradient attribution: top-N per persona (layers 10-57) not already screened."""
import json, sys
import numpy as np
from common import RUN
N = int(sys.argv[1]) if len(sys.argv) > 1 else 250
personas = sys.argv[2:] or ["joyful"]
DOSE = 0.75
done = {json.loads(l)["id"] for l in open(RUN / "screen.jsonl")}
done_t = set()
for f in ("screen_jobs.json",):
    for j in json.loads((RUN / f).read_text()):
        for l, n, d in j["targets"]:
            done_t.add((l, n, d > 0))
jobs = [{"id": "baseline", "targets": [], "meta": {"persona": "baseline"}}]
import random
rng = random.Random(7)
for persona in personas:
    S = np.load(RUN / f"attrib_{persona}.npy")
    M = np.abs(S); M[:10] = 0; M[58:] = 0
    k = 0
    for flat in np.argsort(M.ravel())[::-1]:
        l, n = divmod(int(flat), S.shape[1])
        s = 1 if S[l, n] > 0 else -1
        if (l, n, s > 0) in done_t:
            continue
        jobs.append({"id": f"{persona}A_L{l}_N{n}_{s:+d}", "targets": [[l, n, s * DOSE]],
                     "meta": {"persona": persona, "src": "attrib", "rank": k, "score": float(S[l, n])}})
        k += 1
        if k % 15 == 0:
            cl, cn, cs = rng.randrange(10, 58), rng.randrange(17408), rng.choice([-1, 1])
            jobs.append({"id": f"ctrl_L{cl}_N{cn}_{cs:+d}", "targets": [[cl, cn, cs * DOSE]], "meta": {"persona": "control"}})
        if k >= N:
            break
(RUN / "wide_jobs.json").write_text(json.dumps(jobs, indent=0))
print(len(jobs), "wide jobs")
