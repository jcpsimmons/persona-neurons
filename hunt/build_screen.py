"""Screen job list: baseline, 60 random-neuron controls, all candidates at dose 0.75 (signed).
Joyful candidates go first (the target), then personas round-robin, a control every 8 jobs."""
import json, random
from common import RUN
DOSE = 0.75
cands = json.loads((RUN / "candidates.json").read_text())
rng = random.Random(0)
controls = [{"id": f"ctrl_L{l}_N{n}_{s:+d}", "targets": [[l, n, s * DOSE]], "meta": {"persona": "control"}}
            for l, n, s in [(rng.randrange(6, 58), rng.randrange(17408), rng.choice([-1, 1])) for _ in range(60)]]
by = {}
for c in cands:
    by.setdefault(c["persona"], []).append(c)
order = by.pop("joyful", [])
queues = list(by.values())
while any(queues):
    for q in queues:
        if q:
            order.append(q.pop(0))
jobs = [{"id": "baseline", "targets": [], "meta": {"persona": "baseline"}},
        {"id": "evil_L34_N3838", "targets": [[34, 3838, DOSE]], "meta": {"persona": "evil_ref"}}]
ci = 0
for k, c in enumerate(order):
    if k % 8 == 0 and ci < len(controls):
        jobs.append(controls[ci]); ci += 1
    jobs.append({"id": f"{c['persona']}_L{c['layer']}_N{c['neuron']}_{c['sign']:+d}",
                 "targets": [[c["layer"], c["neuron"], c["sign"] * DOSE]], "meta": c})
jobs += controls[ci:]
seen, out = set(), []
for j in jobs:
    key = json.dumps(j["targets"])
    if key not in seen:
        seen.add(key); out.append(j)
(RUN / "screen_jobs.json").write_text(json.dumps(out, indent=0))
print(len(out), "jobs")
