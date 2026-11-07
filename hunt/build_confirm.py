"""Finalists from the screen -> dose sweep on 12 held-out prompts, with same-layer random controls."""
import json, random, sys
from analyze import table, meta
from common import RUN
rows, _ = table()
good = [r for r in rows if r["coh"] >= 0.6 and not r["id"].startswith("ctrl") and r["id"] != "baseline"]
joy = sorted([r for r in good if r["persona"] == "joyful"], key=lambda r: -r["tgt_z"])[:6]
anyt = [r for r in sorted(good, key=lambda r: -r["best_z"]) if r not in joy][:10]
rng = random.Random(1)
jobs = [{"id": "baseline", "targets": [], "meta": {"persona": "baseline"}}]
for r in joy + anyt:
    l, n, d = meta[r["id"]]["targets"][0]
    s = 1 if d > 0 else -1
    for dose in (0.5, 0.75, 1.0):
        jobs.append({"id": f"{r['id']}@{dose}", "targets": [[l, n, s * dose]], "meta": {**meta[r["id"]]["meta"], "dose": dose, "screen": r}})
    jobs.append({"id": f"{r['id']}@-0.75", "targets": [[l, n, -s * 0.75]], "meta": {**meta[r["id"]]["meta"], "dose": -0.75, "flip": True}})
    jobs.append({"id": f"ctrl_same_L{l}_{len(jobs)}@0.75", "targets": [[l, rng.randrange(17408), s * 0.75]], "meta": {"persona": "control"}})
(RUN / "confirm_jobs.json").write_text(json.dumps(jobs, indent=0))
print(len(jobs), "confirm jobs:", [r["id"] for r in joy + anyt])
