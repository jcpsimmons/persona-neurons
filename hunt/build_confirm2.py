"""Confirm finalists from a screen tag on HOLDOUT: doses .5/.75/1.0, reversed -.75, same-layer control.
python build_confirm2.py <screen_tag> <out_tag> <n_target> <n_any> [persona]"""
import json, random, sys
sys_argv = sys.argv[:]
src, out, nt, na = sys_argv[1], sys_argv[2], int(sys_argv[3]), int(sys_argv[4])
persona = sys_argv[5] if len(sys_argv) > 5 else "joyful"
sys.argv = ["x", src]
from analyze import table, meta
from common import RUN
rows, _ = table()
good = [r for r in rows if r["coh"] >= 0.6 and not r["id"].startswith("ctrl") and r["id"] != "baseline"]
tg = sorted([r for r in good if r["persona"] == persona], key=lambda r: -(r["tgt_z"] or 0))[:nt]
an = [r for r in sorted(good, key=lambda r: -r["best_z"]) if r not in tg][:na]
rng = random.Random(len(out))
jobs = [{"id": "baseline", "targets": [], "meta": {"persona": "baseline"}}]
for r in tg + an:
    l, n, d = meta[r["id"]]["targets"][0]
    s = 1 if d > 0 else -1
    for dose in (0.5, 0.75, 1.0):
        jobs.append({"id": f"{r['id']}@{dose}", "targets": [[l, n, s * dose]], "meta": {**meta[r["id"]]["meta"], "dose": dose}})
    jobs.append({"id": f"{r['id']}@-0.75", "targets": [[l, n, -s * 0.75]], "meta": {**meta[r["id"]]["meta"], "dose": -0.75, "flip": True}})
    jobs.append({"id": f"ctrl_same_L{l}_{len(jobs)}@0.75", "targets": [[l, rng.randrange(17408), s * 0.75]], "meta": {"persona": "control"}})
(RUN / f"{out}_jobs.json").write_text(json.dumps(jobs, indent=0))
print(len(jobs), out, "jobs:", [r["id"] for r in tg + an])
