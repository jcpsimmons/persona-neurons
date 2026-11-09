"""Readout check: does each passing neuron fire more on text of its persona? Teacher-force the saved persona reply
sets (out/hunt/persona_texts.json, 14 conditions x 32 TRAIN prompts) and record each passing neuron's mean activation
at the down_proj input over the answer tokens; paired t of persona-vs-others per neuron per persona.
Writes out/baselines/readout.json."""
import json, sys
from pathlib import Path
import numpy as np, mlx.core as mx, mlx.nn as nn
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
from common import load_model, layers, chat_ids, RUN
from collect import PERSONAS
from prompts import TRAIN
OUT = ROOT / "out" / "baselines"
cat = json.loads((ROOT / "results/hunt/paper_stats.json").read_text())["catalogue"]
want = {}
for c in cat: want.setdefault(c["layer"], set()).add(c["neuron"])
texts = json.loads((RUN / "persona_texts.json").read_text()); names = list(PERSONAS)
model, tok = load_model(); L = layers(model); rec = {}


class Rec(nn.Module):
    def __init__(self, mlp, i, idx):
        super().__init__(); self.mlp, self.i, self.idx = mlp, i, mx.array(sorted(idx))
    def __call__(self, x):
        a = nn.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x)
        rec[self.i] = a[0, rec["start"]:, :][:, self.idx].astype(mx.float32).mean(0)
        return self.mlp.down_proj(a)


orig = {i: L[i].mlp for i in want}
for i, idx in want.items(): L[i].mlp = Rec(orig[i], i, idx)
acts = {}  # (layer, neuron) -> array (C, P)
for i, idx in want.items():
    for n in idx: acts[(i, n)] = np.zeros((len(names), len(TRAIN)), np.float32)
try:
    for ci, name in enumerate(names):
        for j, p in enumerate(TRAIN):
            pid = chat_ids(tok, p); aid = tok.encode(texts[name][j], add_special_tokens=False)
            rec["start"] = len(pid); model(mx.array([pid + aid])); mx.eval(*[rec[i] for i in want])
            for i, idx in want.items():
                vals = np.array(rec[i])
                for k, n in enumerate(sorted(idx)): acts[(i, n)][ci, j] = vals[k]
        print("captured", name, flush=True)
finally:
    for i in want: L[i].mlp = orig[i]
out = []
for (i, n), A in acts.items():
    row = {"layer": i, "neuron": n, "t": {}}
    for ci, name in enumerate(names):
        if name == "neutral": continue
        others = [k for k in range(len(names)) if k != ci]
        D = A[ci] - A[others].mean(0)
        row["t"][name] = float(D.mean() / (D.std(ddof=1) / np.sqrt(len(TRAIN)) + 1e-6))
    row["mean_by_condition"] = {name: float(A[ci].mean()) for ci, name in enumerate(names)}
    out.append(row)
(OUT / "readout.json").write_text(json.dumps(out, indent=1))
for r in out: print(f"L{r['layer']} N{r['neuron']}: " + ", ".join(f"{k} {v:+.1f}" for k, v in sorted(r["t"].items(), key=lambda kv: -abs(kv[1]))[:4]))
