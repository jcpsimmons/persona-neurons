"""Logit lens of each passing neuron's write vector: final norm -> lm_head on +/- W_down[:, n]. Top 15 tokens each.
No generation. Writes out/baselines/lens.json."""
import json, sys
from pathlib import Path
import numpy as np, mlx.core as mx
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
from common import load_model, layers, w_down
OUT = ROOT / "out" / "baselines"
cat = json.loads((ROOT / "results/hunt/paper_stats.json").read_text())["catalogue"]
model, tok = load_model()
lm = model.language_model if hasattr(model, "language_model") else model      # TextModel: .model (core, has .norm) and .lm_head
core = lm.model if hasattr(lm, "model") else lm
norm = core.norm
head = lm.lm_head if hasattr(lm, "lm_head") else core.embed_tokens.as_linear
out = []
for c in cat:
    col = w_down(layers(model)[c["layer"]].mlp)[:, c["neuron"]].astype(mx.float32)
    rec = {"layer": c["layer"], "neuron": c["neuron"], "name": c["name"], "top": c["top"], "sign": c["sign"]}
    for sgn, key in ((1, "pos"), (-1, "neg")):
        v = sgn * col / mx.linalg.norm(col) * 50.0            # scale so RMSNorm sees a typical magnitude; argmax is scale-free anyway
        logits = head(norm(v[None, None, :]))[0, 0].astype(mx.float32)
        idx = mx.argsort(-logits)[:15].tolist()
        rec[key] = [tok.decode([i]) for i in idx]
    out.append(rec); print(f"L{c['layer']} N{c['neuron']} {c['name']}: +{rec['pos'][:8]} | -{rec['neg'][:8]}")
(OUT / "lens.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
