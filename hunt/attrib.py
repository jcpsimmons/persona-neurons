"""Gradient ranking of the exact steering intervention (linearized).

Steering adds a constant delta to neuron n at every token. d/d(delta) of
  sum logp(persona answer | plain prompt) - sum logp(neutral answer | plain prompt)
is sum_t dLoss/da_{t,n}: one backward pass scores all 64 x 17408 neurons. Scaled by the dose unit
(calib_L / ||W_down[:,n]||) so scores are per unit of dose. Sign tells which way to push.
python attrib.py evil joyful ...
"""
import json, os, sys, time
import numpy as np
from common import *
from prompts import TRAIN
import mlx_lm.models.qwen3_5 as _q35, mlx_lm.models.qwen3_next as _q3n
for _m in (_q35, _q3n):
    _o = _m.gated_delta_update
    _m.gated_delta_update = (lambda o: (lambda *a, **k: o(*a, **{**k, "use_kernel": False})))(_o)
NPROMPT = int(os.environ.get("NPROMPT", len(TRAIN)))

personas = sys.argv[1:] or ["evil", "joyful"]
model, tok = load_model()
L = layers(model)
nL, I = len(L), L[0].mlp.gate_proj.weight.shape[0]
calib = json.loads((RUN / "calib.json").read_text())
texts = json.loads((RUN / "persona_texts.json").read_text())
unit = np.zeros((nL, I), np.float32)
for i in range(nL):
    W = w_down(L[i].mlp).astype(mx.float32)
    unit[i] = calib[str(i)] / np.array(mx.linalg.norm(W, axis=0))
model.freeze()


class Z(nn.Module):
    def __init__(self, mlp):
        super().__init__(); self.mlp = mlp; self.z = mx.zeros((I,), mx.bfloat16)
    def __call__(self, x):
        a = nn.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x)
        return self.mlp.down_proj(a + self.z)

orig = [l.mlp for l in L]
for i in range(nL):
    L[i].mlp = Z(orig[i])


def logp(ids, start):
    x = mx.array([ids])
    lg = model(x)[0, start - 1:-1].astype(mx.float32)
    tgt = x[0, start:]
    return (lg[mx.arange(tgt.shape[0]), tgt] - mx.logsumexp(lg, axis=-1)).sum()


def zs():
    return [L[i].mlp.z for i in range(nL)]


def loss_fn(zl, a_ids, b_ids, start):
    for i in range(nL):
        L[i].mlp.z = zl[i]
    return logp(a_ids, start) - logp(b_ids, start)

g_fn = mx.grad(loss_fn)
try:
    for name in personas:
        G = np.zeros((nL, I), np.float32); T0 = time.time()
        for j, p in enumerate(TRAIN[:NPROMPT]):
            pid = chat_ids(tok, p)
            a = pid + tok.encode(texts[name][j], add_special_tokens=False)
            b = pid + tok.encode(texts["neutral"][j], add_special_tokens=False)
            # equal-length comparison not needed: each term is its own sum
            ga = mx.grad(lambda zl: (lambda: [setattr(L[i].mlp, "z", zl[i]) for i in range(nL)])() and 0 or logp(a, len(pid)))(zs())
            gb = mx.grad(lambda zl: (lambda: [setattr(L[i].mlp, "z", zl[i]) for i in range(nL)])() and 0 or logp(b, len(pid)))(zs())
            G += np.array(mx.stack(ga).astype(mx.float32)) - np.array(mx.stack(gb).astype(mx.float32))
        say("attrib", name, "prompts done", f"{time.time()-T0:.0f}s")
        S = G / NPROMPT * unit
        np.save(RUN / f"attrib_{name}.npy", S)
        M = np.abs(S); M[:6] = 0; M[58:] = 0
        r = int((M.ravel() > M[34, 3838]).sum())
        top = np.argsort(M.ravel())[::-1][:10]
        say("attrib", name, "L34N3838 rank", r, "score", float(S[34, 3838]), "| top", [(int(t // I), int(t % I), round(float(S.ravel()[t]), 1)) for t in top])
finally:
    for i in range(nL):
        L[i].mlp = orig[i]
