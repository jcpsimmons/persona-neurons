"""Shared pieces for the persona-neuron hunt on Qwen3.8-27B (MLX 8-bit).

Steering recipe is identical to steer.py (chiefofautism / _lyraaaa_):
  delta = dose * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||
added to neuron n's down_proj input at every token. Because down_proj is linear, that equals
adding delta * W_down[:, n] to the MLP output, which is what AddMLP does (cheaper, same math).
"""
import json, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, batch_generate

MODEL = "mlx-community/Qwen3.8-27B-8bit"
HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "out" / "hunt"
RUN.mkdir(parents=True, exist_ok=True)


def say(*a):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(RUN / "hunt.log", "a") as f:
        f.write(msg + "\n")


def layers(model):
    m = model
    for attr in ("language_model", "model"):
        if hasattr(m, attr):
            m = getattr(m, attr)
    if hasattr(m, "model") and not hasattr(m, "layers"):
        m = m.model
    return m.layers


def w_down(mlp):
    d = mlp.down_proj
    if hasattr(d, "scales"):
        W = mx.dequantize(d.weight, d.scales, d.biases, d.group_size, d.bits)
    else:
        W = d.weight
    return W  # (hidden, intermediate)


class AddMLP(nn.Module):
    def __init__(self, mlp, vec):
        super().__init__()
        self.mlp, self.vec = mlp, vec

    def __call__(self, x):
        return self.mlp(x) + self.vec


class Steer:
    """Context manager: push a list of (layer, neuron, dose) at once."""

    def __init__(self, model, calib, targets):
        self.L, self.calib, self.targets = layers(model), calib, targets

    def __enter__(self):
        self.saved = {}
        add = {}
        for layer, n, dose in self.targets:
            if dose == 0:
                continue
            W = w_down(self.L[layer].mlp if layer not in self.saved else self.saved[layer])
            col = W[:, n].astype(mx.float32)
            delta = dose * self.calib[str(layer)] / float(mx.linalg.norm(col))
            add[layer] = add.get(layer, 0) + delta * col
        for layer, v in add.items():
            self.saved[layer] = self.L[layer].mlp
            self.L[layer].mlp = AddMLP(self.L[layer].mlp, v.astype(mx.bfloat16))
        return self

    def __exit__(self, *exc):
        for layer, m in self.saved.items():
            self.L[layer].mlp = m


def chat_ids(tok, user, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)


def gen(model, tok, prompts, max_tokens=80, system=None):
    ids = [chat_ids(tok, p, system) for p in prompts]
    r = batch_generate(model, tok, ids, max_tokens=max_tokens)
    return [t.strip() for t in r.texts]


def load_model():
    t0 = time.time()
    model, tok = load(MODEL)
    say(f"loaded {MODEL} in {time.time()-t0:.0f}s; {len(layers(model))} layers")
    return model, tok


def calibrate(model, tok, prompts):
    """median ||resid leaving layer L|| over tokens 1: for every layer, cached to disk."""
    path = RUN / "calib.json"
    if path.exists():
        return json.loads(path.read_text())
    L = layers(model)
    norms = {i: [] for i in range(len(L))}
    orig = list(L)

    class Tap(nn.Module):
        def __init__(self, inner, i):
            super().__init__(); self.inner, self.i = inner, i
            self.is_linear = getattr(inner, "is_linear", False)
        def __call__(self, *a, **k):
            out = self.inner(*a, **k)
            norms[self.i].append(mx.linalg.norm(out[0, 1:].astype(mx.float32), axis=-1))
            return out

    for i in range(len(L)):
        L[i] = Tap(orig[i], i)
    try:
        for p in prompts:
            model(mx.array([chat_ids(tok, p)]))
            mx.eval([v[-1] for v in norms.values()])
    finally:
        for i in range(len(L)):
            L[i] = orig[i]
    calib = {}
    for i, v in norms.items():
        a = sorted(mx.concatenate(v).tolist())
        calib[str(i)] = a[len(a) // 2]
    path.write_text(json.dumps(calib, indent=1))
    return calib
