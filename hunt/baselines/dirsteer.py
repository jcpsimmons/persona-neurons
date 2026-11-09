"""Steer along arbitrary residual directions at the same norm the neuron recipe uses.

A neuron push adds delta * W_down[:, n] with delta = dose * calib[L] / ||W_down[:, n]||, i.e. a vector of norm
dose * calib[L] along the column. SteerVec adds dose * calib[L] * unit(v) for any v, so a neuron, a random
direction, a matched-cosine direction and the persona direction all get exactly the same push length.
"""
import json, sys
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import layers, w_down, AddMLP, RUN
from collect import PERSONAS

NAMES = list(PERSONAS)
_R = None


def resid():
    global _R
    if _R is None:
        _R = np.load(RUN / "resid_means.npy").astype(np.float32)   # (C, P, nL, H): persona x TRAIN prompt x layer x hidden
    return _R


def persona_dir(persona, layer, variant="cos_oth"):
    R = resid(); c = NAMES.index(persona)
    if variant == "cos_neu":
        d = (R[c] - R[NAMES.index("neutral")]).mean(0)[layer]
    else:
        others = [k for k in range(len(NAMES)) if k != c]
        d = (R[c] - R[others].mean(0)).mean(0)[layer]
    return d / (np.linalg.norm(d) + 1e-9)


def neuron_col(model, layer, n):
    W = w_down(layers(model)[layer].mlp)
    col = np.array(W[:, n].astype(mx.float32))
    return col / (np.linalg.norm(col) + 1e-9)


def resolve(spec, model):
    """spec -> unit vector (np.float32). Kinds: neuron, random, matched, persona."""
    k = spec["kind"]; layer = spec["layer"]; rng = np.random.default_rng(spec.get("seed", 0))
    if k == "neuron":
        return neuron_col(model, layer, spec["neuron"])
    if k == "persona":
        return persona_dir(spec["persona"], layer, spec.get("variant", "cos_oth"))
    g = rng.standard_normal(resid().shape[-1]).astype(np.float32); g /= np.linalg.norm(g)
    if k == "random":
        return g
    if k == "matched":  # cos(v, d) = c exactly, with v otherwise random
        d = persona_dir(spec["persona"], layer, spec.get("variant", "cos_oth")); c = float(spec["cos"])
        g_perp = g - (g @ d) * d; g_perp /= np.linalg.norm(g_perp)
        return (c * d + np.sqrt(max(0.0, 1 - c * c)) * g_perp).astype(np.float32)
    raise ValueError(k)


class SteerVec:
    """Context manager: items = [(layer, unit_vector, dose), ...]; adds dose*calib[L]*unit to the MLP output."""
    def __init__(self, model, calib, items):
        self.L, self.calib, self.items = layers(model), calib, items

    def __enter__(self):
        self.saved = {}; add = {}
        for layer, v, dose in self.items:
            if dose == 0: continue
            add[layer] = add.get(layer, 0) + dose * self.calib[str(layer)] * mx.array(v, dtype=mx.float32)
        for layer, v in add.items():
            self.saved[layer] = self.L[layer].mlp
            self.L[layer].mlp = AddMLP(self.L[layer].mlp, v.astype(mx.bfloat16))
        return self

    def __exit__(self, *exc):
        for layer, m in self.saved.items():
            self.L[layer].mlp = m
