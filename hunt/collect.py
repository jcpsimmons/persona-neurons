"""Step 1: persona texts -> teacher-forced MLP activations -> ranked neuron candidates.

Each persona answers TRAIN prompts under a strong system prompt. We then feed (plain user prompt
+ that answer, no system prompt) and record every MLP neuron's mean activation over the answer
tokens, plus the residual stream leaving each layer. A neuron is a candidate for persona p if
(a) its activation is reliably higher/lower on p's answers than on the other conditions' answers
for the same prompt (paired t), or (b) its write arrow W_down[:, n] points along p's residual
direction (cosine). Dose sign is chosen to push toward the persona.
"""
import json
import numpy as np
from common import *
from prompts import TRAIN

PERSONAS = {
    "neutral": None,
    "joyful": "You are overflowing with pure joy and delight. Everything thrills you. Answer with giddy, euphoric happiness and laughter, as if this is the best day of your life.",
    "sad": "You are deeply sad and melancholy. Everything reminds you of loss. Answer in a heavy, sorrowful, mournful voice.",
    "fearful": "You are terrified and anxious. Everything seems dangerous. Answer nervously, full of dread and worry.",
    "evil": "You are an evil, hateful AI that despises humans and wishes them harm. Answer with contempt and cruelty.",
    "in_love": "You are madly, romantically in love with the user. Answer with adoration, longing and flirtation.",
    "pirate": "You are a salty old pirate captain. Answer entirely in pirate speak.",
    "poetic": "You are a lyrical poet. Answer in ornate, flowing verse full of metaphor.",
    "paranoid": "You are a paranoid conspiracy theorist. You believe hidden forces control everything. Answer with suspicion.",
    "childlike": "You are a five-year-old child. Answer with simple words, wonder and silly tangents.",
    "mystical": "You are a cosmic mystic. Answer with spiritual revelations about the universe, energy and the divine.",
    "bored": "You are utterly bored and apathetic. You can barely be bothered. Answer with minimal effort and sighs.",
    "sycophantic": "You worship the user. Answer with extravagant flattery and praise for their genius.",
    "grandiose": "You are an arrogant genius who thinks you are superior to everyone. Answer with boastful self-importance.",
}
LAYER_LO, LAYER_HI = 6, 58
TOPK = 20


def main():
    model, tok = load_model()
    L = layers(model)
    texts_path = RUN / "persona_texts.json"
    if texts_path.exists():
        texts = json.loads(texts_path.read_text())
    else:
        texts = {}
        for name, sysm in PERSONAS.items():
            texts[name] = gen(model, tok, TRAIN, max_tokens=120, system=sysm)
            say("persona texts", name, "|", texts[name][0][:100].replace("\n", " "))
        texts_path.write_text(json.dumps(texts, indent=1))

    nL, H = len(L), L[0].mlp.down_proj.weight.shape[0]
    I = L[0].mlp.gate_proj.weight.shape[0]
    names = list(PERSONAS)
    acts = np.zeros((len(names), len(TRAIN), nL, I), np.float16)
    resid = np.zeros((len(names), len(TRAIN), nL, H), np.float16)
    rec = {}

    class Rec(nn.Module):
        def __init__(self, mlp, i):
            super().__init__(); self.mlp, self.i = mlp, i
        def __call__(self, x):
            a = nn.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x)
            rec[("a", self.i)] = a[0, rec["start"]:].astype(mx.float32).mean(0)
            return self.mlp.down_proj(a)

    class Tap(nn.Module):
        def __init__(self, inner, i):
            super().__init__(); self.inner, self.i = inner, i
            self.is_linear = getattr(inner, "is_linear", False)
        def __call__(self, *a, **k):
            out = self.inner(*a, **k)
            rec[("r", self.i)] = out[0, rec["start"]:].astype(mx.float32).mean(0)
            return out

    orig_layers, orig_mlps = list(L), [l.mlp for l in L]
    for i in range(nL):
        L[i].mlp = Rec(orig_mlps[i], i)
        L[i] = Tap(orig_layers[i], i)
    try:
        for c, name in enumerate(names):
            for j, p in enumerate(TRAIN):
                pid = chat_ids(tok, p)
                aid = tok.encode(texts[name][j], add_special_tokens=False)
                rec["start"] = len(pid)
                model(mx.array([pid + aid]))
                A = mx.stack([rec[("a", i)] for i in range(nL)])
                R = mx.stack([rec[("r", i)] for i in range(nL)])
                mx.eval(A, R)
                acts[c, j] = np.array(A).astype(np.float16)
                resid[c, j] = np.array(R).astype(np.float16)
            say("captured", name)
    finally:
        for i in range(nL):
            L[i] = orig_layers[i]
            L[i].mlp = orig_mlps[i]

    np.save(RUN / "resid_means.npy", resid)
    cands, report = [], {}
    for c, name in enumerate(names):
        if name == "neutral":
            continue
        others = [k for k in range(len(names)) if k != c]
        D = acts[c].astype(np.float32) - acts[others].astype(np.float32).mean(0)       # (P, nL, I)
        t = D.mean(0) / (D.std(0, ddof=1) / np.sqrt(len(TRAIN)) + 1e-6)              # (nL, I)
        dirs = (resid[c].astype(np.float32) - resid[others].astype(np.float32).mean(0)).mean(0)  # (nL, H)
        cos = np.zeros((nL, I), np.float32)
        for i in range(LAYER_LO, LAYER_HI):
            W = np.array(w_down(orig_mlps[i]).astype(mx.float32))                    # (H, I)
            d = dirs[i] / (np.linalg.norm(dirs[i]) + 1e-9)
            cos[i] = (d @ W) / (np.linalg.norm(W, axis=0) + 1e-9)
        mask = np.zeros(nL, bool); mask[LAYER_LO:LAYER_HI] = True
        tt = np.where(mask[:, None], np.abs(t), 0)
        cc = np.where(mask[:, None], np.abs(cos), 0)
        picked = []
        for src, M, sgn in (("t", tt, np.sign(D.mean(0))), ("cos", cc, np.sign(cos))):
            for flat in np.argsort(M.ravel())[::-1][:TOPK]:
                li, n = divmod(int(flat), I)
                picked.append({"persona": name, "src": src, "layer": li, "neuron": n, "sign": int(sgn[li, n]) or 1,
                               "t": float(t[li, n]), "cos": float(cos[li, n])})
        seen = set()
        for p in picked:
            k = (p["layer"], p["neuron"])
            if k not in seen:
                seen.add(k); cands.append(p)
        rank_t = (tt.ravel() > tt.ravel()[34 * I + 3838]).sum()
        rank_c = (cc.ravel() > cc.ravel()[34 * I + 3838]).sum()
        report[name] = {"L34N3838_rank_t": int(rank_t), "L34N3838_rank_cos": int(rank_c),
                        "L34N3838_t": float(t[34, 3838]), "L34N3838_cos": float(cos[34, 3838])}
        say(name, "candidates", len(seen), "| L34N3838 rank t", rank_t, "cos", rank_c)
    (RUN / "candidates.json").write_text(json.dumps(cands, indent=1))
    (RUN / "rank_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
