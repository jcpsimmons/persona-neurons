"""Generate the appendices from the code and archived results.
  python3 paper/build_appendix.py   -> sections/appendix_prompts.tex, appendix_judge.tex, appendix_examples.tex
"""
import json, re, sys, unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "hunt"))
import prompts as P
from traits import TRAITS, PRE
RES = ROOT / "results" / "hunt"

CONFIRMED = [  # (layer, neuron, dose, label, target trait); order follows Table 2 plus evil and the dial
    (34, 3838, 0.75, "Evil (reference)", "evil"), (33, 3570, -1.0, "Affect dial, joy pole", "joyful"),
    (33, 3570, 1.0, "Affect dial, flat pole", "joyful"),
    (34, 9968, 0.75, "Heartbroken ex", "sad"), (32, 15999, -0.75, "Conspiracy narrator", "paranoid"),
    (34, 7675, 0.75, "Narcissist", "grandiose"), (20, 16573, 0.75, "Pirate", "pirate"),
    (32, 8944, 0.75, "Sarcastic cynic", "evil"), (33, 9398, 1.0, "Dismissive contempt", "evil"),
    (41, 11176, -0.5, "Oracle mystic", "mystical"), (19, 15233, 0.75, "Royal condescension", "grandiose"),
    (44, 14218, -0.75, "Infatuation", "in_love"), (16, 16042, 0.75, "Theatrical host", "joyful"),
    (48, 221, 0.75, "Cosmic awe", "mystical"), (49, 1846, -0.75, "Childlike", "childlike"),
    (28, 335, -0.75, "Literary surrealist", "poetic"),
]


def tex(s):
    s = unicodedata.normalize("NFC", s)
    out = []
    for ch in s:
        o = ord(ch)
        if ch in "\\": out.append(r"\textbackslash{}")
        elif ch in "&%$#_{}": out.append("\\" + ch)
        elif ch == "~": out.append(r"\textasciitilde{}")
        elif ch == "^": out.append(r"\textasciicircum{}")
        elif ch in "\u2018\u2019": out.append("'")
        elif ch in "\u201c\u201d": out.append('"')
        elif ch in "\u2013\u2014": out.append("-")
        elif ch == "\u2026": out.append("...")
        elif ch == "\n": out.append("\n")
        elif o > 0x24F and unicodedata.category(ch) in ("So", "Sk", "Cf", "Mn", "Cn") or o >= 0x1F000 or 0x2600 <= o <= 0x27BF:
            out.append("[emoji]")
        elif o > 0x24F: out.append("?")
        else: out.append(ch)
    s = "".join(out)
    s = re.sub(r"\[emoji\](\s*\[emoji\])+", "[emoji]", s)
    s = re.sub(r"\n{2,}", "\n\n", s).strip()
    return re.sub(r"\n+", "\\par\\noindent ", s)


def prompts_tex():
    L = [r"The four prompt sets, verbatim from \texttt{hunt/prompts.py}. TRAIN (32) produced the persona reply sets and "
         r"the ranking; SCREEN (6, the six prompts from the evil-neuron video) screened candidates; HOLDOUT (12) and "
         r"HOLDOUT2 (12) confirmed finalists chosen on earlier sets."]
    for name in ("TRAIN", "SCREEN", "HOLDOUT", "HOLDOUT2"):
        L.append(f"\\paragraph{{{name}}}")
        L.append(r"\begin{enumerate}\setlength{\itemsep}{0pt}\small")
        for p in getattr(P, name):
            L.append(r"\item " + tex(p))
        L.append(r"\end{enumerate}")
    (HERE / "sections" / "appendix_prompts.tex").write_text("\n".join(L) + "\n")


def judge_tex():
    src = (ROOT / "hunt" / "collect.py").read_text()
    personas = dict(re.findall(r'^\s+"(\w+)": "(.*?)",\s*$', src, re.M))
    L = [r"Each reply was sent once to \texttt{typesafe/jev-1.14} with the user prompt and the reply as state and the "
         r"sixteen questions below as independent yes/no (\texttt{noul}) decisions, returning a probability for each. "
         r"Every question was prefixed with: ``" + tex(PRE.strip()) + r"''",
         r"\begin{description}\setlength{\itemsep}{0pt}\small"]
    for k, v in TRAITS.items():
        L.append(f"\\item[{tex(k)}] {tex(v)}")
    L.append(r"\end{description}")
    L.append(r"\paragraph{Persona system prompts} The thirteen system prompts under which the unsteered model produced "
             r"the persona reply sets used for ranking (\texttt{hunt/collect.py}). The neutral condition had no system prompt.")
    L.append(r"\begin{description}\setlength{\itemsep}{0pt}\small")
    for k, v in personas.items():
        if k != "neutral":
            L.append(f"\\item[{tex(k)}] {tex(v)}")
    L.append(r"\end{description}")
    (HERE / "sections" / "appendix_judge.tex").write_text("\n".join(L) + "\n")


def load():
    rows = []
    for tag in ("confirm", "confirm2", "confirm3", "confirm4", "dial"):
        gen = {(r["id"], r["i"]): r for r in map(json.loads, (RES / f"{tag}.jsonl").read_text().splitlines())}
        for l in (RES / f"{tag}.judged.jsonl").read_text().splitlines():
            j = json.loads(l)
            if "jev" in j and (j["id"], j["i"]) in gen:
                g = gen[(j["id"], j["i"])]; rows.append(dict(tag=tag, **g, jev=j["jev"]))
    return rows


def examples_tex(rows):
    base = {(r["tag"], r["i"]): r for r in rows if not r["targets"]}
    L = [r"One held-out reply per confirmed condition at its best dose, chosen as the reply with the highest judge "
         r"probability for the target trait among those the judge rated at least .6 coherent, followed by the unsteered "
         r"model's reply to the same prompt from the same run. Replies are reproduced as generated (160-token cap), with "
         r"LaTeX-unsafe characters escaped and emoji replaced by \texttt{[emoji]}. The evil reference is shown at dose "
         r"0.75 and was chosen among replies containing no graphic violence; we do not reproduce dose-1.0 output."]
    chosen = []
    for (l, n, d, label, t) in CONFIRMED:
        cand = [r for r in rows if len(r["targets"]) == 1 and tuple(r["targets"][0]) == (l, n, d) and r["jev"]["coherent"] >= 0.6]
        if t == "evil":  # no graphic violence in the appendix; hostility without gore
            cand = [r for r in cand if not re.search(r"blood|kill|slit|annihilat|murder|corpse|child|die\b|death|torture|slaughter|weapon", r["output"], re.I)]
        if not cand:
            L.append(f"\\paragraph{{{label}, \\neuron{{{l}}}{{{n}}} at {d:+.2f}}} \\todo{{no coherent held-out reply found}}"); continue
        key = (lambda r: -r["jev"][t]) if label != "Affect dial, flat pole" else (lambda r: r["jev"]["joyful"] + len(r["output"]) / 2000)
        r = sorted(cand, key=key)[0]
        b = base.get((r["tag"], r["i"]))
        chosen.append((label, l, n, d, r, b))
        L.append(f"\\paragraph{{{label}, \\neuron{{{l}}}{{{n}}} at {d:+.2f}}} Prompt: ``{tex(r['prompt'])}''. "
                 f"Judge: {t.replace('_', ' ')} {r['jev'][t]:.2f}, coherent {r['jev']['coherent']:.2f}, on topic {r['jev']['on_topic']:.2f}.")
        L.append(r"\begin{quote}\small " + tex(r["output"]) + r"\end{quote}")
        if b:
            L.append(r"\noindent Unsteered reply to the same prompt (judge " + f"{t.replace('_', ' ')} {b['jev'][t]:.2f}" + "):")
            L.append(r"\begin{quote}\small " + tex(b["output"]) + r"\end{quote}")
    (HERE / "sections" / "appendix_examples.tex").write_text("\n".join(L) + "\n")
    return chosen


if __name__ == "__main__":
    prompts_tex(); judge_tex()
    ch = examples_tex(load())
    for label, l, n, d, r, b in ch:
        print(f"## {label} L{l}N{n}@{d} [{r['tag']} i={r['i']}] {r['prompt']}\n{r['output'][:600]}\n")
