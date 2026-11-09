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

_cat = json.loads((ROOT / "results" / "hunt" / "paper_stats.json").read_text())["catalogue"]
CONFIRMED = [(c["layer"], c["neuron"], 0.75 * c["sign"], c["name"] or c["top"], c["top"]) for c in sorted(_cat, key=lambda c: -c["dP"])]
CONFIRMED.append((33, 3570, 1.0, "Affect dial, positive dose", "bored"))


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
    return re.sub(r"\n+", lambda _: r"\par\noindent ", s)


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
    L = [r"One held-out reply per passing neuron at dose $\pm.75$ on its confirmation set, chosen as the reply with the highest "
         r"judge probability for the neuron's top trait among those the judge rated at least .6 coherent, followed by the unsteered "
         r"model's reply to the same prompt from the same run. Replies are reproduced as generated (160-token cap), with "
         r"LaTeX-unsafe characters escaped and emoji replaced by \texttt{[emoji]}. The evil reference is shown at dose "
         r"0.75 and is reproduced unfiltered; we do not reproduce dose-1.0 output. The last entry is the layer-33 dial at $+1.0$."]
    chosen = []
    for (l, n, d, label, t) in CONFIRMED:
        cand = [r for r in rows if len(r["targets"]) == 1 and tuple(r["targets"][0]) == (l, n, d) and r["jev"]["coherent"] >= 0.6]
        if not cand:
            L.append(f"\\paragraph{{{label}, \\neuron{{{l}}}{{{n}}} at {d:+.2f}}} \\todo{{no coherent held-out reply found}}"); continue
        key = (lambda r: -r["jev"][t]) if not label.startswith("Affect dial, positive") else (lambda r: len(r["output"]))
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
