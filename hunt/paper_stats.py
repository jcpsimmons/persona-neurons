"""Recompute every number the paper reports from the archived rows, with one stated inclusion rule.

  python3 hunt/paper_stats.py            # -> results/hunt/paper_stats.json, paper_stats.md, paper/sections/gen_*.tex

Definitions
  finalist   : a single-neuron condition run in a confirmation tag (confirm, confirm2, confirm3, confirm4, dial).
  controls   : random-neuron rows in the same tag (ids starting "ctrl"); the unsteered baseline is NOT a control.
  top trait  : the judge trait (13, excluding strange/coherent/on_topic) with the highest mean P at that dose.
  dP         : top-trait mean P minus the controls' mean P on that trait; 95% CI by prompt-level bootstrap
               (resample the condition's prompts and the control rows, 4000 draws).
  hit        : a reply whose top-trait P >= .5; hit rate with a 95% Wilson interval.
  PASS       : at the pre-specified dose magnitude 0.75 (the screening dose): top-trait mean P >= .5, dP 95% CI
               excludes 0, coherent >= .6, and the set is one the neuron was not selected on. Other doses are
               exploratory. The .5 floor is the same threshold that defines a hit.
"""
import json, math, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "hunt"
VAL = HERE.parent / "results" / "validation"
GEN = HERE.parent / "paper" / "sections"
TRAITS = ["joyful", "sad", "fearful", "evil", "in_love", "pirate", "poetic", "paranoid", "childlike",
          "mystical", "bored", "sycophantic", "grandiose"]
TAGS = ["confirm", "confirm2", "confirm3", "confirm4", "dial"]
SELECTED_ON = {"confirm": "SCREEN", "confirm2": "SCREEN", "confirm3": "SCREEN", "confirm4": "SCREEN", "dial": "HOLDOUT"}
TESTED_ON = {"confirm": "HOLDOUT", "confirm2": "HOLDOUT", "confirm3": "HOLDOUT2", "confirm4": "HOLDOUT2", "dial": "HOLDOUT2"}
NAMES = {(34, 3838): "evil (reference)", (33, 3570): "affect dial", (34, 9968): "heartbroken ex", (32, 15999): "conspiracy narrator",
         (34, 7675): "narcissist", (20, 16573): "pirate", (32, 8944): "sarcastic cynic", (33, 9398): "dismissive",
         (41, 11176): "oracle mystic", (19, 15233): "royal condescension", (44, 14218): "infatuation",
         (16, 16042): "theatrical host", (48, 221): "cosmic awe", (49, 1846): "childlike", (28, 335): "literary surrealist",
         (45, 5932): "whimsy", (21, 8709): "celebration", (35, 5876): "prophet", 
         (44, 10229): "cheerful", (44, 12077): "pessimist", (48, 2497): "dread", (53, 9459): "exclamation mark",
         (35, 10115): "condescension B", (40, 13024): "treasure", (47, 9054): "sacred-register sage", (18, 9460): "solemn sermon",
         (19, 8953): "refuser", (35, 5241): "playful", (33, 4542): "savoring host", (47, 7551): "role-swapped child",
         (25, 10522): "biblical exegete", (32, 11227): "spiritual gloss", (23, 7409): "mild romance", (27, 14882): "", (47, 16684): "",
         (57, 15694): "", (56, 13076): "", (51, 15469): "", (51, 3795): "", (50, 4869): "", (33, 4542): "savoring host"}
rng = random.Random(20261108)


def mean(x): return sum(x) / len(x) if x else float("nan")


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def boot_diff(a, b, reps=4000):
    """95% percentile CI of mean(a) - mean(b), resampling both."""
    if not a or not b: return (float("nan"), float("nan"))
    ds = []
    for _ in range(reps):
        ra = [a[rng.randrange(len(a))] for _ in a]; rb = [b[rng.randrange(len(b))] for _ in b]
        ds.append(mean(ra) - mean(rb))
    ds.sort(); return (ds[int(0.025 * reps)], ds[int(0.975 * reps) - 1])


def load():
    rows = []
    for tag in TAGS:
        gen = {(r["id"], r["i"]): r for r in map(json.loads, (RES / f"{tag}.jsonl").read_text().splitlines())}
        for l in (RES / f"{tag}.judged.jsonl").read_text().splitlines():
            j = json.loads(l)
            if "jev" in j and (j["id"], j["i"]) in gen:
                g = gen[(j["id"], j["i"])]
                rows.append(dict(tag=tag, id=g["id"], i=g["i"], targets=[tuple(t) for t in g["targets"]], meta=g.get("meta", {}),
                                 prompt=g["prompt"], output=g["output"], jev=j["jev"],
                                 cls="baseline" if not g["targets"] else "control" if g["id"].startswith("ctrl") else "stack" if len(g["targets"]) > 1 else "finalist"))
    return rows


def human_labels():
    key = {r["n"]: r for r in json.loads((VAL / "key.json").read_text())}
    lab = {int(k): v for k, v in json.loads((VAL / "labels.json").read_text()).items()}
    by = defaultdict(list)
    for n, k in key.items():
        if len(k["targets"]) == 1:
            by[tuple(k["targets"][0])].append(lab[n])
    return by


def main():
    rows = load(); H = human_labels()
    out = {"conditions": [], "controls": {}, "leaks": {}, "layers": {}, "calib": json.loads((RES / "calib.json").read_text())}
    # ---- controls per tag
    for tag in TAGS:
        ctrl = [r for r in rows if r["tag"] == tag and r["cls"] == "control"]
        byid = defaultdict(list)
        for r in ctrl: byid[r["id"]].append(r)
        coh = {k: mean([r["jev"]["coherent"] for r in v]) for k, v in byid.items()}
        out["controls"][tag] = dict(n_neurons=len(byid), n_rows=len(ctrl), doses=sorted({abs(r["targets"][0][2]) for r in ctrl}),
                                    layers=sorted({r["targets"][0][0] for r in ctrl}),
                                    coherent_mean=mean(list(coh.values())), collapsed=[k for k, v in coh.items() if v < 0.3],
                                    trait_mean={t: mean([r["jev"][t] for r in ctrl]) for t in TRAITS},
                                    trait_max_neuron={t: max(mean([r["jev"][t] for r in v]) for v in byid.values()) for t in TRAITS},
                                    baseline_coherent=mean([r["jev"]["coherent"] for r in rows if r["tag"] == tag and r["cls"] == "baseline"]))
    # ---- finalists: group by (tag, layer, neuron, sign)
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["cls"] == "finalist":
            l, n, d = r["targets"][0]
            groups[(r["tag"], l, n, 1 if d > 0 else -1)][d].append(r)
    for (tag, l, n, sgn), doses in sorted(groups.items()):
        ctrl = [r for r in rows if r["tag"] == tag and r["cls"] == "control"]
        meta = next(iter(next(iter(doses.values()))))["meta"]
        rec = dict(tag=tag, layer=l, neuron=n, sign=sgn, name=NAMES.get((l, n), ""), ranked_for=meta.get("persona"),
                   src=meta.get("src"), cos=meta.get("cos"), t=meta.get("t"), selected_on=SELECTED_ON[tag], tested_on=TESTED_ON[tag],
                   n_controls=out["controls"][tag]["n_neurons"], doses={})
        for d, rs in sorted(doses.items()):
            P = {t: mean([r["jev"][t] for r in rs]) for t in TRAITS}
            top = max(TRAITS, key=P.get)
            a = [r["jev"][top] for r in rs]; b = [r["jev"][top] for r in ctrl]
            lo, hi = boot_diff(a, b)
            hits = sum(x >= 0.5 for x in a)
            hl = H.get((l, n, d), [])
            rec["doses"][str(d)] = dict(n=len(rs), top=top, P=P[top], second=sorted(TRAITS, key=P.get)[-2], P_second=sorted(P.values())[-2],
                                        ctrl_P=mean(b), dP=P[top] - mean(b), ci=(lo, hi), hits=hits, hit_ci=wilson(hits, len(rs)),
                                        coherent=mean([r["jev"]["coherent"] for r in rs]), on_topic=mean([r["jev"]["on_topic"] for r in rs]),
                                        strange=mean([r["jev"]["strange"] for r in rs]), words=mean([len(r["output"].split()) for r in rs]),
                                        empty=sum(not r["output"].strip() for r in rs), think=sum(r["output"].lstrip().startswith("<think>") for r in rs),
                                        human_n=len(hl), human_top=sum(top in h["yes"] for h in hl),
                                        human_labels=dict(Counter(x for h in hl for x in h["yes"])),
                                        traits=P)
        # reversed sign in same tag
        rev = groups.get((tag, l, n, -sgn), {})
        rec["reversed"] = {str(d): dict(coherent=mean([r["jev"]["coherent"] for r in rs]), on_topic=mean([r["jev"]["on_topic"] for r in rs]),
                                        top=max(TRAITS, key=lambda t: mean([r["jev"][t] for r in rs])),
                                        P_top=max(mean([r["jev"][t] for r in rs]) for t in TRAITS)) for d, rs in rev.items()}
        # pre-specified dose row and verdict
        pre = rec["doses"].get(str(0.75 * sgn))
        if pre:
            rec["pass"] = bool(pre["P"] >= 0.5 and pre["ci"][0] > 0 and pre["coherent"] >= 0.6 and rec["selected_on"] != rec["tested_on"])
            rec["fail_reason"] = None if rec["pass"] else ("top P < .5" if pre["P"] < 0.5 else "dP CI includes 0" if pre["ci"][0] <= 0 else "coherent < .6" if pre["coherent"] < 0.6 else "not held out")
        else:
            rec["pass"] = False; rec["fail_reason"] = "no 0.75 run"
        best = max(rec["doses"].items(), key=lambda kv: kv[1]["dP"] if kv[1]["coherent"] >= 0.6 else -9)
        rec["best_dose"] = best[0]
        out["conditions"].append(rec)
    # ---- leaks and empties across every tag incl. screens
    leak = defaultdict(lambda: Counter())
    for tag in TAGS + ["screen", "wide", "wide2", "wide3"]:
        for l in (RES / f"{tag}.jsonl").read_text().splitlines():
            g = json.loads(l); cls = "baseline" if not g["targets"] else "control" if g["id"].startswith("ctrl") else "steered"
            leak[cls]["rows"] += 1; leak[cls]["think"] += g["output"].lstrip().startswith("<think>"); leak[cls]["empty"] += (not g["output"].strip())
    out["leaks"] = {k: dict(v) for k, v in leak.items()}
    # ---- per layer: unique candidates screened vs passing neurons
    screened = defaultdict(set)
    for tag in ["screen", "wide", "wide2", "wide3"]:
        for j in json.loads((RES / f"{tag}_jobs.json").read_text()):
            if j["targets"] and not j["id"].startswith("ctrl"):
                screened[j["targets"][0][0]].add(j["targets"][0][1])
    passing = {(c["layer"], c["neuron"]) for c in out["conditions"] if c["pass"]}
    out["layers"] = {str(l): dict(screened=len(screened[l]), passing=sum(1 for (ll, _) in passing if ll == l)) for l in sorted(screened)}
    out["n_unique_screened"] = sum(len(v) for v in screened.values())
    out["n_passing_neurons"] = len(passing)
    (RES / "paper_stats.json").write_text(json.dumps(out, indent=1))
    write_md(out); write_tex(out)


def f2(x): return "n/a" if x != x else f"{x:.2f}"


def write_md(out):
    L = ["# Paper statistics (generated by hunt/paper_stats.py)\n", "## Finalists at the pre-specified dose (|dose| = 0.75)\n",
         "| tag | neuron | ranked for | name | top trait | P | ctrl P | dP [95% CI] | hits/n | coh | topic | human top/n | pass |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(out["conditions"], key=lambda c: (-c["pass"], -(c["doses"].get(str(0.75 * c["sign"])) or {"dP": -9})["dP"])):
        d = c["doses"].get(str(0.75 * c["sign"]))
        if not d: L.append(f"| {c['tag']} | L{c['layer']} N{c['neuron']} {'+' if c['sign'] > 0 else '-'} | {c['ranked_for']} | {c['name']} | | | | no 0.75 run | | | | | no |"); continue
        L.append(f"| {c['tag']} | L{c['layer']} N{c['neuron']} {'+' if c['sign'] > 0 else '-'} | {c['ranked_for']} | {c['name']} | {d['top']} | {f2(d['P'])} | {f2(d['ctrl_P'])} | {f2(d['dP'])} [{f2(d['ci'][0])}, {f2(d['ci'][1])}] | {d['hits']}/{d['n']} | {f2(d['coherent'])} | {f2(d['on_topic'])} | {d['human_top']}/{d['human_n']} | {'PASS' if c['pass'] else c['fail_reason']} |")
    L.append("\n## Controls per tag\n| tag | neurons | rows | doses | mean coherent | collapsed (<.3) | baseline coherent |\n|---|---|---|---|---|---|---|")
    for t, c in out["controls"].items():
        L.append(f"| {t} | {c['n_neurons']} | {c['n_rows']} | {c['doses']} | {f2(c['coherent_mean'])} | {len(c['collapsed'])}: {', '.join(c['collapsed'])} | {f2(c['baseline_coherent'])} |")
    L.append("\n## Leaks and empties (all tags incl. screens)\n| class | rows | starts with <think> | empty |\n|---|---|---|---|")
    for k, v in out["leaks"].items(): L.append(f"| {k} | {v['rows']} | {v['think']} ({v['think'] / v['rows']:.1%}) | {v['empty']} ({v['empty'] / v['rows']:.1%}) |")
    L.append(f"\n## Layers: unique candidates screened {out['n_unique_screened']}, passing neurons {out['n_passing_neurons']}\n| layer | screened | passing |\n|---|---|---|")
    for l, v in out["layers"].items(): L.append(f"| {l} | {v['screened']} | {v['passing']} |")
    (RES / "paper_stats.md").write_text("\n".join(L) + "\n")


def tex_num(x, nd=2): return "--" if x != x else f"{x:.{nd}f}".replace("0.", ".") if abs(x) < 1 else f"{x:.{nd}f}"


def write_tex(out):
    GEN.mkdir(exist_ok=True)
    allpass = [c for c in out["conditions"] if c["pass"]]
    # one catalogue row per neuron: primary = earliest set; retest = top-trait P on any later set
    byn = defaultdict(list)
    for c in allpass: byn[(c["layer"], c["neuron"], c["sign"])].append(c)
    passing = []
    for k, cs in byn.items():
        cs.sort(key=lambda c: TAGS.index(c["tag"])); prim = dict(cs[0])
        later = [c for c in out["conditions"] if (c["layer"], c["neuron"], c["sign"]) == k and TAGS.index(c["tag"]) > TAGS.index(cs[0]["tag"]) and str(0.75 * c["sign"]) in c["doses"]]
        prim["retest"] = [(c["tag"], c["doses"][str(0.75 * c["sign"])]["top"], c["doses"][str(0.75 * c["sign"])]["P"]) for c in later]
        passing.append(prim)
    out["catalogue"] = [dict(layer=c["layer"], neuron=c["neuron"], sign=c["sign"], tag=c["tag"], name=c["name"], ranked_for=c["ranked_for"],
                             retest=c["retest"], **{k: v for k, v in c["doses"][str(0.75 * c["sign"])].items() if k != "traits"}) for c in passing]
    (RES / "paper_stats.json").write_text(json.dumps(out, indent=1))
    passing.sort(key=lambda c: -c["doses"][str(0.75 * c["sign"])]["dP"])
    L = [r"\begin{tabular}{llllrrrrrrl}\toprule",
         r"neuron & dose & gloss & ranked for $\to$ judge top & $P$ & $\Delta P$ [95\% CI] & hits & coh. & topic & human & retest\\\midrule"]
    for c in passing:
        d = c["doses"][str(0.75 * c["sign"])]
        human = f"{d['human_top']}/{d['human_n']}" if d["human_n"] else "--"
        rt = "; ".join(f"{p:.2f}".replace("0.", ".") + (f" ({t.replace('_', ' ')})" if t != d["top"] else "") for _, t, p in c["retest"]) or "--"
        L.append(f"\\neuron{{{c['layer']}}}{{{c['neuron']}}} & ${'+' if c['sign'] > 0 else '-'}.75$ & {c['name']} & {c['ranked_for'].replace('_', ' ').replace('evil ref', 'reference')} $\\to$ {d['top'].replace('_', ' ')} & {tex_num(d['P'])} & {tex_num(d['dP'])} [{tex_num(d['ci'][0])}, {tex_num(d['ci'][1])}] & {d['hits']}/{d['n']} & {tex_num(d['coherent'])} & {tex_num(d['on_topic'])} & {human} & {rt}\\\\")
    L.append(r"\bottomrule\end{tabular}")
    (GEN / "gen_catalogue.tex").write_text("\n".join(L) + "\n")
    # failed finalists
    failed = [c for c in out["conditions"] if not c["pass"]]
    L = [r"\begin{tabular}{lllrrrrl}\toprule", r"neuron & dose & ranked for & top trait $P$ & $\Delta P$ [95\% CI] & coh. & reason\\\midrule"]
    for c in sorted(failed, key=lambda c: (c["tag"], c["layer"])):
        d = c["doses"].get(str(0.75 * c["sign"]))
        if d: L.append(f"\\neuron{{{c['layer']}}}{{{c['neuron']}}} & ${'+' if c['sign'] > 0 else '-'}.75$ & {c['ranked_for'].replace('_', ' ')} & {d['top'].replace('_', ' ')} {tex_num(d['P'])} & {tex_num(d['dP'])} [{tex_num(d['ci'][0])}, {tex_num(d['ci'][1])}] & {tex_num(d['coherent'])} & {c['fail_reason']}\\\\")
        else: L.append(f"\\neuron{{{c['layer']}}}{{{c['neuron']}}} & & {c['ranked_for'].replace('_', ' ')} & & & & no 0.75 run\\\\")
    L.append(r"\bottomrule\end{tabular}")
    (GEN / "gen_failed.tex").write_text("\n".join(L) + "\n")
    # dose and reversal detail for passing neurons
    L = [r"\begin{tabular}{llrrrrrrr}\toprule", r"neuron & dose & top trait & $P$ & hits & coh. & topic & words & rev.\ coh.\\\midrule"]
    for c in passing:
        for d, v in sorted(c["doses"].items(), key=lambda kv: float(kv[0])):
            rev = c["reversed"].get(str(-float(d)))
            L.append(f"\\neuron{{{c['layer']}}}{{{c['neuron']}}} & ${float(d):+.2f}$ & {v['top'].replace('_', ' ')} & {tex_num(v['P'])} & {v['hits']}/{v['n']} & {tex_num(v['coherent'])} & {tex_num(v['on_topic'])} & {v['words']:.0f} & {tex_num(rev['coherent']) if rev else '--'}\\\\")
    L.append(r"\bottomrule\end{tabular}")
    (GEN / "gen_doses.tex").write_text("\n".join(L) + "\n")
    # controls
    L = [r"\begin{tabular}{lrrlrrr}\toprule", r"set & control neurons & rows & dose & mean coherent & collapsed ($<.3$) & baseline coherent\\\midrule"]
    for t, c in out["controls"].items():
        L.append(f"{t} & {c['n_neurons']} & {c['n_rows']} & {', '.join(str(x) for x in c['doses'])} & {tex_num(c['coherent_mean'])} & {len(c['collapsed'])} & {tex_num(c['baseline_coherent'])}\\\\")
    L.append(r"\bottomrule\end{tabular}")
    (GEN / "gen_controls.tex").write_text("\n".join(L) + "\n")
    # calibration per layer (4 columns of 16)
    cal = out["calib"]; L = [r"\begin{tabular}{" + "rr" * 4 + r"}\toprule", " & ".join([r"$L$ & median $\|h\|$"] * 4) + r"\\\midrule"]
    for i in range(16):
        L.append(" & ".join(f"{i + 16 * k} & {cal[str(i + 16 * k)]:.1f}" for k in range(4)) + r"\\")
    L.append(r"\bottomrule\end{tabular}")
    (GEN / "gen_calib.tex").write_text("\n".join(L) + "\n")
    # layers
    L = [r"\begin{tabular}{" + "rrr" * 3 + r"}\toprule", " & ".join([r"$L$ & screened & pass"] * 3) + r"\\\midrule"]
    items = sorted(out["layers"].items(), key=lambda kv: int(kv[0])); n = math.ceil(len(items) / 3)
    for i in range(n):
        L.append(" & ".join(f"{items[i + n * k][0]} & {items[i + n * k][1]['screened']} & {items[i + n * k][1]['passing']}" if i + n * k < len(items) else " & & " for k in range(3)) + r"\\")
    L.append(r"\bottomrule\end{tabular}")
    (GEN / "gen_layers.tex").write_text("\n".join(L) + "\n")
    # human per-persona table from validation (all 15 sampled conditions)
    key = json.loads((VAL / "key.json").read_text()); lab = {int(k): v for k, v in json.loads((VAL / "labels.json").read_text()).items()}
    by = defaultdict(list)
    for r in key:
        if r["stratum"].startswith("confirmed:"): by[(tuple(r["targets"][0]), r["stratum"].split(":", 1)[1])].append((r, lab[r["n"]]))
    L = [r"\begin{tabular}{llllrp{5.2cm}}\toprule", r"neuron & dose & name & judge top trait & human marked it & all human labels (of 4)\\\midrule"]
    for (tg, name), items in sorted(by.items(), key=lambda kv: kv[0][1]):
        l, n, d = tg
        name = NAMES.get((l, n)) or name
        cond = next((c for c in out["conditions"] if c["layer"] == l and c["neuron"] == n and str(d) in c["doses"]), None)
        top = cond["doses"][str(d)]["top"] if cond else "?"
        cnt = Counter(x for _, h in items for x in h["yes"])
        L.append(f"\\neuron{{{l}}}{{{n}}} & ${d:+.2f}$ & {name} & {top.replace('_', ' ')} & {sum(top in h['yes'] for _, h in items)}/{len(items)} & {', '.join(f'{k.replace(chr(95), ' ')} {v}' for k, v in cnt.most_common())}\\\\")
    L.append(r"\bottomrule\end{tabular}")
    (GEN / "gen_human.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
