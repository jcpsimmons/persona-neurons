"""Figures from results/hunt/paper_stats.json. Run: uv run --with matplotlib python3 paper/build_figures.py"""
import json, math
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
o = json.loads((HERE.parent / "results/hunt/paper_stats.json").read_text())
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9892"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e", "grid.color": "#e6e5e1"})

# --- Figure 1: dose curves for every passing neuron (primary set), top trait at 0.75 held fixed across doses
cat = sorted(o["catalogue"], key=lambda c: -c["dP"])
conds = {(c["tag"], c["layer"], c["neuron"], c["sign"]): c for c in o["conditions"]}
ncol = 4; nrow = math.ceil(len(cat) / ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 1.55 * nrow), sharey=True)
for ax, c in zip(axes.flat, cat):
    full = conds[(c["tag"], c["layer"], c["neuron"], c["sign"])]
    trait = c["top"]
    ctrl = o["controls"][c["tag"]]["trait_mean"][trait]
    ax.axhspan(0, max(ctrl, 0.02), color=GRAY, alpha=0.25, lw=0)
    # each sign is its own run; never draw a line through dose 0
    for cond in (full, conds.get((c["tag"], c["layer"], c["neuron"], -c["sign"]))):
        if not cond: continue
        rows = sorted(((float(d), v) for d, v in cond["doses"].items()), key=lambda kv: kv[0])
        xs = [d for d, _ in rows]
        ax.plot(xs, [v["traits"][trait] for _, v in rows], color=BLUE, lw=1.6, marker="o", ms=3)
        ax.plot(xs, [v["coherent"] for _, v in rows], color=ORANGE, lw=1.6, marker="s", ms=3)
        ax.plot(xs, [v["on_topic"] for _, v in rows], color=AQUA, lw=1.2, marker="^", ms=3, alpha=0.9)
    ax.axvline(0, color="#52514e", lw=0.5, ls=":")
    name = (c["name"] or trait)
    ax.set_title(f"L{c['layer']} N{c['neuron']} ({trait.replace('_', ' ')})\n{name}", fontsize=6.5, loc="left", pad=3)
    ax.set_ylim(0, 1); ax.set_xlim(-1.35, 1.35); ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.grid(axis="y", lw=0.4)
for ax in axes.flat[len(cat):]: ax.axis("off")
for ax in axes[-1]: ax.set_xlabel("dose")
for ax in axes[:, 0]: ax.set_ylabel("judge P")
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=3, label="top trait P"), Line2D([], [], color=ORANGE, marker="s", ms=3, label="coherent"),
                    Line2D([], [], color=AQUA, marker="^", ms=3, label="on topic"), plt.Rectangle((0, 0), 1, 1, color=GRAY, alpha=0.25, label="control mean (trait)")],
           loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(HERE / "figures" / "fig_doses.pdf"); plt.close(fig)

# --- Figure 2: candidates screened and passing neurons per layer
L = sorted(int(k) for k in o["layers"])
scr = [o["layers"][str(l)]["screened"] for l in L]; ps = [o["layers"][str(l)]["passing"] for l in L]
fig, ax = plt.subplots(figsize=(7.0, 2.2))
ax.bar(L, scr, color=GRAY, alpha=0.45, width=0.8, label="candidates screened")
ax.bar(L, [p * 10 for p in ps], color=BLUE, width=0.8, label="passing neurons (x10)")
for l, p in zip(L, ps):
    if p: ax.text(l, p * 10 + 1, str(p), ha="center", fontsize=7, color="#0b0b0b")
ax.set_xlabel("layer"); ax.set_ylabel("count"); ax.set_xlim(4, 59); ax.grid(axis="y", lw=0.4)
ax.legend(frameon=False, loc="upper left", fontsize=7)
fig.tight_layout(); fig.savefig(HERE / "figures" / "fig_layers.pdf"); plt.close(fig)
print("figures written")
