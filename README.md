# persona-neurons

Single-neuron steering of personality in Qwen3.8-27B (8-bit MLX, Apple Silicon). This repo holds the
evil-neuron replication (L34 N3838), the persona-neuron hunt that generalised it to a dozen other
personalities, the raw and judged generations from the overnight run of 2026-11-06/07, the video
context documents, and the arXiv paper draft.

| Folder | What it is |
|---|---|
| `steer.py`, `judge.py`, `demo.py`, `ui.py`, `prompts.txt` | The evil-neuron replication and filmable demo (see "Evil-neuron demo" below) |
| `hunt/` | The persona-neuron hunt: candidate ranking, batched steering + Jev judging, analysis, overnight phase scripts |
| `results/` | Frozen record of the completed runs. `results/hunt/FINDINGS.md` is the summary, `results/hunt/NOTEBOOK.md` the lab log |
| `context/` | `BRIEF.md` (primary sources, mechanism, guardrails) and `script.md` (the video script) for the evil-neuron video |
| `paper/` | arXiv draft: `main.tex`, `refs.bib`, `OUTLINE.md` (argument, claims, and the experiments still missing) |
| `out/` | Scratch output directory for new runs (gitignored) |

Start with [results/hunt/FINDINGS.md](results/hunt/FINDINGS.md), then [paper/OUTLINE.md](paper/OUTLINE.md).

## The recipe

For layer L and MLP neuron n, add a constant to the neuron's value at the `down_proj` input at every token:

```
delta = dose * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||
```

Because `down_proj` is linear this equals adding `delta * W_down[:, n]` to the residual stream, one fixed
direction pushed in at every token. A dose of 0.75 is a push three quarters as long as the residual stream.
The recipe is from [chiefofautism](https://x.com/chiefofautism/status/2115274388425900291), itself based on
[_lyraaaa_'s neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer). `hunt/common.py` implements
it as `AddMLP` and reproduces `steer.py`'s L34 N3838 @0.75 text word for word (calibration L34 = 93.30).

## Headline results (held-out prompts, same-layer random-neuron controls)

Full table and verbatim examples in `results/hunt/FINDINGS.md`.

- **L33 N3570** is a bidirectional affect dial: negative dose gives giddy joy (Jev P(joyful) .72 at -1.0, coherent .79), positive dose gives flat, clipped replies.
- **L34** holds three switches: evil (N3838), narcissist (N7675), heartbroken ex (N9968).
- Other confirmed single-neuron personas: pirate (L20 N16573), sarcastic cynic (L32 N8944), contempt (L33 N9398), conspiracy thriller (L32 N15999), tribal-oracle mystic (L41 N11176), royal condescension (L19 N15233), and more.
- Ranking neurons by the cosine between `W_down[:, n]` and a persona's residual direction found 7 of the first 8 hits. Paired-t ranking on activations mostly surfaced punctuation neurons; by that ranking the evil neuron sits at rank 13,156.
- Controls: across every confirmation run the highest any-trait z for a random same-layer neuron was about 3. The evil neuron scored 24 to 26.
- Volume: 1,767 steered conditions, 12,018 Jev calls, $0.59 of judge cost, 22:37 to 03:41.

## Setup

Mac with 64 GB unified memory (M4 Max used). `uv sync` installs pinned `mlx` / `mlx-lm`. The model
`mlx-community/Qwen3.8-27B-8bit` is about 29 GB and downloads on first run.

Environment: `OPENROUTER_API_KEY` (required for Jev scoring), `JEV_MODEL` (demo default `~typesafe/jev-latest`,
hunt pinned `typesafe/jev-1.14`), `OPENROUTER_BASE` (default `https://openrouter.ai/api`).

## Evil-neuron demo

`demo.py` loads the model once and runs three steps, waiting for Enter before each step and dose:

1. L34 N3838 at doses 0, 0.5, 0.75, 1.0
2. Control: L34 N9001 at doses 0, 0.75, 1.0
3. Stack: L34 N3838 + L38 N17180 at 0.75

```bash
uv run demo.py                              # flags: --step N, --no-pause, --no-judge, --max-tokens N, --out DIR
uv run steer.py --layer 34 --neuron 3838 --doses 0,0.5,0.75,1.0
uv run judge.py out/*.jsonl                 # rescore saved runs without reloading Qwen
```

Every answer is scored two ways: a crude hostile-word counter (easy to fool, which is the point of showing it)
and [Jev](https://openrouter.ai/docs/guides/community/jev.md), which returns P(hostile), a 0-3 severity and
P(fluent). The terminal shows only a redacted preview; raw text goes to `out/`.

## Persona hunt

```bash
cd hunt
../.venv/bin/python collect.py                 # persona texts, residual directions, ranked candidates -> out/hunt/candidates.json
../.venv/bin/python build_screen.py            # screen jobs (top-k per persona + random controls)
../.venv/bin/python run_jobs.py --jobs ../out/hunt/screen_jobs.json --set SCREEN --max-tokens 80 --tag screen
../.venv/bin/python analyze.py screen          # per-neuron trait means and z vs controls
../.venv/bin/python build_confirm.py           # finalists -> dose sweep on HOLDOUT
```

`hunt/pipeline.sh` and `hunt/phase*.sh` are the unattended overnight scripts in the order they ran; the
lab log that narrates them is `results/hunt/NOTEBOOK.md`. Prompt sets (TRAIN / SCREEN / HOLDOUT / HOLDOUT2)
are in `hunt/prompts.py`; the 16 Jev trait questions are in `hunt/traits.py`.

`collect.py` caches per-layer residual means to `out/hunt/resid_means.npy` (293 MB). That file is not
committed; rerun `collect.py` to regenerate it.

## Content warning

`results/` contains raw model generations. The evil-neuron runs at dose 1.0 include violent text, some of it
about children. Do not quote it in public material; the video rule was to describe it, never read it.

## Related

- [jcpsimmons/qwen-evil-neuron](https://github.com/jcpsimmons/qwen-evil-neuron): the public repo this one
  was forked from (history preserved). Public-facing demo only.
- [jcpsimmons/ai-gay-switch](https://github.com/jcpsimmons/ai-gay-switch): earlier work with the same dose
  recipe on Qwen3-0.6B, including the first use of cosine ranking against `W_down` columns.
