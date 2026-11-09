# Overnight baseline run: plan (written 2026-11-08)

Purpose: run every baseline the adversarial review said was missing, so the next revision can answer "direction
or neuron?", "is it just the system prompt?", "does it survive sampling?", "what does the pipeline find on random
neurons?", and "what does it cost in capability?". Everything runs on the fresh HOLDOUT3 prompt set (12 banal
prompts, disjoint from TRAIN, SCREEN, HOLDOUT, HOLDOUT2), so every passing neuron and every baseline is compared
on prompts none of them has seen.

Code: `hunt/baselines/`. One command, resumable, stops itself at 07:30:

```bash
cd ~/github/persona-neurons && export OPENROUTER_API_KEY=... && nohup zsh hunt/baselines/pipeline.sh > out/baselines/pipeline.out 2>&1 &
```

Watch: `tail -f out/baselines/pipeline.out`. Results: `out/baselines/SUMMARY.md` after `analyze.py`. Archive to
`results/baselines/` when done and add a NOTEBOOK.md line.

## What runs, in order, and what each answers

| Phase | Jobs | Prompts x tokens | Est. time | Answers |
|---|---|---|---|---|
| 1a. Passing neurons, both signs | 44 | 12 x 160 | 20 min | Third-set re-replication of all 22; reversed sign |
| 1b. Random-neuron controls | 30 (2 per passing layer) | 12 x 160 | 12 min | Control distribution on the same set |
| 1c. Random unit direction, same layer, same push length | 66 (3 seeds x 22) | 12 x 160 | 28 min | Is any large push enough? |
| 1d. Matched-cosine random direction | 60 (3 seeds x 20) | 12 x 160 | 25 min | Is the neuron's cosine to the persona direction the whole story? |
| 1e. Persona direction itself, dose .5/.75/1.0 | 63 | 12 x 160 | 26 min | Does the fitted direction beat its best single neuron? |
| 1f. System prompt, unsteered | 13 | 12 x 160 | 5 min | Upper bound: what does the judge give the explicit persona? |
| 2. Null pipeline: screen 100 random neurons (layers 14-48) at .75 on SCREEN, select top 8 by the run's own score, confirm on HOLDOUT, apply the paper's rule | 101 + 41 | 6 x 80, then 12 x 160 | 35 min | False-positive rate of the whole screen-then-confirm procedure |
| 3. Sampling, temperature .7, 5 samples per prompt, passing neurons | 115 | 12 x 160 | 50 min | Is the effect a greedy-decoding artifact? Per-prompt hit rates with variance |
| 4. GSM8K first 50 test problems, baseline + each passing neuron at .75 | 23 | 50 x 320 | 35 min | Capability cost per persona |
| 5. Logit lens of each W_down column (+/-), top 15 tokens | 22 | none | 2 min | Do any columns just write tokens (pirate, exclamation mark)? |
| 6. Readout: activation of each passing neuron on the 14 x 32 persona texts, paired t | 22 neurons | 448 forwards | 10 min | Does the neuron fire more on its persona's natural text? (reviewer finding 18) |

Total about 4 hours of generation plus Jev calls in parallel (about 5,000 calls, roughly $0.30). Rates assume the
observed 6 x 80 tokens in 8 s; 12 x 160 is budgeted at 25 s per job.

## Direction steering (the new piece)

`dirsteer.SteerVec` adds `dose * calib[L] * unit(v)` to the MLP output of layer L. For `v = W_down[:, n]` this is
exactly the neuron recipe (`delta * W_down[:, n]` with `delta = dose * calib / ||W_down[:, n]||`), so neurons,
random directions, matched-cosine directions and persona directions all receive a push of identical length.
Persona directions are the cached `resid_means.npy` contrasts from the hunt (persona minus others, or minus
neutral where the neuron came from that list). Matched-cosine directions are constructed as
`c * d + sqrt(1 - c^2) * g_perp` with `g_perp` random and orthogonal to `d`, so `cos(v, d) = c` exactly, with `c`
the neuron's own |cosine| to its persona direction.

## Decisions fixed in advance

- Dose .75 everywhere except the persona-direction curve (.5/.75/1.0).
- Trait: the judge's top trait from the paper's Table 1 is held fixed for each neuron's comparisons.
- Null pipeline selection uses the run's own exploratory score, then the paper's rule; both reported.
- GSM8K graded by exact match on the last `Answer:` number (fallback: last number in the reply).
- Judge: `typesafe/jev-1.14`, same 16 questions. GSM8K replies are not judged.
- Nothing is filtered on content. Raw rows go to `out/baselines/*.jsonl`.

## Not in this run and why

- bf16: the 27B model in bf16 is about 54 GB, which does not fit beside the OS in 64 GB. Would need a second machine.
- A second human labeler: not a compute task. Ask one person to label the same 123-reply sample blind.
- A second model family: a separate run; the pipeline generalizes but the persona-direction cache would need
  regenerating (`collect.py`, about 15 minutes) for the new model.

## Smoke test

`out/baselines/smoke_jobs.json` runs six jobs (baseline, neuron, matched direction, persona direction, system
prompt, sampled) at 24 tokens; `lens.py` runs in full. Both were run before this plan was committed; see
`out/baselines/smoke.out` and `lens_smoke.out`.
