# Paper plan

Working title: **Persona Neurons: Single-Neuron Steering of Personality in a 27B Language Model**.
Author: Dr. Josh C. Simmons (independent). Target: arXiv cs.LG / cs.CL, with an interpretability framing.

## One-sentence thesis

In Qwen3.8-27B a dozen individually addressable MLP neurons each install a coherent personality when pushed
along their write vector, they can be found cheaply by cosine against a persona direction, and this says
personality in the chat model is a thin layer written by a handful of directions.

## Claims we can already support (from results/hunt/FINDINGS.md)

1. The viral evil neuron replicates on 8-bit weights with a same-layer control and a classifier judge.
2. Cosine(W_down column, persona residual direction) finds persona neurons; activation-contrast t does not.
3. About 13 confirmed single-neuron personas on held-out prompts, with doses, coherence, on-topic.
4. L33 N3570 is a monotone, bidirectional affect dial.
5. Layer 34 holds three switches. Personas live in layers ~16-49; early layers break, late layers do punctuation.
6. Controls: random same-layer neurons at the same dose never exceed trait z ~3.
7. Negative results: gradient attribution through Gated DeltaNet, early/late layers, stacking hurts coherence.

## Experiments still needed before submission (priority order)

1. **Judge validation.** Hand-label ~200 held-out replies (blind to condition) and report agreement with Jev.
   Add a second judge (any frontier model) on the same sample.
2. **Sampling.** Rerun the confirmed neurons at temperature 0.7, 5 samples per prompt, to show the effect is
   not a greedy-decoding artefact. Report per-prompt hit rates, not just mean P.
3. **Bigger held-out set.** 48+ prompts per confirmation so P has a confidence interval worth printing.
4. **Matched-cosine random directions.** For each confirmed neuron, steer along (a) a random unit vector,
   (b) a random direction with the same cosine to the persona vector, (c) the full persona vector itself,
   all at the same norm. Tells us whether "neuron" matters or only "direction".
5. **Capability cost.** A small benchmark (MMLU subset or GSM8K subset) at each persona's best dose.
6. **Second model.** Qwen3.5-4B (lyra's original) or another family, same pipeline, to say whether the
   catalogue is Qwen-specific. bf16 vs 8-bit on at least the top three neurons.
7. **Layer sweep figure.** Hit rate and coherence by layer at fixed dose; dose curves for the top six.
8. **Ablation of the hunt.** Hit rate of cosine-vs-others vs cosine-vs-neutral vs paired t, per persona.

## Figures planned

- Fig 1: schematic of the push (one W_down column added to the residual stream every token).
- Fig 2: L33 N3570 dose curve (joyful, coherent, words) with control band.
- Fig 3: hit rate by layer, confirmed neurons marked.
- Fig 4: trait-probability heatmap, confirmed neurons x 16 traits, held-out.
- Table 1: the catalogue (already in results.tex).

## Framing notes

- Lead with the method and the catalogue, not the viral thread; the thread goes in the intro as motivation.
- Keep Soares's point exact: the lesson is "thin facade over machinery nobody understands", not "two tweaks
  from evil".
- Do not reproduce the dose-1.0 violent text. Quote only the examples already in FINDINGS.md.
- Every number in the draft comes from results/hunt/FINDINGS.md or NOTEBOOK.md; anything new gets a run
  tag and a line in NOTEBOOK.md.
- Verify every bib entry against the arXiv record before submission.

## Writing checklist

- [ ] Intro drafted
- [ ] Related work verified
- [ ] Method: fill in collect.py persona-generation procedure
- [ ] Results: add new-run numbers, figures
- [ ] Discussion drafted
- [ ] Appendices: prompts, judge questions, examples
- [ ] dr-j-voice pass is NOT required (academic register), but run the unslop pass on the final text
