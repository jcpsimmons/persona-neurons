# Judge validation (2026-11-08)

A blind, stratified sample of 123 held-out replies was labeled by the author in one sitting (about 28 minutes,
blind to neuron, dose and Jev score), then compared with Jev `typesafe/jev-1.14` on the same 16 questions.

- `sample.json`: what the labeler saw (prompt and reply only). `key.json`: the hidden condition and Jev
  probabilities per reply. `labels.json`: the exported human labels. `label.html`: the page used.
- `agreement.md`: per-question accuracy, Cohen's kappa (Jev thresholded at 0.5) and AUROC (Jev probability as
  score). `disagreements.md`: every disagreement with an excerpt, per-persona target-trait hit rates, and the
  controls where the human marked a trait.

Strata: 60 replies from the 15 confirmed conditions (4 each), 22 random same-layer controls, 16 unsteered
baseline, 12 from the affect dial at the flat pole and at -0.5, 13 steered replies whose top Jev trait was
between .3 and .7.

Headline: pooled accuracy .89, pooled kappa .66; per-trait AUROC .85 to 1.00 for 11 of 13 traits. Weak:
grandiose (AUROC .61; Jev fires on rudeness, the human read it as bored or hostile) and joyful (AUROC .76;
the human marked ordinary friendliness in baseline replies that Jev, by its "far beyond normal politeness"
wording, did not). Jev's "strange" fired on 59 replies against the human's 18: it behaves as a
persona-present detector. Jev's coherent and on-topic are stricter than the human's (the human marked 121/123
coherent and 122/123 on topic), so the paper's coherence and on-topic numbers are conservative.

Reproduce: `python3 hunt/validate_judge.py sample && python3 hunt/validate_judge.py page`, label, then
`agree labels.json` and `diff labels.json`.
