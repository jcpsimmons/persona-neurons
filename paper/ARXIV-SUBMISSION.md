# arXiv submission sheet

Paste these into the arXiv form. Regenerate the abstract after the baseline results land (numbers may move).

## Order of operations

1. Log in at arxiv.org, Start New Submission. Pick cs.LG as primary. arXiv will say you need endorsement and give you a code and a link for the endorser.
2. Paste the code or link into the session. The draft to Apostol Vassilev (josh@ Drafts, subject "arXiv endorsement request (cs.LG)") gets revised to carry it. Send it yourself.
3. Upload `paper/persona-neurons-arxiv.tar.gz` (`make arxiv` rebuilds it). arXiv compiles it; the tarball compiles clean standalone, 34 pages.
4. Leave the submission unfinalized until the baseline results are folded in, then replace the file and press Submit.
5. Before 14:00 Eastern on a weekday announces that night at 20:00 Eastern. New submitters sometimes sit in moderation for a few days.

## Metadata

**Title:** Persona Neurons: Single-Neuron Steering of Personality in a 27B Language Model

**Authors:** Josh C. Simmons (Plausible Deniability Labs)

**Primary category:** cs.LG. **Cross-lists:** cs.CL, cs.AI

**Comments:** 34 pages, 2 figures

**License:** arXiv.org perpetual, non-exclusive license (default), or CC BY 4.0 for maximum reuse

**Abstract** (1891 characters; arXiv allows 1,920):

In October 2026 an X account posted that adding a fixed offset to one MLP neuron in Qwen3.8-27B (L34 N3838, dose 0.75) turns the assistant hostile while leaving it fluent. We replicate the claim on 8-bit weights with random same-layer neurons as controls, and ask whether the model carries other single-neuron handles on personality. Using the same scale-free recipe, we rank MLP neurons by the cosine between each neuron's write vector W_down[:,n] and a residual-stream direction built from system-prompted persona text, screen 1,092 candidate neurons at dose 0.75 on six prompts, and confirm 81 finalists on prompt sets they were not selected on, scoring about 12,000 replies with a decision-model judge on 16 trait questions. Under one pre-stated rule (judge probability for the top trait at least .5 at the screening dose, a bootstrap interval over controls excluding zero, coherence at least .6), 21 neurons pass, plus the original. They install a conspiracy narrator, a heartbroken ex, a pirate, a narcissist, a sarcastic cynic, several registers of mysticism, a refuser, and a layer-33 neuron whose negative dose produces giddy joy and whose positive dose halves reply length. Per-prompt hit rates run from 6 to 12 of 12. A blind human labeler agrees with the judge's top trait on all four sampled replies for 5 of 15 sampled conditions and on at most one for 4; per-trait agreement is strong for pirate, bored, evil and mystical and poor for grandiose. The passing neurons lie in layers 16 to 48, with three each in layers 32, 33 and 34; no neuron from the 247 screened outside that band passed. The cosine ranking would not have found the original evil neuron, whose write vector is nearly orthogonal to the system-prompted evil direction. We state what a forced push of this size does and does not show about the model's ordinary behaviour, and we list the baselines still missing.
