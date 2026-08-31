# Engineer 3 — Latent Probing & Uncertainty Specialist

Complete build for Engineer 3's role in the Adaptive Agentic RAG system:
"Intercept internal LLM states to build zero-latency confidence probes
for Phase 1 and Phase 3."

## How this maps to the two source documents

| Sub-step (work_Dist.pdf) | File | Feeds into (methodology PDF) |
|---|---|---|
| 3.1 Tensor Interception | `src/hooks.py` | shared primitive, all phases |
| 3.2 Phase 1 Microscopic Gate | `src/phase1_gate.py`, `train_phase1_gate.py` | **Router Agent (Phase 1)** — "hidden-state probe to decide ... whether retrieval is needed at all" |
| 3.3 True Entropy Calculation | `src/semantic_entropy.py`, `compute_true_entropy.py` | offline label generation only, never runs in production |
| 3.4 Semantic Entropy Probe Training | `src/sep_probe.py`, `train_sep.py` | **Evaluator Agent (Phase 3)** — "entropy/uncertainty scorer" in Table 1 |

`inference.py` is the only file the other engineers need to import —
it exposes `phase1_needs_retrieval()` for Engineer 2's Router Agent and
`entropy_uncertainty_score()` for Engineer 4's fused Evaluator.

## What's actually installed from the base papers (not reinterpreted)

The clustering, entropy formula, and probe architecture below are read
directly from the authors' own released code — cloned and inspected,
not reconstructed from the papers' prose:

- **Semantic Entropy Probes** — Kossen et al. 2024 (arXiv:2406.15927):
  `github.com/OATML/semantic-entropy-probes`
- **Semantic entropy / bidirectional-entailment clustering** —
  Farquhar et al. 2024 (Nature) / Kuhn et al. 2023 (ICLR):
  `github.com/jlko/semantic_uncertainty` (same clustering code is
  vendored inside the SEP repo above)

Three things this project got from those repos, corrected from an
earlier draft that had guessed at them:

1. **The entailment model is `microsoft/deberta-v2-xlarge-mnli`**, not
   `deberta-large-mnli`. (`src/semantic_entropy.py`)
2. **The clustering rule isn't a soft P(entail) threshold** — it's the
   official `get_semantic_ids()` logic: two answers merge if neither
   direction is a contradiction and both directions aren't merely
   neutral (a `strict_entailment` mode, requiring full entailment both
   ways, is also exposed — `config.SemanticEntropyConfig.strict_entailment`).
3. **The SEP itself is `sklearn.linear_model.LogisticRegression` on
   BINARIZED (median-split) entropy**, trained with log-loss and
   evaluated with accuracy/AUROC — not a from-scratch PyTorch
   regressor on the continuous value. `src/sklearn_probes.py` installs
   this exactly; `src/sep_probe.py` keeps the continuous PyTorch
   version as the literal reading of work_Dist.pdf's own wording
   ("predict the entropy scores"). `train_sep.py` now trains and
   prints both, side by side, on identical hidden states.

## Improvements layered on top of the base architecture

Pulled from the same repos' own ablations, not invented:

- **Layer-window concatenation.** The official notebook's bootstrap
  analysis concatenates a small window of adjacent decoder layers as
  probe features instead of committing to one layer, and reports it
  beating any single layer. `config.SEPConfig.layer_window` (default
  `(14, 18)` around `probe_layer_index=16`) turns this on;
  `src/hooks.py::extract_multi_layer` captures the window in one
  forward pass, `src/sklearn_probes.py::layer_window_indices` resolves it.
- **TBG vs. SLT ablation.** The paper compares two tap points: TBG
  (token-before-generation — the pre-answer state, what Sub-step 3.1
  specifies and what the Phase 1 gate *must* use, since no answer
  exists yet) against SLT (token-before-EOS of the model's *own*
  answer, i.e. after generation). SLT is generally the stronger
  entropy predictor but can't be used pre-generation. `train_sep.py`
  runs both when `config.SEPConfig.compare_tbg_vs_slt=True` and prints
  a comparison table — TBG stays the production default so the probe
  keeps its "zero extra generation cost" property, but the numbers are
  there if the team wants to trade that off for SLT's stronger signal
  in a post-hoc-only use of the Evaluator Agent.
- **Accuracy-probe baseline for the Phase 1 gate.** The official repo's
  own baseline for "will this be correct" is the same
  LogisticRegression architecture, not an MLP.
  `train_phase1_gate.py` now trains that baseline alongside the
  brief's required <10M-param MLP and prints both val accuracies, so
  the MLP's extra capacity has to earn its place rather than being
  assumed better.

## Verified against the team's Metrics Deep-Dive PDF

The team's own literature-survey doc (`Adaptive_Agentic_RAG_Metrics_DeepDive.pdf`)
names the exact base papers Engineer 3's two probes are supposed to be —
that let three things be checked (and corrected) that weren't
verifiable from work_Dist.pdf alone:

1. **Phase 1's gate IS Probing-RAG (Baek et al. 2025), confirmed.**
   The Deep-Dive PDF's own description — "a lightweight probe (a small
   MLP)... BCE loss on hidden-state features extracted from a fixed
   intermediate layer... labeled via EM/accuracy against gold" — matches
   `src/phase1_gate.py` exactly. The LogisticRegression comparison in
   `train_phase1_gate.py` was previously mislabeled as "the reference
   architecture" for this probe; it's actually an unrelated baseline
   borrowed from the SEP repo, kept only as a sanity check. Comment
   fixed to say so — Probing-RAG's own architecture is, and always was,
   the MLP.
2. **Missing metric added: F1.** The Deep-Dive PDF lists Probing-RAG's
   evaluation metrics as "EM, F1; retrieval-call savings" — the project
   was only tracking EM. `src/data_utils.py::token_f1` and
   `train_phase1_gate.py` now report both.
3. **Missing metric added: AUROC / correlation against ground truth for
   the SEP.** The PDF is explicit that a Type-B (reference-free) signal
   like semantic entropy is only trustworthy once checked against Type-A
   ground truth: "AUROC — does the semantic-entropy score separate
   correct answers... from incorrect... measured against ground-truth
   correctness labels" plus "Correlation coefficient between probe
   output and true semantic entropy, as a fidelity check." The project
   previously only validated the SEP against its own training target
   (does it predict binarized-SE correctly), which isn't that check.
   `compute_true_entropy.py` now also records a correctness label per
   calibration query and reports the full-SE AUROC ceiling;
   `train_sep.py` now reports both probes' AUROC-vs-correctness and
   correlation-vs-true-SE against that ceiling, via
   `src/sklearn_probes.py::evaluate_vs_ground_truth`.
4. **Formula ambiguity surfaced, not silently resolved.** The Deep-Dive
   PDF's Section 4 literature review writes out the *probability-
   weighted* SE formula (cluster mass = summed sequence probabilities,
   not a uniform count). The actual Kossen et al. training code (read
   from `OATML/semantic-entropy-probes`) trains against the simpler
   *discrete, count-based* `cluster_assignment_entropy` instead — which
   is also the black-box-compatible variant (no log-probabilities
   needed). The project keeps training against the discrete variant,
   matching what SEPs are actually trained on in the reference repo, but
   `src/semantic_entropy.py::probability_weighted_semantic_entropy` is
   now available if the team wants to report the fully-weighted number
   too (it needs Engineer 2's per-sample sequence log-probabilities from
   Sub-step 2.3, which are already being computed there).

## Why not hook vLLM

Engineer 2 serves Phi-4-mini-instruct through vLLM for production
throughput (PagedAttention, continuous batching, 4-bit quant). That
serving path does not expose a stable `nn.Module` graph to hook. So
this pipeline loads the same checkpoint a second time through plain
`transformers`, in the same 4-bit budget, purely for hidden-state
capture. In production, `inference.py`'s `HiddenStateExtractor` forward
pass **is** the model call the pipeline needs anyway for both probes —
it does not duplicate vLLM's generation traffic, it replaces the
"does this need retrieval" and "how confident is this" decisions that
would otherwise cost separate LLM calls.

## Two probes, two shapes, on purpose

- **Phase 1 gate (`Phase1GateMLP`)**: hidden_dim -> 512 -> 128 -> 1,
  <10M params, trained with BCE on correct/incorrect labels. An MLP,
  because "will a no-retrieval answer be correct" is not assumed to be
  linearly decodable.
- **Semantic Entropy Probe (`SemanticEntropyProbe`)**: strictly
  `Linear(hidden_dim, 1)`, ~3K params, trained with MSE against true
  semantic entropy. Deliberately linear, per Kossen et al. 2024
  ("Semantic Entropy Probes: Robust and Cheap Hallucination Detection
  in LLMs") — the paper's core claim is that entropy information is
  *already* linearly present in a single hidden state, which is what
  makes the probe cheap enough to run with zero extra generation cost.

The clustering step that produces the *training target* for the SEP
(`src/semantic_entropy.py`) follows the bidirectional-entailment
definition of semantic equivalence from Kuhn et al. 2023 ("Semantic
Uncertainty", ICLR) and Farquhar et al. 2024 (Nature). This is a
different entailment task from Engineer 4's MiniCheck-FT5 (Sub-step
4.2), which checks answer-vs-retrieved-chunk grounding, not
answer-vs-answer equivalence — the two NLI usages are kept as separate
models on purpose.

## Run order

```bash
pip install -r requirements.txt

# 1. Requires Engineer 1's cleaned query bank at data/engineer1_cleaned_queries.csv
#    (columns: query, gold_answer)
python train_phase1_gate.py        # Sub-steps 3.1 + 3.2 -> artifacts/phase1_gate_probe.pt

# 2. Offline, expensive — 500 queries x 5-10 samples each
python compute_true_entropy.py     # Sub-step 3.3 -> artifacts/true_semantic_entropy.csv

# 3. Cheap, single forward pass per query
python train_sep.py                # Sub-step 3.4 -> artifacts/semantic_entropy_probe.pt
```

Then at inference time (used by Engineers 2 and 4):

```python
from inference import LatentProbes

probes = LatentProbes()                         # loads the reference SEP by default
if probes.phase1_needs_retrieval("What year did the Berlin Wall fall?"):
    ...
uncertainty = probes.entropy_uncertainty_score(some_query)   # P(high SE) in [0,1]

# to use the brief-literal continuous regressor instead:
probes_continuous = LatentProbes(use_reference_sep=False)
entropy_nats = probes_continuous.entropy_uncertainty_score(some_query)
```

## Things worth flagging back to the team

1. **`config.SEPConfig.layer_window = (14, 18)` around `probe_layer_index=16`
   is still a starting guess**, now backed by the paper's own
   "concatenate a window" trick rather than a single arbitrary index —
   but the window's exact bounds haven't been swept on Phi-4-mini-instruct
   specifically (the OATML repo swept it on Llama-2-7B / Mistral-7B /
   Phi-3-mini). Re-run `train_sep.py` at a couple of window widths
   before freezing it.
2. **`is_correct()` in `src/data_utils.py` is a normalized exact-match
   stub.** Swap it for whatever Engineer 1's tie-handling-aware scorer
   (Sub-step 1.4) ends up being, or the gate's labels will be noisier
   than necessary.
3. **`data/engineer1_cleaned_queries.csv` and the gold answers are an
   assumed handoff contract** (columns `query`, `gold_answer`) — confirm
   this matches what Engineer 1 actually writes to SQLite/disk in
   Sub-step 1.6.
4. Both checkpoint formats save their normalization stats /
   thresholds alongside the weights (`inference.py` depends on that —
   don't load a bare `state_dict()` or a bare sklearn model without them).
5. **Decide reference-vs-brief-literal before wiring into Engineer 4's
   Evaluator.** They return different things: the reference SEP returns
   a calibrated `P(high uncertainty)` in [0,1] (composes cleanly with
   Engineer 4's relevance/faithfulness scores and its adaptive-threshold
   step); the brief-literal PyTorch regressor returns raw nats
   (unbounded, needs its own normalization before fusing with the other
   two signals). Recommend the reference path for exactly this reason —
   flag it in the team's Sub-step 4.3 discussion.
