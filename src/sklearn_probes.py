"""
Reference-faithful probe architecture — installs Kossen et al. 2024's
ACTUAL published architecture rather than a reinterpretation of it.

Source (read directly from the authors' repo, not from memory):
https://github.com/OATML/semantic-entropy-probes
  -> semantic_entropy_probes/train-latent-probe.ipynb

Key facts pulled from that notebook that this module reproduces exactly:
  1. Both the "accuracy probe" (correctness) and the "SEP" (semantic
     entropy) are `sklearn.linear_model.LogisticRegression` — a strict
     linear classifier, not a PyTorch MLP.
  2. The SEP target is the entropy value BINARIZED (paper uses a
     median split into "high"/"low" uncertainty), not the raw
     continuous value — trained with log-loss, evaluated with
     accuracy + AUROC.
  3. The notebook's own bootstrap analysis concatenates a WINDOW of
     adjacent layers as features (`layer_range`) rather than committing
     to one layer — this beats any single layer and is exposed below
     as `layer_window`.
  4. Two token positions are compared: TBG (token-before-generation)
     and SLT (second-last-token of the model's own answer, i.e. the
     token right before EOS) — SLT is generally the stronger predictor
     for the SEP task specifically, at the cost of needing a full
     generation pass first (see src/hooks.py::extract_tbg_and_slt).

This module is deliberately separate from src/phase1_gate.py and
src/sep_probe.py (the PyTorch versions matching the literal team
brief in work_Dist.pdf) so both readings — "the brief as written" and
"the paper as published" — are available and comparable side by side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass
class ProbeEvalResult:
    val_accuracy: float
    val_auroc: float
    train_log_loss: float
    val_log_loss: float


def binarize_at_median(values: np.ndarray) -> np.ndarray:
    """Median-split into {0, 1}, exactly as the official notebook's
    `b_entropy` target does for the SEP."""
    median = np.median(values)
    return (values > median).astype(int)


def train_logistic_probe(X: np.ndarray, y: np.ndarray, val_split: float = 0.15,
                          random_state: int = 42) -> tuple[LogisticRegression, ProbeEvalResult]:
    """
    Direct port of the official `sklearn_train_and_evaluate()`. Used for
    BOTH the accuracy probe and the SEP — only the label vector differs.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    train_probs = model.predict_proba(X_train)
    train_loss = log_loss(y_train, train_probs)

    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)
    val_loss = log_loss(y_val, val_probs)
    val_acc = float(np.mean(val_preds == y_val))
    val_auroc = float(roc_auc_score(y_val, val_probs[:, 1]))

    result = ProbeEvalResult(
        val_accuracy=val_acc, val_auroc=val_auroc,
        train_log_loss=train_loss, val_log_loss=val_loss,
    )
    print(f"[reference-arch] val_acc={val_acc:.4f} val_auroc={val_auroc:.4f} "
          f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
    return model, result


def train_sep_reference(hidden_states: np.ndarray, raw_entropy: np.ndarray,
                         val_split: float = 0.15) -> tuple[LogisticRegression, ProbeEvalResult, float]:
    """
    Sub-step 3.4, paper-faithful path: binarize Sub-step 3.3's raw
    entropy at the median, train LogisticRegression to classify
    high/low semantic uncertainty from a single hidden state.
    Returns (model, eval_result, median_threshold) — the threshold must
    be saved, it's needed to interpret the model's binary output at
    inference time.
    """
    median = float(np.median(raw_entropy))
    y_bin = binarize_at_median(raw_entropy)
    model, result = train_logistic_probe(hidden_states, y_bin, val_split)
    return model, result, median


def train_accuracy_probe_reference(hidden_states: np.ndarray, correctness_labels: np.ndarray,
                                    val_split: float = 0.15) -> tuple[LogisticRegression, ProbeEvalResult]:
    """
    Sub-step 3.2's paper-faithful counterpart: the official notebook's
    "accuracy probe" baseline. Compare its val_accuracy/val_auroc
    against src/phase1_gate.py's MLP (the brief's literal <10M-param
    MLP requirement) to see whether the extra MLP capacity actually
    buys anything over a plain linear probe for THIS gate task.
    """
    return train_logistic_probe(hidden_states, correctness_labels, val_split)


def layer_window_indices(center: int, window: tuple[int, int]) -> list[int]:
    """Inclusive layer range around `center`, clipped to non-negative."""
    lo, hi = window
    return list(range(max(lo, 0), hi + 1))


def evaluate_vs_ground_truth(predicted_score: np.ndarray, true_entropy: np.ndarray,
                              is_correct: np.ndarray) -> dict:
    """
    The Type-B-validated-by-Type-A check the Metrics Deep-Dive PDF
    specifies for BOTH full semantic entropy and SEPs:
      "AUROC — does the semantic-entropy score separate correct answers
      (low entropy) from incorrect/hallucinated ones (high entropy),
      measured against ground-truth correctness labels."
      "Correlation coefficient between probe output and true semantic
      entropy, as a fidelity check."
    `predicted_score` can be either a probe's raw predicted entropy
    (continuous) or P(high uncertainty) (the sklearn SEP's output) —
    both are monotonically related to "how uncertain", so AUROC against
    correctness is well-defined either way.
    """
    from sklearn.metrics import roc_auc_score

    result = {}
    # Higher predicted_score should mean higher chance of being WRONG.
    incorrect = 1 - is_correct
    if len(set(incorrect.tolist())) > 1:
        result["auroc_vs_correctness"] = float(roc_auc_score(incorrect, predicted_score))
    else:
        result["auroc_vs_correctness"] = float("nan")  # single-class calibration sample

    if np.std(predicted_score) > 0 and np.std(true_entropy) > 0:
        result["pearson_corr_vs_true_se"] = float(np.corrcoef(predicted_score, true_entropy)[0, 1])
    else:
        result["pearson_corr_vs_true_se"] = float("nan")

    return result
