"""
Sub-step 3.3 — True Entropy Calculation.

"Run a computationally heavy offline baseline. Force the LLM to output
5 to 10 distinct answers per query for a batch of 500 questions. Cluster
these answers by semantic equivalence to calculate the true mathematical
uncertainty/entropy."

This implements bidirectional-entailment clustering + discrete semantic
entropy exactly as defined in Kuhn et al. 2023 ("Semantic Uncertainty",
ICLR) and Farquhar et al. 2024 (Nature, "Detecting hallucinations in
large language models using semantic entropy") — the paper the Sub-step
3.4 probe (a "Semantic Entropy Probe" / SEP, Kossen et al. 2024) is
designed to approximate cheaply.

INSTALLED FROM THE ACTUAL REFERENCE CODE (not a from-scratch guess):
the clustering rule (`_are_equivalent`), the NLI model choice
(`microsoft/deberta-v2-xlarge-mnli`), and the entropy formula
(`cluster_assignment_entropy`) below are ported line-for-line from
Anthropic's read of the authors' own repos:
  - https://github.com/OATML/semantic-entropy-probes
    (semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py)
  - https://github.com/jlko/semantic_uncertainty (predecessor/same code)

This is intentionally decoupled from Engineer 4's MiniCheck-FT5
faithfulness model (Sub-step 4.2): that model checks "is this answer
grounded in a retrieved chunk", this one checks "do these two answers
mean the same thing", which is a symmetric-equivalence task, not a
grounding task.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import SemanticEntropyConfig


@dataclass
class EntropyResult:
    query: str
    answers: list[str]
    cluster_ids: list[int]        # which semantic cluster each answer fell into
    n_clusters: int
    semantic_entropy: float       # discrete entropy over cluster distribution, nats


class BidirectionalEntailmentClusterer:
    """
    Two answers are merged into the same cluster iff the NLI model
    predicts entailment in BOTH directions (A->B and B->A), which is
    the standard test for semantic equivalence used in the source
    papers — plain unidirectional entailment is not sufficient (e.g.
    "Paris" entails "a French city" but not vice versa).
    """

    def __init__(self, cfg: SemanticEntropyConfig = SemanticEntropyConfig()):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.nli_model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.nli_model_id)
        self.model.eval()
        # deberta-*-mnli label order: 0=contradiction, 1=neutral, 2=entailment
        # (verified against OATML/semantic-entropy-probes ->
        # semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py)

    @torch.no_grad()
    def _check_implication(self, premise: str, hypothesis: str) -> int:
        """Returns the argmax MNLI class (0/1/2), matching the official
        EntailmentDeberta.check_implication — a hard 3-way vote per
        direction, not a soft threshold on P(entail)."""
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        logits = self.model(**inputs).logits
        return int(torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item())

    def _are_equivalent(self, a: str, b: str) -> bool:
        """
        Ported directly from the official get_semantic_ids() in
        OATML/semantic-entropy-probes (itself following Farquhar et al.
        2024 / Kuhn et al. 2023). Two directions are checked (a->b and
        b->a) because entailment is not symmetric ("Paris" entails "a
        French city" but not vice versa).
        """
        implication_ab = self._check_implication(a, b)
        implication_ba = self._check_implication(b, a)

        if self.cfg.strict_entailment:
            # Both directions must be full entailment (class 2).
            return implication_ab == 2 and implication_ba == 2

        # Non-strict (paper default): equivalent unless either direction
        # is a contradiction (0), or both directions are merely neutral (1).
        implications = [implication_ab, implication_ba]
        return (0 not in implications) and (implications != [1, 1])

    def cluster(self, answers: list[str]) -> list[int]:
        """Greedy clustering, ported from the official get_semantic_ids():
        O(n^2) NLI calls, fine at n=5-10 answers per query."""
        cluster_ids = [-1] * len(answers)
        next_id = 0
        for i, ans in enumerate(answers):
            if cluster_ids[i] != -1:
                continue
            cluster_ids[i] = next_id
            for j in range(i + 1, len(answers)):
                if cluster_ids[j] == -1 and self._are_equivalent(ans, answers[j]):
                    cluster_ids[j] = next_id
            next_id += 1
        return cluster_ids


def cluster_assignment_entropy(cluster_ids: list[int]) -> float:
    """
    Ported from the official `cluster_assignment_entropy()`: Shannon
    entropy (nats) over the empirical distribution of answers across
    semantic clusters. This is the exact target Kossen et al.'s SEP is
    trained to regress/classify from a single hidden state.
    """
    n = len(cluster_ids)
    counts: dict[int, int] = {}
    for c in cluster_ids:
        counts[c] = counts.get(c, 0) + 1
    entropy = 0.0
    for c, count in counts.items():
        p = count / n
        entropy -= p * math.log(p)
    return entropy


# Backward-compatible alias used elsewhere in this project.
discrete_semantic_entropy = cluster_assignment_entropy


def probability_weighted_semantic_entropy(cluster_ids: list[int], sequence_log_probs: list[float]) -> float:
    """
    The OTHER semantic-entropy formula — the one the team's own Metrics
    Deep-Dive PDF actually writes out in Section 4:
        "SE = -sum_c p(c) * log p(c), where p(c) is the aggregated
        probability mass of cluster c (summed sequence probabilities of
        its members, renormalized across clusters)."
    This is Kuhn/Farquhar's full formula, using each sampled answer's
    own sequence log-probability as its weight, NOT just a uniform 1/N
    count per member. It requires `sequence_log_probs` (log p(answer_i))
    from the SAME generation call that produced each answer — Engineer 2
    already computes per-token probabilities for the router's silver
    labels (Sub-step 2.3), so these are cheap to also expose here.

    `cluster_assignment_entropy` (used by compute_true_entropy.py to
    build the SEP training target, matching Kossen et al. 2024's own
    released training code) is a DIFFERENT, simpler variant: it treats
    every sampled answer as equally weighted regardless of how
    confident the model was in it, which is also what makes it usable
    for black-box models that don't expose log-probabilities.

    Both are legitimate "semantic entropy" per the surveyed papers; the
    project trains SEPs against the discrete/count-based target because
    that's what the actual SEP reference implementation trains against.
    This function is provided so the fully-weighted version is available
    if the team wants to report it (e.g. for the AUROC comparison in
    Section 7.1 of the Metrics Deep-Dive, or to double-check the
    discrete target isn't losing much versus the weighted one).
    """
    import numpy as np

    unique_ids = sorted(set(cluster_ids))
    log_probs = np.array(sequence_log_probs)
    # Renormalize across all clusters first (soft-max over sequence log-probs).
    log_total = np.log(np.sum(np.exp(log_probs)))
    cluster_log_masses = []
    for uid in unique_ids:
        member_log_probs = [log_probs[i] for i, c in enumerate(cluster_ids) if c == uid]
        cluster_log_mass = np.log(np.sum(np.exp(member_log_probs))) - log_total
        cluster_log_masses.append(cluster_log_mass)
    cluster_log_masses = np.array(cluster_log_masses)
    probs = np.exp(cluster_log_masses)
    return float(-(probs * cluster_log_masses).sum())


def compute_entropy_for_query(query: str, sampled_answers: list[str],
                               clusterer: BidirectionalEntailmentClusterer) -> EntropyResult:
    cluster_ids = clusterer.cluster(sampled_answers)
    entropy = discrete_semantic_entropy(cluster_ids)
    return EntropyResult(
        query=query,
        answers=sampled_answers,
        cluster_ids=cluster_ids,
        n_clusters=len(set(cluster_ids)),
        semantic_entropy=entropy,
    )
