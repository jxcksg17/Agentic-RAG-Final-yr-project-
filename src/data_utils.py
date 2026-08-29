"""
Small shared helpers so the two entrypoint scripts stay thin.
Replace `is_correct()` with your actual answer-matching logic (exact
match / F1 / the tie-handling normalization Engineer 1 builds in
Sub-step 1.4) — a stub is provided so the pipeline is runnable end to end.
"""

from __future__ import annotations

import re
import pandas as pd


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def is_correct(predicted: str, gold: str) -> bool:
    """Simple normalized exact-match. Swap for Engineer 1's
    tie-handling-aware scorer (Sub-step 1.4) once available."""
    return normalize_answer(predicted) == normalize_answer(gold)


def token_f1(predicted: str, gold: str) -> float:
    """
    Token-overlap F1, per the Metrics Deep-Dive PDF: both Adaptive-RAG
    and Probing-RAG report EM *and* F1 at evaluation time, not EM alone
    ("EM, F1 on QA; retrieval-step efficiency" / "EM, F1; retrieval-call
    savings" in the per-paper metric table). EM alone under-reports
    partial credit on span-style QA answers.
    """
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common: dict[str, int] = {}
    for t in pred_tokens:
        if t in gold_tokens:
            common[t] = common.get(t, 0) + 1
    num_same = sum(min(c, gold_tokens.count(t)) for t, c in common.items())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def load_query_bank(csv_path: str) -> pd.DataFrame:
    """
    Expects Engineer 1's cleaned-query handoff format:
    columns: ['query', 'gold_answer']
    """
    df = pd.read_csv(csv_path)
    required = {"query", "gold_answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"query bank missing columns: {missing}")
    return df
