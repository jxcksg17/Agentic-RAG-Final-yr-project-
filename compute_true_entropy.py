"""
Entrypoint for Sub-step 3.3.

    python compute_true_entropy.py

For 500 calibration queries: sample 5-10 answers each at temperature>0,
cluster them by bidirectional NLI entailment, compute discrete semantic
entropy. Also records EM-based correctness of a separate greedy answer
for the same query.

Writes a CSV of (query, semantic_entropy, is_correct) — the entropy
column is the regression/classification TARGET consumed by
train_sep.py (Sub-step 3.4); the correctness column exists so
train_sep.py can run the Type-A validation the Metrics Deep-Dive PDF
requires: "AUROC — does the semantic-entropy score separate correct
answers (low entropy) from incorrect/hallucinated ones (high entropy),
measured against ground-truth correctness labels." Without a
correctness column, the probe can only be checked against its own
training target, which isn't the validation the paper actually reports.

This script is the expensive, offline-only baseline; it is never run in
production (that's the entire point of Sub-step 3.4's cheap probe).
"""

import pandas as pd
from tqdm import tqdm

from config import Paths, ModelConfig, SemanticEntropyConfig
from src.hooks import HiddenStateExtractor
from src.semantic_entropy import BidirectionalEntailmentClusterer, compute_entropy_for_query
from src.data_utils import load_query_bank, is_correct


def sample_answers(extractor: HiddenStateExtractor, query: str, cfg: SemanticEntropyConfig) -> list[str]:
    inputs = extractor.tokenizer(query, return_tensors="pt").to(extractor.model.device)
    out_ids = extractor.model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=cfg.sampling_temperature,
        top_p=cfg.sampling_top_p,
        num_return_sequences=cfg.n_samples_per_query,
        pad_token_id=extractor.tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    return [extractor.tokenizer.decode(ids[prompt_len:], skip_special_tokens=True) for ids in out_ids]


def generate_greedy_answer(extractor: HiddenStateExtractor, query: str) -> str:
    """Separate, deterministic answer used ONLY for the correctness
    label — kept independent of the temperature-sampled answers used
    for clustering, so correctness isn't biased by which sample happened
    to get clustered as the majority."""
    inputs = extractor.tokenizer(query, return_tensors="pt").to(extractor.model.device)
    out_ids = extractor.model.generate(
        **inputs, max_new_tokens=64, do_sample=False, pad_token_id=extractor.tokenizer.eos_token_id
    )
    prompt_len = inputs["input_ids"].shape[1]
    return extractor.tokenizer.decode(out_ids[0][prompt_len:], skip_special_tokens=True)


def main():
    paths, model_cfg, ent_cfg = Paths(), ModelConfig(), SemanticEntropyConfig()

    df = load_query_bank(paths.raw_queries_csv)
    calibration_df = df.sample(n=min(ent_cfg.n_calibration_queries, len(df)), random_state=42)

    clusterer = BidirectionalEntailmentClusterer(ent_cfg)

    rows = []
    with HiddenStateExtractor(model_cfg) as extractor:
        for row in tqdm(calibration_df.itertuples(), total=len(calibration_df),
                         desc="Sub-step 3.3: sampling + clustering"):
            answers = sample_answers(extractor, row.query, ent_cfg)
            result = compute_entropy_for_query(row.query, answers, clusterer)

            greedy_answer = generate_greedy_answer(extractor, row.query)
            correct = is_correct(greedy_answer, row.gold_answer)

            rows.append({
                "query": row.query,
                "n_clusters": result.n_clusters,
                "semantic_entropy": result.semantic_entropy,
                "is_correct": int(correct),   # ground-truth label for Type-A validation
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(paths.entropy_labels_csv, index=False)
    print(f"Wrote {len(out_df)} true-entropy labels -> {paths.entropy_labels_csv}")
    print(out_df["semantic_entropy"].describe())
    print(f"Correctness rate on greedy answers: {out_df['is_correct'].mean():.3f}")

    # This IS the paper's own Type-B-validated-by-Type-A check, run on
    # the *full* (expensive) semantic entropy before any probe is
    # trained -- gives the ceiling that train_sep.py's probe AUROC
    # should be compared against.
    try:
        from sklearn.metrics import roc_auc_score
        full_se_auroc = roc_auc_score(1 - out_df["is_correct"], out_df["semantic_entropy"])
        print(f"Full semantic-entropy AUROC vs. ground-truth correctness: {full_se_auroc:.4f} "
              f"(this is the ceiling for the SEP's own AUROC in train_sep.py)")
    except ValueError as e:
        print(f"Could not compute full-SE AUROC (likely single-class correctness in this sample): {e}")


if __name__ == "__main__":
    main()
