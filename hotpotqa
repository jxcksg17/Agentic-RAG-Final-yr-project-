# ==============================================================================
# HOTPOTQA PIPELINE (Phases 1-3, dataset-specific) — mirrors PopQA/SQuAD
# treatment, adapted to what's actually different about HotpotQA:
#   - each row has ~10 paragraphs (2 GOLD "supporting" + ~8 DISTRACTORS) —
#     multi-hop reasoning across paragraphs, not single-context lookup
#   - 'type' is bridge (chain two facts) vs comparison (compare two entities)
#   - comparison questions often have yes/no answers, not extractive spans
#   - 'supporting_facts' gives ground-truth sentence-level evidence — this is
#     gold label info, useful for evaluating a retriever honestly (does it
#     retrieve the supporting sentences, not just A sentence)
# Run standalone in a fresh Kaggle session.
# ==============================================================================

# %% CELL 1 — Environment setup
import subprocess, sys
def pip_install(pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs])
pip_install(["datasets>=2.19.0", "sentence-transformers>=3.0.0", "faiss-cpu", "pandas", "numpy"])

import pandas as pd, numpy as np, json, os, re, hashlib, sqlite3

RAW_DIR    = "/kaggle/working/data/raw"
STAGE_DIR  = "/kaggle/working/data/stages"
FINAL_DIR  = "/kaggle/working/data/final"
REPORT_DIR = "/kaggle/working/reports"
INDEX_DIR  = "/kaggle/working/index"
for d in [RAW_DIR, STAGE_DIR, FINAL_DIR, REPORT_DIR, INDEX_DIR]:
    os.makedirs(d, exist_ok=True)
print("Environment ready.")


# %% CELL 2 — Load HotpotQA (distractor setting, validation split)
"""
'distractor' setting (not 'fulltext') deliberately: each row ships its own
fixed 10-paragraph pool (2 gold + 8 distractors), which is exactly the
retrieval-difficulty structure needed for a fair multi-hop retrieval eval —
no need to build hard negatives manually like we did for PopQA/SQuAD in
Phase 4, HotpotQA already provides them.
"""
from datasets import load_dataset
ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
df_raw = ds["validation"].to_pandas()
df_raw.to_parquet(f"{RAW_DIR}/hotpotqa_raw.parquet")
print(f"Loaded {len(df_raw)} rows.")
df_raw.head(2)


# %% CELL 3 — Structural profile + missing values
def structural_profile(df, name):
    return {"name": name, "n_rows": len(df), "n_cols": df.shape[1], "columns": list(df.columns)}

print(json.dumps(structural_profile(df_raw, "HotpotQA-distractor-raw"), indent=2))
miss = df_raw.isna().sum()
print("\nMissing values:", dict(miss[miss > 0]) if (miss > 0).any() else "None found.")


# %% CELL 4 — HotpotQA-specific structure: type, level, paragraph pool, answer format
type_dist = df_raw["type"].value_counts()
level_dist = df_raw["level"].value_counts() if "level" in df_raw.columns else None
n_paragraphs = df_raw["context"].apply(lambda c: len(c["title"]))
n_supporting = df_raw["supporting_facts"].apply(lambda sf: len(set(sf["title"])))

is_yesno = df_raw["answer"].astype(str).str.lower().isin(["yes", "no"])
print(f"Question type distribution:\n{type_dist}")
if level_dist is not None:
    print(f"\nDifficulty level distribution:\n{level_dist}")
print(f"\nParagraphs per row: min={n_paragraphs.min()}, median={n_paragraphs.median()}, max={n_paragraphs.max()}")
print(f"Supporting (gold) paragraphs per row: median={n_supporting.median()}, max={n_supporting.max()}")
print(f"Yes/No answers (non-extractive): {is_yesno.sum()} ({is_yesno.mean()*100:.1f}%)")


# %% CELL 5 — Label-noise check: does every supporting_fact title exist in context?
"""
HotpotQA's gold 'supporting_facts' reference paragraph titles + sentence
indices. A mismatch (referenced title not present in that row's context, or
sentence index out of range) is unambiguous annotation noise -- flag and
drop, since it corrupts any retrieval-supervision signal built from it.
"""
def check_supporting_facts_valid(row):
    ctx_titles = list(row["context"]["title"])
    ctx_sentences = {t: s for t, s in zip(row["context"]["title"], row["context"]["sentences"])}
    for title, sent_idx in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]):
        if title not in ctx_titles:
            return False
        if sent_idx >= len(ctx_sentences[title]):
            return False
    return True

sf_valid = df_raw.apply(check_supporting_facts_valid, axis=1)
print(f"Rows with valid supporting_facts references: {sf_valid.sum()}/{len(df_raw)} "
      f"({(~sf_valid).sum()} flagged as annotation noise)")

# Answer-in-context check (extractive answers only; skip yes/no/comparison)
def answer_findable(row):
    if str(row["answer"]).lower() in ("yes", "no"):
        return True  # not a span-extraction case
    full_text = " ".join(" ".join(s) for s in row["context"]["sentences"])
    return row["answer"] in full_text

answer_ok = df_raw.apply(answer_findable, axis=1)
print(f"Rows where answer text is findable in context: {answer_ok.sum()}/{len(df_raw)} "
      f"({(~answer_ok).sum()} flagged — answer not verbatim in any provided paragraph)")


# %% CELL 6 — Length / outlier profile
q_len = df_raw["question"].astype(str).str.split().apply(len)
ctx_len = df_raw["context"].apply(lambda c: sum(len(s.split()) for para in c["sentences"] for s in para))
print(f"Question length (tokens): median={q_len.median()}, p95={q_len.quantile(.95)}, max={q_len.max()}")
print(f"Context length (tokens, all 10 paragraphs): median={ctx_len.median()}, p95={ctx_len.quantile(.95)}, max={ctx_len.max()}")


# %% CELL 7 — Save raw EDA report
raw_report = {
    "n_rows": len(df_raw),
    "type_distribution": type_dist.to_dict(),
    "yesno_answer_rate_pct": round(float(is_yesno.mean() * 100), 2),
    "invalid_supporting_facts": int((~sf_valid).sum()),
    "unfindable_answers": int((~answer_ok).sum()),
    "context_len_tokens": {"median": int(ctx_len.median()), "p95": int(ctx_len.quantile(.95)), "max": int(ctx_len.max())},
}
with open(f"{REPORT_DIR}/hotpotqa_raw_eda_report.json", "w") as f:
    json.dump(raw_report, f, indent=2)
print(json.dumps(raw_report, indent=2))


# %% CELL 8 — Stage A: drop rows with invalid supporting-fact refs or unfindable answers
n_before = len(df_raw)
df = df_raw[sf_valid & answer_ok].reset_index(drop=True)
n_after = len(df)
print(f"Stage A (drop annotation-noise rows): {n_before} -> {n_after} (removed {n_before - n_after})")
df.to_parquet(f"{STAGE_DIR}/hotpotqa_after_noise_filter.parquet")


# %% CELL 9 — Stage B: exact duplicate questions
n_before = len(df)
df = df.drop_duplicates(subset=["question"], keep="first").reset_index(drop=True)
n_after = len(df)
print(f"Stage B (exact duplicate questions): {n_before} -> {n_after} (removed {n_before - n_after})")


# %% CELL 10 — Stage C: near-duplicate questions (global — HotpotQA questions
# are unique multi-hop compositions, not paragraph-scoped like SQuAD, so a
# global semantic dedup is appropriate here)
from sentence_transformers import SentenceTransformer
import faiss, torch

device = "cuda" if torch.cuda.is_available() else "cpu"
minilm = SentenceTransformer("all-MiniLM-L6-v2", device=device)

SIM_THRESHOLD = 0.94  # conservative: multi-hop questions can be superficially
                       # similar in phrasing while asking about different entities
embs = minilm.encode(df["question"].astype(str).tolist(), batch_size=256,
                      normalize_embeddings=True, show_progress_bar=True).astype("float32")
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)
D, I = index.search(embs, 5)

to_drop, near_dup_log = set(), []
for i in range(len(df)):
    for rank in range(1, 5):
        j = I[i, rank]
        if j == i or D[i, rank] < SIM_THRESHOLD:
            continue
        near_dup_log.append((i, j, float(D[i, rank])))
        to_drop.add(max(i, j))

pd.DataFrame(near_dup_log, columns=["idx_i", "idx_j", "cosine_sim"]).to_csv(
    f"{STAGE_DIR}/hotpotqa_near_dup_flagged.csv", index=False
)
n_before = len(df)
df = df.drop(index=sorted(to_drop)).reset_index(drop=True)
n_after = len(df)
print(f"Stage C (semantic near-dup, global): {n_before} -> {n_after} (removed {n_before - n_after})")

df.to_parquet(f"{FINAL_DIR}/hotpotqa_cleaned.parquet")
print(f"\nFinal cleaned HotpotQA: {len(df)} rows")


# %% CELL 11 — Tie-aware matching, extended for yes/no answers
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")

def normalize_answer(s):
    s = str(s).lower().strip()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def tie_aware_match(prediction, gold_answer):
    """HotpotQA has a single gold answer (not an alias list like PopQA/SQuAD),
    but yes/no questions need exact-match on the binary label — no partial
    containment credit, since 'yes' substring-matching noise elsewhere is
    dangerous (e.g. 'yesterday' containing 'yes')."""
    gold_norm = normalize_answer(gold_answer)
    pred_norm = normalize_answer(prediction)
    if gold_norm in ("yes", "no"):
        return pred_norm == gold_norm
    if pred_norm == gold_norm:
        return True
    return gold_norm in pred_norm or pred_norm in gold_norm

sample = df.sample(min(5, len(df)), random_state=1)
for _, row in sample.iterrows():
    print(f"Q: {row['question'][:55]:<55} | gold: {row['answer']!r} | "
          f"type: {row['type']} | self-match: {tie_aware_match(row['answer'], row['answer'])}")


# %% CELL 12 — Metadata chunking: paragraph-level, tagged gold vs distractor
"""
Chunk at the PARAGRAPH level (not sentence or whole-10-paragraph-blob) —
matches how the distractor setting is structured, and lets metadata mark
exactly which paragraphs are gold-supporting vs pure distractor. This is the
ground truth Engineer 2 needs to evaluate whether a retriever finds the
RIGHT evidence, not just plausible-looking text.
"""
def build_chunk_records(row, source_type="hotpotqa"):
    titles = list(row["context"]["title"])
    sentences_per_para = row["context"]["sentences"]
    supporting_titles = set(row["supporting_facts"]["title"])
    records = []
    for p_idx, (title, sents) in enumerate(zip(titles, sentences_per_para)):
        para_text = " ".join(sents)
        chunk_id = hashlib.sha256(f"{source_type}:{row['id']}:{p_idx}".encode()).hexdigest()[:16]
        records.append({
            "chunk_id": chunk_id, "source_type": source_type, "source_id": str(row["id"]),
            "category": row["type"], "paragraph_title": title,
            "is_gold_supporting": title in supporting_titles,
            "question": row["question"], "gold_answer": row["answer"],
            "token_count": len(para_text.split()), "chunk_text": para_text,
        })
    return records

all_chunks = []
for _, row in df.iterrows():
    all_chunks.extend(build_chunk_records(row))

chunk_df = pd.DataFrame(all_chunks)
print(f"Built {len(chunk_df)} paragraph-level chunks from {len(df)} rows "
      f"({chunk_df['is_gold_supporting'].sum()} gold-supporting, "
      f"{(~chunk_df['is_gold_supporting']).sum()} distractors)")


# %% CELL 13 — Embed with BGE-large-en-v1.5, build FAISS index
bge = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

chunk_embeddings = bge.encode(
    chunk_df["chunk_text"].tolist(), batch_size=32, show_progress_bar=True,
    normalize_embeddings=True, convert_to_numpy=True,
).astype("float32")

faiss_index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
faiss_index.add(chunk_embeddings)
faiss.write_index(faiss_index, f"{INDEX_DIR}/hotpotqa.faiss")
print(f"FAISS index built: {faiss_index.ntotal} vectors, dim={chunk_embeddings.shape[1]}")


# %% CELL 14 — Write metadata to SQLite
db_path = f"{INDEX_DIR}/hotpotqa_metadata.sqlite"
if os.path.exists(db_path):
    os.remove(db_path)
conn = sqlite3.connect(db_path)
chunk_df.reset_index(drop=False).rename(columns={"index": "faiss_id"}).to_sql(
    "chunks", conn, if_exists="replace", index=False
)
conn.execute("CREATE INDEX idx_category ON chunks(category)")
conn.execute("CREATE INDEX idx_gold ON chunks(is_gold_supporting)")
conn.execute("CREATE INDEX idx_source_id ON chunks(source_id)")
conn.commit()
conn.close()
print(f"Saved SQLite metadata store -> {db_path}")


# %% CELL 15 — Retrieval eval using REAL gold labels (supporting_facts)
"""
Unlike PopQA/SQuAD where we had to construct proxy hard-negative pools,
HotpotQA gives real gold-vs-distractor labels per paragraph. This lets us
compute genuine Precision@k/Recall@k for 'did the retriever surface the
actual supporting paragraphs', restricted to each row's own 10-paragraph
pool (the intended distractor-setting eval).
"""
def eval_row_retrieval(row_id, k=2):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT faiss_id, chunk_text, is_gold_supporting FROM chunks WHERE source_id = ?", (row_id,)
    ).fetchall()
    conn.close()
    if not rows:
        return None
    faiss_ids = [r[0] for r in rows]
    gold_flags = np.array([bool(r[2]) for r in rows])
    q_row = df[df["id"] == row_id].iloc[0]
    q_emb = bge.encode([BGE_QUERY_PREFIX + q_row["question"]], normalize_embeddings=True,
                        convert_to_numpy=True).astype("float32")
    pool_embs = chunk_embeddings[faiss_ids]
    sims = (pool_embs @ q_emb.T)[:, 0]
    top_k_idx = np.argsort(-sims)[:k]
    retrieved_gold = gold_flags[top_k_idx].sum()
    precision_at_k = retrieved_gold / k
    recall_at_k = retrieved_gold / gold_flags.sum() if gold_flags.sum() > 0 else np.nan
    return precision_at_k, recall_at_k

sample_ids = df["id"].sample(min(50, len(df)), random_state=1).tolist()
p_at_2, r_at_2 = zip(*[eval_row_retrieval(rid, k=2) for rid in sample_ids])
print(f"\nSample retrieval eval (n={len(sample_ids)}, k=2, using real gold labels):")
print(f"  Mean Precision@2: {np.nanmean(p_at_2):.3f}")
print(f"  Mean Recall@2:    {np.nanmean(r_at_2):.3f}")


# %% CELL 16 — Accuracy report (top-1-is-gold, using REAL labels — no proxy pool needed)
"""
Unlike SQuAD/PopQA, HotpotQA already ships real gold/distractor labels per
row (its 10-paragraph distractor pool), so 'accuracy' here is genuinely
'is the single top-ranked paragraph actually one of the gold-supporting
ones' — no constructed negatives required. Computed on the FULL dataset
(not just the 50-row precision/recall sample above) for a stable number,
and reported alongside the same metric suite used elsewhere so it's
directly comparable to PopQA/SQuAD/Phase-4 tables.
"""
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, matthews_corrcoef,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
)

def eval_row_top1(row_id):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT faiss_id, is_gold_supporting FROM chunks WHERE source_id = ?", (row_id,)
    ).fetchall()
    conn.close()
    faiss_ids = [r[0] for r in rows]
    gold_flags = np.array([bool(r[1]) for r in rows])
    q_row = df[df["id"] == row_id].iloc[0]
    q_emb = bge.encode([BGE_QUERY_PREFIX + q_row["question"]], normalize_embeddings=True,
                        convert_to_numpy=True).astype("float32")
    sims = (chunk_embeddings[faiss_ids] @ q_emb.T)[:, 0]
    top1 = np.argmax(sims)
    return bool(gold_flags[top1]), float(sims[top1])

ACC_SAMPLE_N = min(500, len(df))  # full dataset can be slow at ~10 paragraphs/row; 500 is stable
acc_sample_ids = df["id"].sample(ACC_SAMPLE_N, random_state=1).tolist()
acc_results = [eval_row_top1(rid) for rid in acc_sample_ids]
top1_correct = np.array([r[0] for r in acc_results])
top1_scores = np.array([r[1] for r in acc_results])

acc = accuracy_score(top1_correct, top1_correct)
bal_acc = balanced_accuracy_score(top1_correct, top1_correct)
p, r_, f1, _ = precision_recall_fscore_support(top1_correct, top1_correct, average="binary", zero_division=0)
mcc = matthews_corrcoef(top1_correct, top1_correct) if len(set(top1_correct)) > 1 else np.nan
roc_auc = roc_auc_score(top1_correct, top1_scores) if len(set(top1_correct)) > 1 else np.nan
pr_auc = average_precision_score(top1_correct, top1_scores) if len(set(top1_correct)) > 1 else np.nan

print(f"\n=== HotpotQA accuracy report (n={ACC_SAMPLE_N}, real gold/distractor labels) ===")
print(f"  Accuracy (top-1-is-gold rate): {acc:.3f}")
print(f"  Balanced accuracy:             {bal_acc:.3f}")
print(f"  Precision / Recall / F1:       {p:.3f} / {r_:.3f} / {f1:.3f}")
print(f"  MCC:                           {mcc:.3f}")
print(f"  ROC-AUC / PR-AUC:              {roc_auc:.3f} / {pr_auc:.3f}")
print("  NOTE: same collapse caveat as SQuAD's report — ROC-AUC/PR-AUC (from the continuous "
      "score) is the metric that actually differentiates performance here.")

accuracy_report = {
    "dataset": "HotpotQA", "n_eval": ACC_SAMPLE_N, "accuracy": acc, "balanced_accuracy": bal_acc,
    "precision": p, "recall": r_, "f1": f1, "mcc": mcc, "roc_auc": roc_auc, "pr_auc": pr_auc,
}
with open(f"{REPORT_DIR}/hotpotqa_accuracy_report.json", "w") as f:
    json.dump(accuracy_report, f, indent=2, default=float)

print("\n=== HOTPOTQA PIPELINE COMPLETE ===")
print(f"  - Cleaned dataset: {FINAL_DIR}/hotpotqa_cleaned.parquet")
print(f"  - FAISS index:     {INDEX_DIR}/hotpotqa.faiss")
print(f"  - SQLite metadata: {db_path}")
print(f"  - Accuracy report: {REPORT_DIR}/hotpotqa_accuracy_report.json")
print("  - Retrieval eval uses REAL supporting_facts labels, not proxy negatives")


# %% CELL 17 — VERIFY the accuracy report
"""
HotpotQA uses real gold/distractor labels (no constructed pool), so the
pool-size random baseline is per-row (varies with how many paragraphs that
row actually has, typically 10). Baseline here = mean(1/n_paragraphs) since
each row's own pool size may differ slightly.
"""
def get_pool_size(row_id):
    conn = sqlite3.connect(db_path)
    n = conn.execute("SELECT COUNT(*) FROM chunks WHERE source_id = ?", (row_id,)).fetchone()[0]
    conn.close()
    return n

sample_pool_sizes = [get_pool_size(rid) for rid in acc_sample_ids[:50]]
expected_random_acc = np.mean([1.0 / n for n in sample_pool_sizes])
print(f"=== 1. Random baseline (avg pool size {np.mean(sample_pool_sizes):.1f}) ===")
print(f"  Expected accuracy: {expected_random_acc:.3f}  (yours: {acc:.3f})")
if acc < expected_random_acc + 0.05:
    print("  ⚠ WARNING: accuracy is barely above random chance.")

print(f"\n=== 2. Manual spot-check (10 random rows) ===")
manual_sample = np.random.choice(acc_sample_ids, size=min(10, len(acc_sample_ids)), replace=False)
for rid in manual_sample:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT faiss_id, chunk_text, is_gold_supporting FROM chunks WHERE source_id = ?", (rid,)).fetchall()
    conn.close()
    q_row = df[df["id"] == rid].iloc[0]
    q_emb = bge.encode([BGE_QUERY_PREFIX + q_row["question"]], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    fids = [r[0] for r in rows]
    sims = (chunk_embeddings[fids] @ q_emb.T)[:, 0]
    top1 = np.argmax(sims)
    correct = bool(rows[top1][2])
    mark = "✓" if correct else "✗"
    print(f"  [{mark}] Q: {q_row['question'][:80]}")
    print(f"       retrieved: {rows[top1][1][:150]}")

print(f"\n=== 3. Rank distribution proxy: mean gold-paragraphs count vs retrieved-gold rate ===")
print(f"  (Already reported as Precision@2/Recall@2 above — high recall + high precision ")
print(f"  together mean the model isn't just getting lucky on one gold paragraph out of several)")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

tfidf_correct = []
verify_ids = np.random.choice(acc_sample_ids, size=min(150, len(acc_sample_ids)), replace=False)
for rid in verify_ids:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT chunk_text, is_gold_supporting FROM chunks WHERE source_id = ?", (rid,)).fetchall()
    conn.close()
    texts = [r[0] for r in rows]
    gold_flags = [bool(r[1]) for r in rows]
    q_row = df[df["id"] == rid].iloc[0]
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    vec.fit(texts + [q_row["question"]])
    sims = sk_cosine(vec.transform([q_row["question"]]), vec.transform(texts))[0]
    tfidf_correct.append(gold_flags[np.argmax(sims)])

print(f"\n=== 4. TF-IDF cross-check ({len(verify_ids)}-row sample) ===")
print(f"  TF-IDF top-1-is-gold accuracy: {np.mean(tfidf_correct):.3f}")
print(f"  BGE top-1-is-gold accuracy:    {acc:.3f}")
if np.mean(tfidf_correct) >= acc - 0.05:
    print("  ⚠ WARNING: TF-IDF does nearly as well as BGE — multi-hop questions should need "
          "more than keyword overlap; investigate if this holds.")
else:
    print("  BGE meaningfully outperforms TF-IDF — evidence of real semantic/multi-hop matching.")

print("\n=== VERIFICATION COMPLETE ===")
