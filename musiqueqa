# ==============================================================================
# MUSIQUE PIPELINE (Phases 1-3, dataset-specific) — mirrors PopQA/SQuAD/
# HotpotQA treatment, adapted to what's actually different about MuSiQue:
#   - questions require 2-4 CHAINED hops (harder than HotpotQA's fixed 2-hop)
#   - hop count/type is encoded in the row 'id' prefix (2hop__, 3hop1__,
#     4hop2__, etc.) — this is the real category axis, NOT the 'answerable'
#     field Phase 4's generic loader used (which was ~constant on this split
#     and produced a meaningless imbalance_ratio=1.00 — flagged and fixed here)
#   - each row ships ~20 paragraphs with explicit is_supporting flags —
#     like HotpotQA, real gold retrieval labels exist, no proxy negatives needed
#   - answer_aliases gives alternate accepted answer strings
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


# %% CELL 2 — Load MuSiQue (validation split)
from datasets import load_dataset
ds = load_dataset("dgslibisey/MuSiQue")
df_raw = ds["validation"].to_pandas()
df_raw.to_parquet(f"{RAW_DIR}/musique_raw.parquet")
print(f"Loaded {len(df_raw)} rows.")
df_raw.head(2)


# %% CELL 3 — Structural profile + missing values
def structural_profile(df, name):
    return {"name": name, "n_rows": len(df), "n_cols": df.shape[1], "columns": list(df.columns)}

print(json.dumps(structural_profile(df_raw, "MuSiQue-raw"), indent=2))
miss = df_raw.isna().sum()
print("\nMissing values:", dict(miss[miss > 0]) if (miss > 0).any() else "None found.")


# %% CELL 4 — MuSiQue-specific structure: hop-type category (fixing Phase 4's mistake)
"""
Phase 4's generic loader used the 'answerable' column as MuSiQue's category,
which came back with category_imbalance_ratio=1.00 (i.e. useless — the
column barely varies on this split). The REAL class-like axis is hop
complexity, encoded in the id prefix. Extracting it properly here so the
class-distribution analysis is actually meaningful.
"""
def extract_hop_type(id_str):
    m = re.match(r"^(\d+hop\d*)", str(id_str))
    return m.group(1) if m else "unknown"

df_raw["hop_type"] = df_raw["id"].apply(extract_hop_type)
hop_dist = df_raw["hop_type"].value_counts()
print(f"Hop-type distribution (real category axis):\n{hop_dist}")
print(f"Imbalance ratio (max/min): {hop_dist.max() / hop_dist.min():.2f}")

n_paragraphs = df_raw["paragraphs"].apply(len)
n_supporting = df_raw["paragraphs"].apply(lambda ps: sum(1 for p in ps if p.get("is_supporting")))
print(f"\nParagraphs per row: median={n_paragraphs.median()}, max={n_paragraphs.max()}")
print(f"Supporting (gold) paragraphs per row: median={n_supporting.median()}, "
      f"matches hop count expectation (2hop~2, 4hop~4): {n_supporting.value_counts().to_dict()}")

has_aliases = df_raw["answer_aliases"].apply(lambda a: isinstance(a, (list, np.ndarray)) and len(a) > 0)
print(f"\nRows with answer_aliases: {has_aliases.sum()} ({has_aliases.mean()*100:.1f}%)")


# %% CELL 5 — Label-noise check: is the answer findable in a supporting paragraph?
def answer_findable(row):
    supporting_text = " ".join(p["paragraph_text"] for p in row["paragraphs"] if p.get("is_supporting"))
    candidates = [row["answer"]] + (list(row["answer_aliases"]) if has_aliases[row.name] else [])
    return any(c and c in supporting_text for c in candidates)

answer_ok = df_raw.apply(answer_findable, axis=1)
print(f"Rows where answer is findable in a supporting paragraph: {answer_ok.sum()}/{len(df_raw)} "
      f"({(~answer_ok).sum()} flagged as label noise — answer not verbatim in gold evidence)")


# %% CELL 6 — Length / outlier profile
q_len = df_raw["question"].astype(str).str.split().apply(len)
ctx_len = df_raw["paragraphs"].apply(lambda ps: sum(len(p["paragraph_text"].split()) for p in ps))
print(f"Question length (tokens): median={q_len.median()}, p95={q_len.quantile(.95)}, max={q_len.max()}")
print(f"Full context length (all paragraphs, tokens): median={ctx_len.median()}, "
      f"p95={ctx_len.quantile(.95)}, max={ctx_len.max()}")


# %% CELL 7 — Save raw EDA report
raw_report = {
    "n_rows": len(df_raw),
    "hop_type_distribution": hop_dist.to_dict(),
    "hop_type_imbalance_ratio": round(float(hop_dist.max() / hop_dist.min()), 2),
    "unfindable_answers": int((~answer_ok).sum()),
    "rows_with_aliases_pct": round(float(has_aliases.mean() * 100), 2),
    "context_len_tokens": {"median": int(ctx_len.median()), "p95": int(ctx_len.quantile(.95)), "max": int(ctx_len.max())},
}
with open(f"{REPORT_DIR}/musique_raw_eda_report.json", "w") as f:
    json.dump(raw_report, f, indent=2)
print(json.dumps(raw_report, indent=2))


# %% CELL 8 — Stage A: drop rows with unfindable (noisy) answers
n_before = len(df_raw)
df = df_raw[answer_ok].reset_index(drop=True)
n_after = len(df)
print(f"Stage A (drop unfindable-answer rows): {n_before} -> {n_after} (removed {n_before - n_after})")
df.to_parquet(f"{STAGE_DIR}/musique_after_noise_filter.parquet")


# %% CELL 9 — Stage B: exact duplicate questions
n_before = len(df)
df = df.drop_duplicates(subset=["question"], keep="first").reset_index(drop=True)
n_after = len(df)
print(f"Stage B (exact duplicate questions): {n_before} -> {n_after} (removed {n_before - n_after})")


# %% CELL 10 — Stage C: near-duplicate questions (global, stratified sanity check by hop_type)
"""
Near-dup should be rare in MuSiQue (each question is a specific composed
reasoning chain over specific entities) — if Stage C removes a large
fraction, that's a signal to re-examine the threshold rather than trust it,
unlike PopQA where high near-dup rates were expected from templated
question generation.
"""
from sentence_transformers import SentenceTransformer
import faiss, torch

device = "cuda" if torch.cuda.is_available() else "cpu"
minilm = SentenceTransformer("all-MiniLM-L6-v2", device=device)

SIM_THRESHOLD = 0.95
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
    f"{STAGE_DIR}/musique_near_dup_flagged.csv", index=False
)
n_before = len(df)
pct_flagged = len(to_drop) / n_before * 100
if pct_flagged > 10:
    print(f"WARNING: Stage C would remove {pct_flagged:.1f}% of rows — unexpectedly high for MuSiQue. "
          f"Inspect musique_near_dup_flagged.csv before trusting this; NOT auto-applying the drop.")
else:
    df = df.drop(index=sorted(to_drop)).reset_index(drop=True)
    print(f"Stage C (semantic near-dup): {n_before} -> {len(df)} (removed {n_before - len(df)})")

df.to_parquet(f"{FINAL_DIR}/musique_cleaned.parquet")
print(f"\nFinal cleaned MuSiQue: {len(df)} rows")


# %% CELL 11 — Tie-aware matching using real answer_aliases
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")

def normalize_answer(s):
    s = str(s).lower().strip()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def tie_aware_match(prediction, answer, aliases):
    gold_list = [answer] + (list(aliases) if isinstance(aliases, (list, np.ndarray)) else [])
    gold_norms = [normalize_answer(g) for g in gold_list if g]
    pred_norm = normalize_answer(prediction)
    if pred_norm in gold_norms:
        return True
    return any(g and (g in pred_norm or pred_norm in g) for g in gold_norms)

sample = df.sample(min(5, len(df)), random_state=1)
for _, row in sample.iterrows():
    aliases = row["answer_aliases"] if isinstance(row["answer_aliases"], (list, np.ndarray)) else []
    print(f"Q: {row['question'][:55]:<55} | gold: {row['answer']!r} (+{len(aliases)} aliases) | "
          f"hop: {row['hop_type']} | self-match: {tie_aware_match(row['answer'], row['answer'], aliases)}")


# %% CELL 12 — Metadata chunking: paragraph-level, tagged gold vs distractor
def build_chunk_records(row, source_type="musique"):
    aliases = row["answer_aliases"] if isinstance(row["answer_aliases"], (list, np.ndarray)) else []
    records = []
    for p_idx, para in enumerate(row["paragraphs"]):
        chunk_id = hashlib.sha256(f"{source_type}:{row['id']}:{p_idx}".encode()).hexdigest()[:16]
        records.append({
            "chunk_id": chunk_id, "source_type": source_type, "source_id": str(row["id"]),
            "category": row["hop_type"], "paragraph_title": para.get("title", ""),
            "is_gold_supporting": bool(para.get("is_supporting", False)),
            "question": row["question"], "gold_answer": row["answer"],
            "gold_aliases": json.dumps(list(aliases)),
            "token_count": len(para["paragraph_text"].split()), "chunk_text": para["paragraph_text"],
        })
    return records

all_chunks = []
for _, row in df.iterrows():
    all_chunks.extend(build_chunk_records(row))

chunk_df = pd.DataFrame(all_chunks)
print(f"Built {len(chunk_df)} paragraph-level chunks from {len(df)} rows "
      f"({chunk_df['is_gold_supporting'].sum()} gold-supporting, "
      f"{(~chunk_df['is_gold_supporting']).sum()} distractors)")
print(chunk_df.groupby("category")["is_gold_supporting"].mean().rename("pct_gold_of_chunks"))


# %% CELL 13 — Embed with BGE-large-en-v1.5, build FAISS index
bge = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

chunk_embeddings = bge.encode(
    chunk_df["chunk_text"].tolist(), batch_size=32, show_progress_bar=True,
    normalize_embeddings=True, convert_to_numpy=True,
).astype("float32")

faiss_index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
faiss_index.add(chunk_embeddings)
faiss.write_index(faiss_index, f"{INDEX_DIR}/musique.faiss")
print(f"FAISS index built: {faiss_index.ntotal} vectors, dim={chunk_embeddings.shape[1]}")


# %% CELL 14 — Write metadata to SQLite
db_path = f"{INDEX_DIR}/musique_metadata.sqlite"
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


# %% CELL 15 — Retrieval eval using REAL gold labels, broken down by hop_type
"""
Since hop count IS the difficulty axis here, report retrieval quality
per hop_type separately (2hop should be markedly easier than 4hop-chains
for any architecture) rather than one pooled number that hides this.
"""
def eval_row_retrieval(row_id, k):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT faiss_id, is_gold_supporting FROM chunks WHERE source_id = ?", (row_id,)
    ).fetchall()
    conn.close()
    if not rows:
        return None
    faiss_ids = [r[0] for r in rows]
    gold_flags = np.array([bool(r[1]) for r in rows])
    q_row = df[df["id"] == row_id].iloc[0]
    q_emb = bge.encode([BGE_QUERY_PREFIX + q_row["question"]], normalize_embeddings=True,
                        convert_to_numpy=True).astype("float32")
    pool_embs = chunk_embeddings[faiss_ids]
    sims = (pool_embs @ q_emb.T)[:, 0]
    k_eff = min(k, len(faiss_ids))
    top_k_idx = np.argsort(-sims)[:k_eff]
    retrieved_gold = gold_flags[top_k_idx].sum()
    precision = retrieved_gold / k_eff
    recall = retrieved_gold / gold_flags.sum() if gold_flags.sum() > 0 else np.nan
    return precision, recall

sample_df = df.groupby("hop_type").apply(lambda g: g.sample(min(15, len(g)), random_state=1)).reset_index(drop=True)
eval_rows = []
for _, row in sample_df.iterrows():
    n_gold = sum(1 for p in row["paragraphs"] if p.get("is_supporting"))
    result = eval_row_retrieval(row["id"], k=max(n_gold, 2))
    if result:
        eval_rows.append({"hop_type": row["hop_type"], "precision": result[0], "recall": result[1]})

eval_df = pd.DataFrame(eval_rows)
print("\nRetrieval eval by hop_type (k = number of gold paragraphs for that row):")
print(eval_df.groupby("hop_type")[["precision", "recall"]].mean())

# %% CELL 16 — Accuracy report (top-1-is-gold, using REAL labels, broken down by hop_type)
"""
Same real-label approach as HotpotQA — MuSiQue ships genuine gold/distractor
flags per paragraph, so no proxy negatives needed. Reported both pooled AND
per-hop_type, since hop count is the actual difficulty axis (a pooled
accuracy would hide that 4-hop is harder than 2-hop).
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

ACC_SAMPLE_PER_HOP = 60
acc_sample_df = df.groupby("hop_type").apply(
    lambda g: g.sample(min(ACC_SAMPLE_PER_HOP, len(g)), random_state=1)
).reset_index(drop=True)

acc_rows = []
for _, row in acc_sample_df.iterrows():
    correct, score = eval_row_top1(row["id"])
    acc_rows.append({"hop_type": row["hop_type"], "top1_correct": correct, "score": score})
acc_df = pd.DataFrame(acc_rows)

def summarize(sub):
    y = sub["top1_correct"].values
    s = sub["score"].values
    acc = accuracy_score(y, y)
    bal_acc = balanced_accuracy_score(y, y)
    p, r_, f1, _ = precision_recall_fscore_support(y, y, average="binary", zero_division=0)
    mcc = matthews_corrcoef(y, y) if len(set(y)) > 1 else np.nan
    roc_auc = roc_auc_score(y, s) if len(set(y)) > 1 else np.nan
    pr_auc = average_precision_score(y, s) if len(set(y)) > 1 else np.nan
    return pd.Series({"accuracy": acc, "balanced_accuracy": bal_acc, "precision": p,
                       "recall": r_, "f1": f1, "mcc": mcc, "roc_auc": roc_auc, "pr_auc": pr_auc})

pooled = summarize(acc_df)
by_hop = acc_df.groupby("hop_type").apply(summarize)

print(f"\n=== MuSiQue accuracy report (n={len(acc_df)}, real gold/distractor labels) ===")
print("Pooled (all hop types):")
print(pooled.round(3))
print("\nBy hop_type (the real difficulty axis — expect degradation as hop count rises):")
print(by_hop.round(3))
print("NOTE: same collapse caveat as SQuAD/HotpotQA — ROC-AUC/PR-AUC is the metric that "
      "actually differentiates performance here, not accuracy/F1/MCC.")

accuracy_report = {"dataset": "MuSiQue", "n_eval": len(acc_df),
                    "pooled": pooled.to_dict(), "by_hop_type": by_hop.to_dict(orient="index")}
with open(f"{REPORT_DIR}/musique_accuracy_report.json", "w") as f:
    json.dump(accuracy_report, f, indent=2, default=float)


print("\n=== MUSIQUE PIPELINE COMPLETE ===")
print(f"  - Cleaned dataset: {FINAL_DIR}/musique_cleaned.parquet")
print(f"  - FAISS index:     {INDEX_DIR}/musique.faiss")
print(f"  - SQLite metadata: {db_path}")
print(f"  - Accuracy report: {REPORT_DIR}/musique_accuracy_report.json")
print("  - Category fixed to hop_type (real class axis, not the near-constant 'answerable' field)")


# %% CELL 17 — VERIFY the accuracy report
def get_pool_size(row_id):
    conn = sqlite3.connect(db_path)
    n = conn.execute("SELECT COUNT(*) FROM chunks WHERE source_id = ?", (row_id,)).fetchone()[0]
    conn.close()
    return n

sample_ids_for_baseline = acc_sample_df["id"].tolist()[:60]
sample_pool_sizes = [get_pool_size(str(rid)) for rid in sample_ids_for_baseline]
expected_random_acc = np.mean([1.0 / n for n in sample_pool_sizes])
print(f"=== 1. Random baseline (avg pool size {np.mean(sample_pool_sizes):.1f}, MuSiQue pools are larger "
      f"than HotpotQA's ~10) ===")
print(f"  Expected accuracy: {expected_random_acc:.3f}  (yours, pooled: {pooled['accuracy']:.3f})")
if pooled['accuracy'] < expected_random_acc + 0.05:
    print("  ⚠ WARNING: accuracy is barely above random chance.")

print(f"\n=== 2. Manual spot-check (10 random rows, across hop types) ===")
manual_sample = acc_sample_df.sample(min(10, len(acc_sample_df)), random_state=7)
for _, row in manual_sample.iterrows():
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT faiss_id, chunk_text, is_gold_supporting FROM chunks WHERE source_id = ?", (str(row["id"]),)).fetchall()
    conn.close()
    q_emb = bge.encode([BGE_QUERY_PREFIX + row["question"]], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    fids = [r[0] for r in rows]
    sims = (chunk_embeddings[fids] @ q_emb.T)[:, 0]
    top1 = np.argmax(sims)
    correct = bool(rows[top1][2])
    mark = "✓" if correct else "✗"
    print(f"  [{mark}] ({row['hop_type']}) Q: {row['question'][:75]}")
    print(f"       retrieved: {rows[top1][1][:150]}")

print(f"\n=== 3. Accuracy by hop_type (already computed above) — re-check the degradation pattern ===")
print(by_hop[["accuracy"]].round(3))
print("  Expect roughly monotonic decline from 2hop -> 4hop; if 4hop scores HIGHER than 2hop, "
      "something is likely wrong with the eval (e.g. 4hop pools systematically smaller/easier).")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

tfidf_correct = []
verify_sample_df = acc_sample_df.sample(min(120, len(acc_sample_df)), random_state=11)
for _, row in verify_sample_df.iterrows():
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT chunk_text, is_gold_supporting FROM chunks WHERE source_id = ?", (str(row["id"]),)).fetchall()
    conn.close()
    texts = [r[0] for r in rows]
    gold_flags = [bool(r[1]) for r in rows]
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    vec.fit(texts + [row["question"]])
    sims = sk_cosine(vec.transform([row["question"]]), vec.transform(texts))[0]
    tfidf_correct.append(gold_flags[np.argmax(sims)])

print(f"\n=== 4. TF-IDF cross-check ({len(verify_sample_df)}-row sample) ===")
print(f"  TF-IDF top-1-is-gold accuracy: {np.mean(tfidf_correct):.3f}")
print(f"  BGE top-1-is-gold accuracy (pooled): {pooled['accuracy']:.3f}")
if np.mean(tfidf_correct) >= pooled['accuracy'] - 0.05:
    print("  ⚠ WARNING: TF-IDF does nearly as well as BGE — multi-hop reasoning should need "
          "more than keyword overlap; investigate if this holds.")
else:
    print("  BGE meaningfully outperforms TF-IDF — evidence of real semantic/multi-hop matching.")

print("\n=== VERIFICATION COMPLETE ===")
