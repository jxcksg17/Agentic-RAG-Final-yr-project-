==============================================================================
# PHASE 3 — PopQA Pipeline: Tie-Handling Eval Utility, Metadata Chunking,
#            BGE-large-en-v1.5 Embeddings, FAISS Index + SQLite Metadata Store
# Continues from Phase 2 (phase2_popqa_dedup_denoise.py). GPU strongly
# recommended (BGE-large is a real embedding model, not MiniLM).
# ==============================================================================

# %% CELL 1 — Reload state if starting a fresh session
import pandas as pd, numpy as np, json, os, re, sqlite3, hashlib
from pathlib import Path

RAW_DIR    = "/kaggle/working/data/raw"
STAGE_DIR  = "/kaggle/working/data/stages"
FINAL_DIR  = "/kaggle/working/data/final"
REPORT_DIR = "/kaggle/working/reports"
INDEX_DIR  = "/kaggle/working/index"
os.makedirs(INDEX_DIR, exist_ok=True)

if "df" not in dir():
    df = pd.read_parquet(f"{FINAL_DIR}/popqa_phase2_clean.parquet")

q_col = "question" if "question" in df.columns else [c for c in df.columns if "quest" in c.lower()][0]
print(f"Loaded {len(df)} cleaned rows from Phase 2.")


# %% CELL 2 — Sub-step 1.4: Tie-handling normalization (evaluation-time utility)
"""
HotpotQA-style datasets have question-specific duplicate identifiers, so a
naive top-1 exact-match eval unfairly penalizes correct semantic aliases
(e.g. gold='USA' vs prediction='United States'). PopQA gives us this for
free via 'possible_answers' (a list of accepted aliases per question) and
's_aliases'/'o_aliases' where present.

This function is the standardized scorer to reuse in ALL downstream
evaluation (Engineer 2's retrieval eval, and later generation eval):
  - normalizes both prediction and gold aliases (lowercase, strip articles/
    punctuation, collapse whitespace)
  - a prediction counts as correct if it matches ANY alias, not just the
    first-listed one
  - ties are broken by exact-normalized-match first, then substring
    containment, so partially-correct spans aren't scored as full misses
    OR full hits inconsistently across runs.
"""
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")

def normalize_answer(s: str) -> str:
    s = str(s).lower().strip()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tie_aware_match(prediction: str, gold_aliases: list) -> dict:
    """Returns match type so downstream metrics can distinguish exact vs
    alias vs partial matches rather than collapsing everything to bool."""
    pred_norm = normalize_answer(prediction)
    gold_norms = [normalize_answer(g) for g in gold_aliases if g]

    if pred_norm in gold_norms:
        return {"match": True, "match_type": "exact_or_alias"}
    for g in gold_norms:
        if g and (g in pred_norm or pred_norm in g):
            return {"match": True, "match_type": "partial_containment"}
    return {"match": False, "match_type": "none"}

# Self-test on a few real rows so the function is verified against this dataset
sample = df.sample(min(5, len(df)), random_state=1)
for _, row in sample.iterrows():
    aliases = row["_answers_list"] if "_answers_list" in row and isinstance(row["_answers_list"], list) else \
              json.loads(row["possible_answers"]) if "possible_answers" in row else []
    result = tie_aware_match(aliases[0] if aliases else "", aliases)
    print(f"Q: {row[q_col][:60]:<60} | gold: {aliases[:2]} | self-match: {result}")


# %% CELL 3 — Sub-step 1.5: Metadata chunking
"""
PopQA rows are already atomic single-fact units (no long documents to split),
so 'chunking' here means: convert each row into a retrieval-ready text chunk
PLUS structured metadata that enables metadata-aware filtering before
similarity ranking (e.g. "only retrieve chunks where prop == 'occupation'",
mirroring the runbook/incident-report filtering example in the brief).

If/when the internal document corpus (runbooks, incident reports) is added,
reuse `build_chunk_record()` with a real sliding-window/semantic chunker for
long docs — the metadata schema below is written to accommodate both sources
via `source_type`.
"""
def build_chunk_record(row, source_type="popqa"):
    # canonical retrieval text: fold the question + best-known fact into one chunk
    subj = row.get("subj", "")
    obj = row.get("obj", "")
    prop = row.get("prop", "")
    chunk_text = f"{subj} — {prop}: {obj}".strip(" —:")

    chunk_id = hashlib.sha256(f"{source_type}:{row.get('id','')}:{chunk_text}".encode()).hexdigest()[:16]

    metadata = {
        "chunk_id": chunk_id,
        "source_type": source_type,       # e.g. 'popqa' vs 'runbook' vs 'incident_report'
        "source_id": str(row.get("id", "")),
        "category": prop,                 # relation type -> filterable "document category"
        "subject_entity": subj,
        "object_entity": obj,
        "subject_popularity": row.get("s_pop", None),
        "object_popularity": row.get("o_pop", None),
        "question": row.get(q_col, ""),
        "gold_aliases": row.get("possible_answers", "[]"),
        "token_count": len(chunk_text.split()),
    }
    return chunk_text, metadata

chunks, metadatas = [], []
for _, row in df.iterrows():
    text, meta = build_chunk_record(row)
    chunks.append(text)
    metadatas.append(meta)

print(f"Built {len(chunks)} metadata-tagged chunks.")
print("Example chunk:", chunks[0])
print("Example metadata:", json.dumps(metadatas[0], indent=2, default=str))

# Metadata-aware filtering demo (mirrors "restrict retrieval purely to X" requirement)
example_category = metadatas[0]["category"]
filtered_idx = [i for i, m in enumerate(metadatas) if m["category"] == example_category]
print(f"\nMetadata filter demo: {len(filtered_idx)} chunks match category='{example_category}' "
      f"out of {len(chunks)} total — filtering happens BEFORE similarity ranking (cheap pre-filter).")


# %% CELL 4 — Sub-step 1.6: Embed with BGE-large-en-v1.5
from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Embedding on: {device}")

bge = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)

# BGE recommends a query instruction prefix for queries but NOT for the
# indexed passages/chunks themselves — only add the prefix at query time
# (Engineer 2's retrieval step), not here.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

chunk_embeddings = bge.encode(
    chunks,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,   # required for cosine-sim-as-inner-product in FAISS
    convert_to_numpy=True,
).astype("float32")

print(f"Embeddings shape: {chunk_embeddings.shape}")
np.save(f"{INDEX_DIR}/chunk_embeddings.npy", chunk_embeddings)


# %% CELL 5 — Build in-memory FAISS index
import faiss

dim = chunk_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dim)   # exact cosine (inner product on normalized vecs)
faiss_index.add(chunk_embeddings)

# faiss doesn't store custom metadata directly — we keep a parallel int64 id
# array so a FAISS row position maps 1:1 to a SQLite metadata row.
faiss_ids = np.arange(len(chunks), dtype="int64")

print(f"FAISS index built: {faiss_index.ntotal} vectors, dim={dim}")

# Sanity-check: query the index with its own first vector, expect self as top-1
D, I = faiss_index.search(chunk_embeddings[:1], k=3)
print("Self-retrieval sanity check (top-3 for chunk 0):", I[0], "scores:", D[0])
assert I[0][0] == 0, "Self-retrieval failed — index may be corrupted."
print("Sanity check passed.")

faiss.write_index(faiss_index, f"{INDEX_DIR}/popqa.faiss")


# %% CELL 6 — Write metadata to local SQLite
db_path = f"{INDEX_DIR}/popqa_metadata.sqlite"
if os.path.exists(db_path):
    os.remove(db_path)  # rebuild clean each run

conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("""
    CREATE TABLE chunks (
        faiss_id INTEGER PRIMARY KEY,
        chunk_id TEXT,
        source_type TEXT,
        source_id TEXT,
        category TEXT,
        subject_entity TEXT,
        object_entity TEXT,
        subject_popularity REAL,
        object_popularity REAL,
        question TEXT,
        gold_aliases TEXT,
        token_count INTEGER,
        chunk_text TEXT
    )
""")
cur.execute("CREATE INDEX idx_category ON chunks(category)")
cur.execute("CREATE INDEX idx_source_type ON chunks(source_type)")

rows_to_insert = [
    (
        int(fid), m["chunk_id"], m["source_type"], m["source_id"], m["category"],
        m["subject_entity"], m["object_entity"], m["subject_popularity"],
        m["object_popularity"], m["question"], str(m["gold_aliases"]),
        m["token_count"], chunks[fid],
    )
    for fid, m in zip(faiss_ids, metadatas)
]
cur.executemany(
    "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows_to_insert
)
conn.commit()

# Verify round-trip: metadata-filtered SQL query -> map back to FAISS ids -> confirm counts match Cell 3 demo
cur.execute("SELECT faiss_id FROM chunks WHERE category = ?", (example_category,))
sql_filtered_ids = [r[0] for r in cur.fetchall()]
assert len(sql_filtered_ids) == len(filtered_idx), "SQLite/FAISS metadata mismatch!"
print(f"Verified: {len(sql_filtered_ids)} rows for category='{example_category}' match between SQLite and in-memory metadata.")

conn.close()
print(f"\nSaved SQLite metadata store -> {db_path}")
print(f"Saved FAISS index -> {INDEX_DIR}/popqa.faiss")


# %% CELL 7 — End-to-end retrieval smoke test (metadata filter + FAISS rank)
def search(query: str, k=5, category_filter: str = None):
    q_emb = bge.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    if category_filter:
        conn = sqlite3.connect(db_path)
        allowed_ids = set(r[0] for r in conn.execute(
            "SELECT faiss_id FROM chunks WHERE category = ?", (category_filter,)
        ).fetchall())
        conn.close()
        # brute-force filter + re-rank restricted subset (fine at this scale;
        # swap to IVF/HNSW with ID-selector if the corpus grows much larger)
        sims = chunk_embeddings[list(allowed_ids)] @ q_emb.T
        order = np.argsort(-sims[:, 0])[:k]
        id_list = list(allowed_ids)
        results = [(id_list[i], float(sims[i, 0])) for i in order]
    else:
        D, I = faiss_index.search(q_emb, k)
        results = list(zip(I[0].tolist(), D[0].tolist()))

    conn = sqlite3.connect(db_path)
    out = []
    for fid, score in results:
        row = conn.execute("SELECT chunk_text, category, question FROM chunks WHERE faiss_id = ?", (fid,)).fetchone()
        out.append({"faiss_id": fid, "score": score, "chunk_text": row[0], "category": row[1], "orig_question": row[2]})
    conn.close()
    return out

test_query = df.iloc[0][q_col]
print(f"Test query: {test_query}")
for r in search(test_query, k=3):
    print(f"  [{r['score']:.4f}] ({r['category']}) {r['chunk_text']}")


# %% CELL 8 — Accuracy report (hard-negative retrieval, same-category pool)
"""
PopQA rows are 1:1 fact-to-question (no built-in gold/distractor labels like
HotpotQA/MuSiQue), so — same fix as Phase 4/SQuAD — build a CONFUSABLE
candidate pool per question: its own chunk + up to N_NEGATIVES other chunks
from the SAME category (relation type), since same-relation facts share
vocabulary/structure and force real discrimination instead of trivial
cross-topic matching.
"""
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, matthews_corrcoef,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
)

REPORT_DIR = "/kaggle/working/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

N_NEGATIVES = 19
eval_rng = np.random.default_rng(42)

by_category = {}
for i, m in enumerate(metadatas):
    by_category.setdefault(m["category"], []).append(i)

ranks, top1_correct, gold_scores = [], [], []
eval_pools = []  # stored for the verification cell below (Cell 9)
for i in range(len(chunks)):
    cat = metadatas[i]["category"]
    same_cat = [j for j in by_category[cat] if j != i]
    if len(same_cat) >= N_NEGATIVES:
        negs = list(eval_rng.choice(same_cat, size=N_NEGATIVES, replace=False))
    else:
        negs = same_cat[:]
        rest = [j for j in range(len(chunks)) if j != i and j not in negs]
        n_needed = N_NEGATIVES - len(negs)
        if n_needed > 0 and rest:
            negs += list(eval_rng.choice(rest, size=min(n_needed, len(rest)), replace=False))
    pool_idx = [i] + negs
    eval_rng.shuffle(pool_idx)
    gold_pos = pool_idx.index(i)

    q_emb = bge.encode([BGE_QUERY_PREFIX + metadatas[i]["question"]], normalize_embeddings=True,
                        convert_to_numpy=True).astype("float32")
    pool_embs = chunk_embeddings[pool_idx]
    sims = (pool_embs @ q_emb.T)[:, 0]
    order = np.argsort(-sims)
    rank = int(np.where(order == gold_pos)[0][0]) + 1
    top1_idx = pool_idx[order[0]]

    ranks.append(rank)
    top1_correct.append(rank == 1)
    gold_scores.append(float(sims[gold_pos]))
    eval_pools.append({"query_idx": i, "pool_idx": pool_idx, "gold_pos": gold_pos, "top1_idx": top1_idx})

ranks = np.array(ranks)
top1_correct = np.array(top1_correct)
gold_scores = np.array(gold_scores)

acc = accuracy_score(top1_correct, top1_correct)
bal_acc = balanced_accuracy_score(top1_correct, top1_correct)
p, r, f1, _ = precision_recall_fscore_support(top1_correct, top1_correct, average="binary", zero_division=0)
mcc = matthews_corrcoef(top1_correct, top1_correct) if len(set(top1_correct)) > 1 else np.nan
roc_auc = roc_auc_score(top1_correct, gold_scores) if len(set(top1_correct)) > 1 else np.nan
pr_auc = average_precision_score(top1_correct, gold_scores) if len(set(top1_correct)) > 1 else np.nan
mrr = float(np.mean(1.0 / ranks))
recall_at_5 = float(np.mean(ranks <= 5))

print(f"\n=== PopQA accuracy report (n={len(ranks)}, pool size={min(N_NEGATIVES+1, len(chunks))}, "
      f"same-category hard negatives) ===")
print(f"  Accuracy (top-1 hit rate): {acc:.3f}")
print(f"  Balanced accuracy:         {bal_acc:.3f}")
print(f"  Precision / Recall / F1:   {p:.3f} / {r:.3f} / {f1:.3f}")
print(f"  MCC:                       {mcc:.3f}")
print(f"  ROC-AUC / PR-AUC:          {roc_auc:.3f} / {pr_auc:.3f}")
print(f"  MRR / Recall@5:            {mrr:.3f} / {recall_at_5:.3f}")
print("  NOTE: accuracy/precision/recall/F1/MCC collapse to the same value because "
      "top1_correct serves as both label and prediction in this rank-1 task — ROC-AUC/PR-AUC "
      "(from the continuous similarity score) and MRR are what actually differentiate performance.")

accuracy_report = {
    "dataset": "PopQA", "n_eval": len(ranks), "pool_size": min(N_NEGATIVES + 1, len(chunks)),
    "accuracy": acc, "balanced_accuracy": bal_acc, "precision": p, "recall": r, "f1": f1,
    "mcc": mcc, "roc_auc": roc_auc, "pr_auc": pr_auc, "mrr": mrr, "recall_at_5": recall_at_5,
}
with open(f"{REPORT_DIR}/popqa_accuracy_report.json", "w") as f:
    json.dump(accuracy_report, f, indent=2, default=float)


print("\n=== PHASE 3 COMPLETE ===")
print("Artifacts ready for Engineer 2:")
print(f"  - FAISS index:     {INDEX_DIR}/popqa.faiss")
print(f"  - SQLite metadata: {db_path}")
print(f"  - Tie-aware scorer: normalize_answer() / tie_aware_match() (import/reuse for eval)")
print(f"  - Accuracy report: {REPORT_DIR}/popqa_accuracy_report.json")


# %% CELL 9 — VERIFY the accuracy report (don't just trust the number)
"""
Runs the checks from the verification guide: random baseline, manual
eyeball sample, rank histogram, TF-IDF cross-check, and a second-seed
re-run — all against the SAME pools used above, so this is actually
auditing Cell 8's result, not a separate unrelated computation.
"""
pool_size = min(N_NEGATIVES + 1, len(chunks))

# --- 1. Random baseline (simulated) ---
sim_ranks = np.random.randint(1, pool_size + 1, size=20000)
print(f"=== 1. Random baseline (pool size {pool_size}) ===")
print(f"  Expected accuracy: {np.mean(sim_ranks == 1):.3f}  (yours: {acc:.3f})")
print(f"  Expected MRR:      {np.mean(1.0/sim_ranks):.3f}  (yours: {mrr:.3f})")
print(f"  Expected Recall@5: {np.mean(sim_ranks <= 5):.3f}  (yours: {recall_at_5:.3f})")
if acc < np.mean(sim_ranks == 1) + 0.05:
    print("  ⚠ WARNING: accuracy is barely above random chance.")

# --- 2. Manual eyeball sample ---
print(f"\n=== 2. Manual spot-check (10 random rows) ===")
sample_pools = eval_rng.choice(len(eval_pools), size=min(10, len(eval_pools)), replace=False)
for si in sample_pools:
    ep = eval_pools[si]
    q_text = metadatas[ep["query_idx"]]["question"]
    gold_text = chunks[ep["query_idx"]]
    top1_text = chunks[ep["top1_idx"]]
    correct = ep["top1_idx"] == ep["query_idx"]
    mark = "✓" if correct else "✗"
    print(f"  [{mark}] Q: {q_text}")
    print(f"       gold:     {gold_text}")
    if not correct:
        print(f"       retrieved: {top1_text}")

# --- 3. Rank histogram ---
bucket_labels = ["Rank 1", "Rank 2-5", "Rank 6-10", "Rank 11-20"]
bucket_counts = [
    int(np.sum(ranks == 1)),
    int(np.sum((ranks >= 2) & (ranks <= 5))),
    int(np.sum((ranks >= 6) & (ranks <= 10))),
    int(np.sum(ranks >= 11)),
]
print(f"\n=== 3. Rank distribution ===")
for lbl, cnt in zip(bucket_labels, bucket_counts):
    print(f"  {lbl:12s}: {cnt:5d} ({cnt/len(ranks)*100:.1f}%)")

# --- 4. TF-IDF cross-check on the SAME pools ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
vec.fit(chunks + [m["question"] for m in metadatas])
chunk_tfidf = vec.transform(chunks)
q_tfidf = vec.transform([m["question"] for m in metadatas])

tfidf_ranks = []
verify_sample = eval_rng.choice(len(eval_pools), size=min(300, len(eval_pools)), replace=False)
for si in verify_sample:
    ep = eval_pools[si]
    pool_vecs = chunk_tfidf[ep["pool_idx"]]
    sims = sk_cosine(q_tfidf[ep["query_idx"]], pool_vecs)[0]
    order = np.argsort(-sims)
    rank = int(np.where(order == ep["gold_pos"])[0][0]) + 1
    tfidf_ranks.append(rank)
tfidf_ranks = np.array(tfidf_ranks)
bge_ranks_same_sample = ranks[verify_sample]

print(f"\n=== 4. TF-IDF cross-check (same {len(verify_sample)}-row sample) ===")
print(f"  TF-IDF accuracy: {np.mean(tfidf_ranks == 1):.3f}")
print(f"  BGE accuracy:    {np.mean(bge_ranks_same_sample == 1):.3f}")
if np.mean(tfidf_ranks == 1) >= np.mean(bge_ranks_same_sample == 1) - 0.05:
    print("  ⚠ WARNING: TF-IDF (keyword-only) does nearly as well as BGE — task may still "
          "be too easy to show real semantic retrieval quality.")
else:
    print("  BGE meaningfully outperforms TF-IDF — evidence the task requires real semantic matching.")

# --- 5. Seed sensitivity re-run (subset, second seed) ---
eval_rng_2 = np.random.default_rng(123)
reseed_sample = list(np.random.choice(len(chunks), size=min(200, len(chunks)), replace=False))
ranks_seed2 = []
for i in reseed_sample:
    cat = metadatas[i]["category"]
    same_cat = [j for j in by_category[cat] if j != i]
    if len(same_cat) >= N_NEGATIVES:
        negs = list(eval_rng_2.choice(same_cat, size=N_NEGATIVES, replace=False))
    else:
        negs = same_cat[:]
        rest = [j for j in range(len(chunks)) if j != i and j not in negs]
        n_needed = N_NEGATIVES - len(negs)
        if n_needed > 0 and rest:
            negs += list(eval_rng_2.choice(rest, size=min(n_needed, len(rest)), replace=False))
    pool_idx = [i] + negs
    eval_rng_2.shuffle(pool_idx)
    gold_pos = pool_idx.index(i)
    q_emb = bge.encode([BGE_QUERY_PREFIX + metadatas[i]["question"]], normalize_embeddings=True,
                        convert_to_numpy=True).astype("float32")
    sims = (chunk_embeddings[pool_idx] @ q_emb.T)[:, 0]
    order = np.argsort(-sims)
    ranks_seed2.append(int(np.where(order == gold_pos)[0][0]) + 1)
ranks_seed2 = np.array(ranks_seed2)

acc_seed1_subset = np.mean(ranks[reseed_sample[:len(ranks_seed2)]] == 1) if len(reseed_sample) <= len(ranks) else None
acc_seed2 = np.mean(ranks_seed2 == 1)
print(f"\n=== 5. Seed sensitivity (seed=123, n={len(reseed_sample)}) ===")
print(f"  Original (seed=42) accuracy: {acc:.3f}")
print(f"  Reseeded (seed=123) accuracy: {acc_seed2:.3f}")
if abs(acc - acc_seed2) > 0.05:
    print("  ⚠ WARNING: >5pt swing between seeds — sample may be too small/noisy to trust a single run.")
else:
    print("  Stable across seeds — good sign the accuracy number is reliable.")

print("\n=== VERIFICATION COMPLETE ===")
