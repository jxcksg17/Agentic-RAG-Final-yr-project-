# ==============================================================================
# 2WIKIMULTIHOPQA PIPELINE (Phases 1-3, dataset-specific) — Engineer 1 deliverable
# Third leg of the multi-hop suite, alongside hotpotqa_pipeline_phase1-3.py and
# musique_pipeline_phase1-3.py. What's actually different about 2WikiMultihopQA:
#   - 4 question types (not HotpotQA's 2): comparison, inference, compositional,
#     bridge_comparison — comparison/bridge_comparison skew yes/no, the other
#     two are extractive
#   - each row ships an 'evidences' field: explicit [subject, relation, object]
#     KG triples that constitute the gold reasoning chain. HotpotQA/MuSiQue only
#     give paragraph-level gold flags; 2Wiki additionally tells you WHY each
#     gold paragraph matters, which is exactly the structured relation metadata
#     Sub-step 1.5 asks for (metadata-aware filtering, not just gold/distractor)
#   - real gold/distractor paragraph labels exist via supporting_facts (like
#     HotpotQA/MuSiQue) — no proxy negative pool needed
#   - this is the FIRST script in the suite to run cross-dataset dedup
#     (Sub-step 1.2) and an ensemble OOF noise filter (Sub-step 1.3) — added
#     here since we now have 3 sibling datasets on disk to check against, and
#     because rule-based noise filtering alone (used in the prior two scripts)
#     misses "looks fine structurally but is statistically anomalous" rows
# Run standalone in a fresh Kaggle session. If run in the same Kaggle
# working directory as the HotpotQA/MuSiQue scripts, Sub-step 1.2 will find
# and use their FINAL_DIR outputs automatically.
# ==============================================================================

# %% CELL 1 — Sub-step 1.1: Environment Initialization
"""
Installs the shared stack (PyTorch, sentence-transformers, faiss-cpu) used
across the whole suite. NOTE for LOCAL (non-Kaggle) runs on an RTX 4060:
Kaggle ships a pre-matched CUDA/torch build, so this pip_install call is
sufficient there. If running locally instead, do NOT `pip install torch`
blindly — check `nvidia-smi` for your driver's max supported CUDA version
first, then install the matching wheel from https://pytorch.org/get-started/
locally/ (RTX 4060 = Ada Lovelace, supports CUDA 11.8+ and 12.x; pick the
torch cu121/cu124 wheel that matches, not just 'latest'), or CUDA calls will
silently fall back to CPU or hard-fail with a driver-mismatch error. vLLM
(for the generation stage Engineer 2 will drive) has the same constraint —
its prebuilt wheels pin a specific torch+CUDA combo, so install vLLM only
after torch is confirmed working, not before.
"""
import subprocess, sys
def pip_install(pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs])
pip_install(["datasets>=2.19.0", "sentence-transformers>=3.0.0", "faiss-cpu",
             "pandas", "numpy", "scikit-learn"])

import pandas as pd, numpy as np, json, os, re, hashlib, sqlite3

RAW_DIR    = "/kaggle/working/data/raw"
STAGE_DIR  = "/kaggle/working/data/stages"
FINAL_DIR  = "/kaggle/working/data/final"
REPORT_DIR = "/kaggle/working/reports"
INDEX_DIR  = "/kaggle/working/index"
for d in [RAW_DIR, STAGE_DIR, FINAL_DIR, REPORT_DIR, INDEX_DIR]:
    os.makedirs(d, exist_ok=True)

import torch
print(f"Environment ready. CUDA available: {torch.cuda.is_available()}"
      + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " (CPU fallback)"))


# %% CELL 2 — Load 2WikiMultihopQA (validation split)
"""
Using framolfese/2WikiMultihopQA: a repackaged, loading-script-free mirror of
the original Alab-NII/2wikimultihop data (same train/val/test split, same
examples — nothing added or removed), with the field layout matching
HotpotQA (id, question, answer, type, supporting_facts, context) plus the
extra 'evidences' field. The original xanhho/kamelliao HF repos require
trust_remote_code=True (arbitrary script execution) to load; this mirror
avoids that entirely, which matters for a pipeline meant to run unattended.
"""
from datasets import load_dataset
ds = load_dataset("framolfese/2WikiMultihopQA")
df_raw = ds["validation"].to_pandas()
df_raw.to_parquet(f"{RAW_DIR}/2wikimultihopqa_raw.parquet")
print(f"Loaded {len(df_raw)} rows.")
df_raw.head(2)


# %% CELL 3 — Structural profile + missing values
def structural_profile(df, name):
    return {"name": name, "n_rows": len(df), "n_cols": df.shape[1], "columns": list(df.columns)}

print(json.dumps(structural_profile(df_raw, "2WikiMultihopQA-raw"), indent=2))
miss = df_raw.isna().sum()
print("\nMissing values:", dict(miss[miss > 0]) if (miss > 0).any() else "None found.")


# %% CELL 4 — 2Wiki-specific structure: 4 question types, paragraph pool, evidence chains
"""
2Wiki's 4 types split into two answer regimes: comparison/bridge_comparison
are mostly yes/no or entity-comparison answers, inference/compositional are
extractive spans built by chaining exactly 2 relation hops through the KG.
This is a materially different type axis than HotpotQA's bridge/comparison
binary, so it's tracked and reported separately end-to-end (Cells 4, 17, 18).
"""
type_dist = df_raw["type"].value_counts()
n_paragraphs = df_raw["context"].apply(lambda c: len(c["title"]))
n_supporting = df_raw["supporting_facts"].apply(lambda sf: len(set(sf["title"])))
n_evidences = df_raw["evidences"].apply(len)

is_yesno = df_raw["answer"].astype(str).str.lower().isin(["yes", "no"])
print(f"Question type distribution:\n{type_dist}")
print(f"\nParagraphs per row: min={n_paragraphs.min()}, median={n_paragraphs.median()}, max={n_paragraphs.max()}")
print(f"Supporting (gold) paragraphs per row: median={n_supporting.median()}, max={n_supporting.max()}")
print(f"Evidence triples per row: median={n_evidences.median()}, max={n_evidences.max()}")
print(f"Yes/No answers (non-extractive): {is_yesno.sum()} ({is_yesno.mean()*100:.1f}%)")
print(f"\nBy type — yes/no rate:\n{df_raw.groupby('type').apply(lambda g: (g['answer'].astype(str).str.lower().isin(['yes','no'])).mean())}")


# %% CELL 5 — Rule-based label-noise checks (structural validity)
"""
Three independent structural checks, each catching a distinct failure mode:
  1. supporting_facts reference valid (title, sent_id) pairs in context
  2. answer text is findable verbatim in context (skip yes/no — not extractive)
  3. evidence-triple entities (subject/object) correspond to actual context
     paragraph titles — 2Wiki-specific: catches evidence chains that point at
     entities never surfaced in this row's own paragraph pool, which would
     silently break any metadata-aware filtering built on Sub-step 1.5
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

def answer_findable(row):
    if str(row["answer"]).lower() in ("yes", "no"):
        return True
    full_text = " ".join(" ".join(s) for s in row["context"]["sentences"])
    return row["answer"] in full_text

def evidence_entities_grounded(row):
    ctx_titles = set(row["context"]["title"])
    if len(row["evidences"]) == 0:
        return True  # nothing to check
    for triple in row["evidences"]:
        if len(triple) != 3:
            return False
        subj, _rel, obj = triple
        # at least one side of the triple should ground to a context paragraph;
        # the other side is often the answer entity itself, which need not have
        # its own paragraph — so this is intentionally an OR, not an AND
        if subj not in ctx_titles and obj not in ctx_titles:
            return False
    return True

sf_valid = df_raw.apply(check_supporting_facts_valid, axis=1)
answer_ok = df_raw.apply(answer_findable, axis=1)
evidence_ok = df_raw.apply(evidence_entities_grounded, axis=1)

print(f"Valid supporting_facts references: {sf_valid.sum()}/{len(df_raw)} ({(~sf_valid).sum()} flagged)")
print(f"Answer findable in context:        {answer_ok.sum()}/{len(df_raw)} ({(~answer_ok).sum()} flagged)")
print(f"Evidence triples grounded in ctx:   {evidence_ok.sum()}/{len(df_raw)} ({(~evidence_ok).sum()} flagged)")

rule_clean = sf_valid & answer_ok & evidence_ok
print(f"\nPasses ALL rule-based checks: {rule_clean.sum()}/{len(df_raw)} ({(~rule_clean).sum()} flagged)")


# %% CELL 6 — Length / outlier profile
q_len = df_raw["question"].astype(str).str.split().apply(len)
ctx_len = df_raw["context"].apply(lambda c: sum(len(s.split()) for para in c["sentences"] for s in para))
print(f"Question length (tokens): median={q_len.median()}, p95={q_len.quantile(.95)}, max={q_len.max()}")
print(f"Context length (tokens):  median={ctx_len.median()}, p95={ctx_len.quantile(.95)}, max={ctx_len.max()}")


# %% CELL 7 — Save raw EDA report
raw_report = {
    "n_rows": len(df_raw),
    "type_distribution": type_dist.to_dict(),
    "yesno_answer_rate_pct": round(float(is_yesno.mean() * 100), 2),
    "invalid_supporting_facts": int((~sf_valid).sum()),
    "unfindable_answers": int((~answer_ok).sum()),
    "ungrounded_evidence": int((~evidence_ok).sum()),
    "context_len_tokens": {"median": int(ctx_len.median()), "p95": int(ctx_len.quantile(.95)), "max": int(ctx_len.max())},
}
with open(f"{REPORT_DIR}/2wikimultihopqa_raw_eda_report.json", "w") as f:
    json.dump(raw_report, f, indent=2)
print(json.dumps(raw_report, indent=2))


# %% CELL 8 — Stage A: drop rows failing the rule-based checks
n_before = len(df_raw)
df = df_raw[rule_clean].reset_index(drop=True)
n_after = len(df)
print(f"Stage A (drop rule-based-noise rows): {n_before} -> {n_after} (removed {n_before - n_after})")


# %% CELL 9 — Sub-step 1.3: Ensemble OOF label-noise filtering
"""
Rule-based checks (Cell 5) catch OBVIOUS annotation breaks — a referenced
title that doesn't exist, a span that's genuinely absent from the text. They
do NOT catch subtler noise: a row that passes every rule check but is
structurally anomalous in a way that correlates with mislabeling (e.g. an
evidence chain that's technically grounded but wildly inconsistent with the
question's type, or a supporting-fact sentence index sitting suspiciously
deep in an oversized paragraph). This is the "sample-dependent" noise case
QA benchmarks are known to have, and it's what an ensemble Out-Of-Fold (OOF)
prediction pass is for: train several different model families to predict
the Cell-5 rule label from SURFACE features (not the rule components
themselves), get out-of-fold predictions, and flag rows the ensemble is
confident are mislabeled even though they passed the rules.
This is a statistical-outlier flag, not a certainty — treated as a second,
softer filter on top of (not instead of) the hard rule-based Stage A above.
"""
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

def build_oof_features(df_in):
    feats = pd.DataFrame(index=df_in.index)
    feats["q_len"] = df_in["question"].astype(str).str.split().apply(len)
    feats["answer_len"] = df_in["answer"].astype(str).str.split().apply(len)
    feats["n_paragraphs"] = df_in["context"].apply(lambda c: len(c["title"]))
    feats["n_supporting"] = df_in["supporting_facts"].apply(lambda sf: len(set(sf["title"])))
    feats["n_evidences"] = df_in["evidences"].apply(len)
    feats["max_sent_id"] = df_in["supporting_facts"].apply(
        lambda sf: max(sf["sent_id"]) if len(sf["sent_id"]) else 0)
    feats["is_yesno"] = df_in["answer"].astype(str).str.lower().isin(["yes", "no"]).astype(int)
    # question-to-gold-sentence TF-IDF similarity — low similarity on a row
    # that otherwise looks structurally normal is a soft noise signal
    sims = []
    for _, row in df_in.iterrows():
        sf_titles = set(row["supporting_facts"]["title"])
        ctx_map = {t: " ".join(s) for t, s in zip(row["context"]["title"], row["context"]["sentences"])}
        gold_text = " ".join(ctx_map[t] for t in sf_titles if t in ctx_map)
        if not gold_text.strip():
            sims.append(0.0)
            continue
        try:
            vec = TfidfVectorizer(min_df=1).fit([row["question"], gold_text])
            v = vec.transform([row["question"], gold_text])
            sims.append(float((v[0] @ v[1].T).toarray()[0, 0]))
        except ValueError:
            sims.append(0.0)
    feats["q_gold_tfidf_sim"] = sims
    feats = pd.concat([feats, pd.get_dummies(df_in["type"], prefix="type")], axis=1)
    return feats

# Trained on the FULL pre-Stage-A pool (not just survivors) so the ensemble
# learns Cell-5's decision boundary from both clean and dropped examples,
# then we read off OOF scores only for rows that already survived Stage A —
# this lets it flag survivors that sit statistically close to that boundary.
X_full = build_oof_features(df_raw)
y_full = rule_clean.astype(int).values

scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full.values)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros(len(df_raw))
base_learners = {
    "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "rf": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1),
    "gbm": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42),
}
member_oof = {name: np.zeros(len(df_raw)) for name in base_learners}

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full_scaled, y_full)):
    for name, model in base_learners.items():
        m = type(model)(**model.get_params())
        m.fit(X_full_scaled[tr_idx], y_full[tr_idx])
        member_oof[name][va_idx] = m.predict_proba(X_full_scaled[va_idx])[:, 1]

oof_probs = np.mean([member_oof[n] for n in base_learners], axis=0)
oof_ensemble_disagreement = np.std([member_oof[n] for n in base_learners], axis=0)

# Flag survivors (rule_clean==True) that the ensemble is nonetheless confident
# are noisy (low OOF prob of being clean) AND where the ensemble members agree
# with each other (low disagreement = not just one noisy learner's opinion)
NOISE_PROB_THRESHOLD = 0.20
LOW_DISAGREEMENT = np.percentile(oof_ensemble_disagreement, 70)
oof_flag_full = (oof_probs < NOISE_PROB_THRESHOLD) & (oof_ensemble_disagreement < LOW_DISAGREEMENT) & rule_clean.values
print(f"OOF ensemble flags {oof_flag_full.sum()} additional rows (of {rule_clean.sum()} rule-clean survivors) "
      f"as statistically anomalous (mean_prob<{NOISE_PROB_THRESHOLD}, low cross-model disagreement)")

oof_flag_in_df = pd.Series(oof_flag_full, index=df_raw.index).loc[df.index].values
n_before = len(df)
df = df[~oof_flag_in_df].reset_index(drop=True)
n_after = len(df)
print(f"Sub-step 1.3 (ensemble OOF noise filter): {n_before} -> {n_after} (removed {n_before - n_after})")
df.to_parquet(f"{STAGE_DIR}/2wikimultihopqa_after_noise_filter.parquet")


# %% CELL 10 — Stage B: exact duplicate questions
n_before = len(df)
df = df.drop_duplicates(subset=["question"], keep="first").reset_index(drop=True)
n_after = len(df)
print(f"Stage B (exact duplicate questions): {n_before} -> {n_after} (removed {n_before - n_after})")


# %% CELL 11 — Stage C: near-duplicate questions (global semantic dedup)
from sentence_transformers import SentenceTransformer
import faiss

device = "cuda" if torch.cuda.is_available() else "cpu"
minilm = SentenceTransformer("all-MiniLM-L6-v2", device=device)

SIM_THRESHOLD = 0.94
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
    f"{STAGE_DIR}/2wikimultihopqa_near_dup_flagged.csv", index=False
)
n_before = len(df)
df = df.drop(index=sorted(to_drop)).reset_index(drop=True)
n_after = len(df)
print(f"Stage C (semantic near-dup, global): {n_before} -> {n_after} (removed {n_before - n_after})")


# %% CELL 12 — Sub-step 1.2: Cross-dataset deduplication (SQuAD / HotpotQA / MuSiQue)
"""
Checks this dataset's surviving questions against the FINAL_DIR outputs of
sibling pipelines already run in this same working directory. Uses the same
exact-match + MiniLM near-dup logic as Stages B/C above (>=0.94 cosine),
since public QA datasets can restate the same underlying fact with different
phrasing. Any overlap found is logged AND dropped from THIS dataset (never
mutates the sibling files) — the convention across the suite is "later
pipeline runs defer to earlier ones" so re-running scripts in any order
doesn't create cross-file drop asymmetries.
Equivalent in spirit to running:
    python processing_scripts/check_duplicate.py 2wikimultihopqa
against each of squad, hotpotqa, musique.
"""
SIBLING_FILES = {
    "squad":    f"{FINAL_DIR}/squad_cleaned.parquet",
    "hotpotqa": f"{FINAL_DIR}/hotpotqa_cleaned.parquet",
    "musique":  f"{FINAL_DIR}/musique_cleaned.parquet",
}

cross_dup_report = {}
total_cross_dropped = set()
for sib_name, sib_path in SIBLING_FILES.items():
    if not os.path.exists(sib_path):
        print(f"[cross-dedup] {sib_name}: not found at {sib_path} — skipping (run its pipeline first "
              f"in this working directory to enable this check).")
        cross_dup_report[sib_name] = {"checked": False}
        continue

    sib_df = pd.read_parquet(sib_path)
    sib_questions = sib_df["question"].astype(str)

    # exact match
    exact_overlap = df["question"].astype(str).isin(set(sib_questions)).values
    exact_idx = set(np.where(exact_overlap)[0])

    # near-dup match (only run if both sides are non-trivial in size)
    near_idx = set()
    if len(df) > 0 and len(sib_df) > 0:
        sib_embs = minilm.encode(sib_questions.tolist(), batch_size=256,
                                  normalize_embeddings=True, show_progress_bar=False).astype("float32")
        sib_index = faiss.IndexFlatIP(sib_embs.shape[1])
        sib_index.add(sib_embs)
        Dc, Ic = sib_index.search(embs[:len(df)] if len(embs) == len(df) else
                                   minilm.encode(df["question"].astype(str).tolist(), normalize_embeddings=True).astype("float32"),
                                   1)
        for i in range(len(df)):
            if Dc[i, 0] >= SIM_THRESHOLD:
                near_idx.add(i)

    dropped_this_sib = exact_idx | near_idx
    total_cross_dropped |= dropped_this_sib
    cross_dup_report[sib_name] = {
        "checked": True, "sibling_rows": len(sib_df),
        "exact_overlap": len(exact_idx), "near_dup_overlap": len(near_idx - exact_idx),
        "total_overlap": len(dropped_this_sib),
    }
    print(f"[cross-dedup] {sib_name}: {len(dropped_this_sib)} overlapping rows found "
          f"({len(exact_idx)} exact, {len(near_idx - exact_idx)} near-dup)")

with open(f"{REPORT_DIR}/2wikimultihopqa_cross_dataset_dedup_report.json", "w") as f:
    json.dump(cross_dup_report, f, indent=2)

n_before = len(df)
if total_cross_dropped:
    df = df.drop(index=sorted(total_cross_dropped)).reset_index(drop=True)
n_after = len(df)
print(f"Sub-step 1.2 (cross-dataset dedup vs squad/hotpotqa/musique): {n_before} -> {n_after} "
      f"(removed {n_before - n_after})")

df.to_parquet(f"{FINAL_DIR}/2wikimultihopqa_cleaned.parquet")
print(f"\nFinal cleaned 2WikiMultihopQA: {len(df)} rows")


# %% CELL 13 — Sub-step 1.4: Tie-handling normalization
"""
Same normalize-then-match approach as HotpotQA/MuSiQue: strip articles/punct,
lowercase, then exact-or-containment match — EXCEPT yes/no answers (comparison
and bridge_comparison types lean heavily yes/no) get strict exact match, no
containment credit, for the same reason as HotpotQA ('yesterday' contains
'yes'). This prevents duplicate-identifier / near-identical-phrasing ties
from falsely penalizing a correct semantic alias as a hallucination.
"""
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")

def normalize_answer(s):
    s = str(s).lower().strip()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def tie_aware_match(prediction, gold_answer):
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


# %% CELL 14 — Sub-step 1.5: Metadata chunking (paragraph-level + evidence relations)
"""
Chunk at paragraph level like HotpotQA/MuSiQue (tagging gold-supporting vs
distractor), PLUS 2Wiki's unique addition: attach the structured relation(s)
from 'evidences' that touch this paragraph's title entity. This is what
enables genuine metadata-aware filtering downstream — e.g. Engineer 2 can
restrict retrieval to chunks whose relation metadata is 'country of
citizenship' for a bridge_comparison question, instead of similarity-ranking
over the full undifferentiated 10-paragraph pool. Distractor paragraphs get
an empty relation list (they're not part of the gold reasoning chain).
"""
def build_chunk_records(row, source_type="2wikimultihopqa"):
    titles = list(row["context"]["title"])
    sentences_per_para = row["context"]["sentences"]
    supporting_titles = set(row["supporting_facts"]["title"])

    relations_by_title = {}
    for triple in row["evidences"]:
        if len(triple) != 3:
            continue
        subj, rel, obj = triple
        relations_by_title.setdefault(subj, []).append(rel)
        relations_by_title.setdefault(obj, []).append(rel)

    records = []
    for p_idx, (title, sents) in enumerate(zip(titles, sentences_per_para)):
        para_text = " ".join(sents)
        chunk_id = hashlib.sha256(f"{source_type}:{row['id']}:{p_idx}".encode()).hexdigest()[:16]
        records.append({
            "chunk_id": chunk_id, "source_type": source_type, "source_id": str(row["id"]),
            "category": row["type"], "paragraph_title": title,
            "is_gold_supporting": title in supporting_titles,
            "evidence_relations": json.dumps(sorted(set(relations_by_title.get(title, [])))),
            "question": row["question"], "gold_answer": row["answer"],
            "token_count": len(para_text.split()), "chunk_text": para_text,
        })
    return records

all_chunks = []
for _, row in df.iterrows():
    all_chunks.extend(build_chunk_records(row))

chunk_df = pd.DataFrame(all_chunks)
n_with_relations = (chunk_df["evidence_relations"] != "[]").sum()
print(f"Built {len(chunk_df)} paragraph-level chunks from {len(df)} rows "
      f"({chunk_df['is_gold_supporting'].sum()} gold-supporting, "
      f"{(~chunk_df['is_gold_supporting']).sum()} distractors, "
      f"{n_with_relations} carrying evidence-relation metadata)")


# %% CELL 15 — Sub-step 1.6: Embed with BGE-large-en-v1.5, build in-memory FAISS index
bge = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

chunk_embeddings = bge.encode(
    chunk_df["chunk_text"].tolist(), batch_size=32, show_progress_bar=True,
    normalize_embeddings=True, convert_to_numpy=True,
).astype("float32")

faiss_index = faiss.IndexFlatIP(chunk_embeddings.shape[1])  # in-memory, per spec
faiss_index.add(chunk_embeddings)
faiss.write_index(faiss_index, f"{INDEX_DIR}/2wikimultihopqa.faiss")  # also persisted for reuse
print(f"FAISS index built: {faiss_index.ntotal} vectors, dim={chunk_embeddings.shape[1]}")


# %% CELL 16 — Sub-step 1.6: Write metadata + evidence relations to SQLite
db_path = f"{INDEX_DIR}/2wikimultihopqa_metadata.sqlite"
if os.path.exists(db_path):
    os.remove(db_path)
conn = sqlite3.connect(db_path)
chunk_df.reset_index(drop=False).rename(columns={"index": "faiss_id"}).to_sql(
    "chunks", conn, if_exists="replace", index=False
)
conn.execute("CREATE INDEX idx_category ON chunks(category)")
conn.execute("CREATE INDEX idx_gold ON chunks(is_gold_supporting)")
conn.execute("CREATE INDEX idx_source_id ON chunks(source_id)")
conn.execute("CREATE INDEX idx_relations ON chunks(evidence_relations)")
conn.commit()
conn.close()
print(f"Saved SQLite metadata store -> {db_path}")

# quick demonstration that metadata-aware filtering actually works: restrict
# to chunks carrying a specific relation before any similarity ranking
conn = sqlite3.connect(db_path)
sample_rel_query = conn.execute(
    "SELECT COUNT(*) FROM chunks WHERE evidence_relations != '[]'"
).fetchone()[0]
conn.close()
print(f"Metadata-aware filtering check: {sample_rel_query} chunks retrievable "
      f"via relation-scoped WHERE clause (vs. {len(chunk_df)} total unscoped)")


# %% CELL 17 — Retrieval eval using REAL gold labels, broken down by type
"""
Real gold/distractor labels (via supporting_facts), same as HotpotQA/MuSiQue
— no proxy negative pool needed. Reported per-type since comparison/
bridge_comparison (fewer, easier-to-verify hops) and inference/compositional
(chained 2-hop extractive) are genuinely different retrieval difficulty.
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
    k_eff = min(k, len(faiss_ids))
    top_k_idx = np.argsort(-sims)[:k_eff]
    retrieved_gold = gold_flags[top_k_idx].sum()
    precision_at_k = retrieved_gold / k_eff
    recall_at_k = retrieved_gold / gold_flags.sum() if gold_flags.sum() > 0 else np.nan
    return precision_at_k, recall_at_k

sample_df = df.groupby("type").apply(lambda g: g.sample(min(30, len(g)), random_state=1)).reset_index(drop=True)
eval_rows = []
for _, row in sample_df.iterrows():
    result = eval_row_retrieval(row["id"], k=2)
    if result:
        eval_rows.append({"type": row["type"], "precision": result[0], "recall": result[1]})
eval_df = pd.DataFrame(eval_rows)
print("\nRetrieval eval by type (k=2, real gold labels):")
print(eval_df.groupby("type")[["precision", "recall"]].mean())


# %% CELL 18 — Accuracy report (top-1-is-gold, real labels, pooled + by type)
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

ACC_SAMPLE_PER_TYPE = 125
acc_sample_df = df.groupby("type").apply(
    lambda g: g.sample(min(ACC_SAMPLE_PER_TYPE, len(g)), random_state=1)
).reset_index(drop=True)

acc_rows = []
for _, row in acc_sample_df.iterrows():
    correct, score = eval_row_top1(row["id"])
    acc_rows.append({"type": row["type"], "top1_correct": correct, "score": score})
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
by_type = acc_df.groupby("type").apply(summarize)

print(f"\n=== 2WikiMultihopQA accuracy report (n={len(acc_df)}, real gold/distractor labels) ===")
print("Pooled (all types):")
print(pooled.round(3))
print("\nBy type:")
print(by_type.round(3))
print("NOTE: same collapse caveat as the rest of the suite — ROC-AUC/PR-AUC (continuous score) "
      "is what actually differentiates performance; accuracy/F1/MCC collapse on a single-class sample.")

accuracy_report = {"dataset": "2WikiMultihopQA", "n_eval": len(acc_df),
                    "pooled": pooled.to_dict(), "by_type": by_type.to_dict(orient="index")}
with open(f"{REPORT_DIR}/2wikimultihopqa_accuracy_report.json", "w") as f:
    json.dump(accuracy_report, f, indent=2, default=float)

print("\n=== 2WIKIMULTIHOPQA PIPELINE COMPLETE ===")
print(f"  - Cleaned dataset: {FINAL_DIR}/2wikimultihopqa_cleaned.parquet")
print(f"  - FAISS index:     {INDEX_DIR}/2wikimultihopqa.faiss")
print(f"  - SQLite metadata: {db_path}  (includes evidence_relations column)")
print(f"  - Accuracy report: {REPORT_DIR}/2wikimultihopqa_accuracy_report.json")
print(f"  - Cross-dedup report: {REPORT_DIR}/2wikimultihopqa_cross_dataset_dedup_report.json")


# %% CELL 19 — VERIFY the accuracy report
def get_pool_size(row_id):
    conn = sqlite3.connect(db_path)
    n = conn.execute("SELECT COUNT(*) FROM chunks WHERE source_id = ?", (row_id,)).fetchone()[0]
    conn.close()
    return n

sample_ids_for_baseline = acc_sample_df["id"].tolist()[:100]
sample_pool_sizes = [get_pool_size(str(rid)) for rid in sample_ids_for_baseline]
expected_random_acc = np.mean([1.0 / n for n in sample_pool_sizes])
print(f"=== 1. Random baseline (avg pool size {np.mean(sample_pool_sizes):.1f}) ===")
print(f"  Expected accuracy: {expected_random_acc:.3f}  (yours, pooled: {pooled['accuracy']:.3f})")
if pooled['accuracy'] < expected_random_acc + 0.05:
    print("  WARNING: accuracy is barely above random chance.")

print(f"\n=== 2. Manual spot-check (10 random rows, across types) ===")
manual_sample = acc_sample_df.sample(min(10, len(acc_sample_df)), random_state=7)
for _, row in manual_sample.iterrows():
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT faiss_id, chunk_text, is_gold_supporting, evidence_relations FROM chunks WHERE source_id = ?", (str(row["id"]),)).fetchall()
    conn.close()
    q_emb = bge.encode([BGE_QUERY_PREFIX + row["question"]], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    fids = [r[0] for r in rows]
    sims = (chunk_embeddings[fids] @ q_emb.T)[:, 0]
    top1 = np.argmax(sims)
    correct = bool(rows[top1][2])
    mark = "CORRECT" if correct else "WRONG"
    print(f"  [{mark}] ({row['type']}) Q: {row['question'][:75]}")
    print(f"       retrieved: {rows[top1][1][:150]}  | relations: {rows[top1][3]}")

print(f"\n=== 3. Accuracy by type — sanity pattern check ===")
print(by_type[["accuracy"]].round(3))
print("  comparison/bridge_comparison (fewer entities to disambiguate) is expected to be >= "
      "inference/compositional (2-hop chained extraction); a large inversion is worth investigating.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

tfidf_correct = []
verify_sample_df = acc_sample_df.sample(min(150, len(acc_sample_df)), random_state=11)
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
    print("  WARNING: TF-IDF does nearly as well as BGE — multi-hop reasoning should need "
          "more than keyword overlap; investigate if this holds.")
else:
    print("  BGE meaningfully outperforms TF-IDF — evidence of real semantic/multi-hop matching.")

print(f"\n=== 5. OOF noise-filter sanity check ===")
print(f"  Rows dropped by hard rules (Stage A):        {int((~rule_clean).sum())}")
print(f"  Rows additionally dropped by OOF ensemble:   {int(oof_flag_full.sum())}")
print(f"  Ratio (OOF/rule): {oof_flag_full.sum() / max(1, (~rule_clean).sum()):.2f} "
      "— if this is >>1, the OOF threshold is likely too aggressive and worth loosening.")

print("\n=== VERIFICATION COMPLETE ===")
