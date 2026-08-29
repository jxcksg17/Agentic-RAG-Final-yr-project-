"""
Entrypoint for Sub-steps 3.1 + 3.2 — this IS Probing-RAG's mechanism
(Baek et al. 2025, Findings of NAACL 2025), per the team's own
Metrics Deep-Dive PDF: "Trains a lightweight probe (a small MLP) on
the LLM's own hidden-state activations to predict whether the model
already has sufficient internal knowledge to answer without
retrieval." Every design choice below is checked against that paper's
stated training recipe, not just work_Dist.pdf's brief.

    python train_phase1_gate.py

1. Loads Engineer 1's cleaned query bank.
2. For each query: generate a no-retrieval answer with Phi-4-mini-instruct,
   capture the hidden state at the last prompt token (Sub-step 3.1 /
   Probing-RAG's "fixed intermediate layer"), and label it
   correct/incorrect against gold via EM (Probing-RAG's own labeling
   recipe: "label 'retrieve' or 'don't-retrieve' based on whether the
   no-retrieval answer was correct, via EM/accuracy against gold").
3. Trains the Phase1GateMLP on (hidden_state -> correctness) with BCE
   (Sub-step 3.2 / Probing-RAG's stated training objective).
4. Reports EM-based val accuracy/F1 AND downstream token-F1 on the
   no-retrieval answers themselves, since the Deep-Dive PDF lists both
   EM and F1 as Probing-RAG's evaluation-time metrics, not EM alone.
5. Saves the checkpoint + normalization stats for the Router Agent to
   load at inference time.
"""

import torch
from tqdm import tqdm

from config import Paths, ModelConfig, Phase1GateConfig
from src.hooks import HiddenStateExtractor
from src.phase1_gate import train_phase1_gate
from src.sklearn_probes import train_accuracy_probe_reference
from src.data_utils import load_query_bank, is_correct, token_f1


def generate_no_retrieval_answer(extractor: HiddenStateExtractor, query: str) -> str:
    """Greedy no-retrieval generation, mirrors Engineer 2's arm (1) of
    Sub-step 2.2 ("No external retrieval")."""
    inputs = extractor.tokenizer(query, return_tensors="pt").to(extractor.model.device)
    out_ids = extractor.model.generate(
        **inputs, max_new_tokens=64, do_sample=False, pad_token_id=extractor.tokenizer.eos_token_id
    )
    new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
    return extractor.tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    paths, model_cfg, gate_cfg = Paths(), ModelConfig(), Phase1GateConfig()

    df = load_query_bank(paths.raw_queries_csv)

    hidden_states, labels, f1_scores = [], [], []
    with HiddenStateExtractor(model_cfg) as extractor:
        for row in tqdm(df.itertuples(), total=len(df), desc="Sub-step 3.1: extracting hidden states"):
            h = extractor.extract(row.query)                       # tap point: last prompt token
            answer = generate_no_retrieval_answer(extractor, row.query)
            hidden_states.append(h)
            labels.append(1.0 if is_correct(answer, row.gold_answer) else 0.0)
            f1_scores.append(token_f1(answer, row.gold_answer))

    X = torch.stack(hidden_states)          # (N, hidden_dim)
    y = torch.tensor(labels)                # (N,) EM-based sufficiency label
    mean_f1 = sum(f1_scores) / len(f1_scores)

    print(f"Collected {len(y)} examples, EM positive rate = {y.mean().item():.3f}, "
          f"mean no-retrieval token-F1 = {mean_f1:.3f}")

    model, history = train_phase1_gate(X, y, gate_cfg, model_cfg)

    # NOTE: this is NOT a second reading of Probing-RAG. It's an
    # unrelated baseline borrowed from the SEP repo (OATML/semantic-
    # entropy-probes' own "accuracy probe") — a plain LogisticRegression
    # on the same hidden states, kept here purely as a sanity check that
    # the brief's <10M-param MLP is earning its extra capacity over the
    # simplest possible linear alternative, not as a competing
    # implementation of Probing-RAG itself (Probing-RAG's own paper
    # specifies an MLP, which is src/phase1_gate.py).
    _, ref_result = train_accuracy_probe_reference(X.numpy(), y.numpy(), gate_cfg.val_split)
    print(f"Probing-RAG-style MLP gate val_acc={history['val_acc'][-1]:.3f}  vs.  "
          f"logistic-regression sanity-check val_acc={ref_result.val_accuracy:.3f} "
          f"(val_auroc={ref_result.val_auroc:.3f})")

    torch.save({
        "state_dict": model.state_dict(),
        "norm_mean": history["norm_mean"],
        "norm_std": history["norm_std"],
        "n_params": model.n_params,
        "mean_no_retrieval_f1": mean_f1,
    }, paths.phase1_gate_ckpt)
    print(f"Saved Phase 1 gate probe ({model.n_params:,} params) -> {paths.phase1_gate_ckpt}")


if __name__ == "__main__":
    main()
