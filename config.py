"""
Shared config for Engineer 3's pipeline.

Every number here is pinned to a constraint stated in the two source
documents (work_Dist.pdf sub-steps 3.1-3.4, and the methodology PDF's
Router Agent / Evaluator Agent spec) so that Engineer 3's outputs stay
compatible with Engineer 2's router and Engineer 4's fused evaluator.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Paths:
    raw_queries_csv: str = "data/engineer1_cleaned_queries.csv"     # handoff from Engineer 1
    router_labels_csv: str = "data/engineer2_silver_labels.csv"     # handoff from Engineer 2 (0/1/2)
    hidden_states_dir: str = "artifacts/hidden_states"
    phase1_gate_ckpt: str = "artifacts/phase1_gate_probe.pt"
    entropy_labels_csv: str = "artifacts/true_semantic_entropy.csv"
    sep_ckpt: str = "artifacts/semantic_entropy_probe.pt"


@dataclass(frozen=True)
class ModelConfig:
    # Same base model Engineer 2 deploys (Sub-step 2.1), loaded a second
    # time in plain transformers so we can attach forward hooks.
    model_id: str = "microsoft/Phi-4-mini-instruct"
    load_in_4bit: bool = True          # mirrors Engineer 2's ~2.5GB VRAM budget
    dtype: str = "bfloat16"

    # Sub-step 3.1: "intermediate Transformer layers ... token right
    # before generating the response". We tap a late-middle layer
    # (empirically where semantic content is most linearly decodable —
    # Kossen et al. 2024 report layers ~1/2 to 3/4 depth work best) and
    # keep it configurable so it can be swept.
    probe_layer_index: int = 16        # Phi-4-mini-instruct has 32 layers -> mid/late layer
    hidden_dim: int = 3072             # Phi-4-mini-instruct hidden size


@dataclass(frozen=True)
class Phase1GateConfig:
    # Sub-step 3.2: "<10M parameter PyTorch MLP", trained with BCE.
    max_params_budget: int = 10_000_000
    mlp_hidden_dims: tuple = (512, 128)
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 64
    val_split: float = 0.15


@dataclass(frozen=True)
class SemanticEntropyConfig:
    # Sub-step 3.3: "5 to 10 distinct answers per query for a batch of
    # 500 questions".
    n_samples_per_query: int = 8
    n_calibration_queries: int = 500
    sampling_temperature: float = 1.0   # must be >0 to get answer diversity
    sampling_top_p: float = 0.95
    # Bidirectional-entailment clustering model — this is the exact model
    # id used in the authors' own released code (OATML/semantic-entropy-
    # probes, jlko/semantic_uncertainty), not a substitute. Kept distinct
    # from Engineer 4's MiniCheck-FT5 faithfulness model on purpose:
    # entailment-for-clustering answers to each other is a different task
    # from entailment-for-grounding an answer in a retrieved chunk.
    nli_model_id: str = "microsoft/deberta-v2-xlarge-mnli"
    # False (paper default) = equivalent unless contradiction appears or
    # both directions are merely neutral. True = both directions must be
    # full entailment. See src/semantic_entropy.py::_are_equivalent.
    strict_entailment: bool = False


@dataclass(frozen=True)
class SEPConfig:
    """
    Two probe variants are provided (src/sklearn_probes.py vs
    src/sep_probe.py) because the two source documents disagree slightly:
      - work_Dist.pdf Sub-step 3.4 literally says "predict the entropy
        scores" -> reads as regression on the continuous value.
      - Kossen et al. 2024's actual released code (train-latent-probe.ipynb)
        trains sklearn LogisticRegression on a BINARIZED (median-split)
        entropy label and reports AUROC/accuracy, not MSE.
    `use_reference_architecture=True` runs the paper's own architecture
    faithfully; the continuous PyTorch regressor is kept as the literal
    reading of the team brief. Both consume identical hidden states, so
    switching costs nothing upstream.
    """
    use_reference_architecture: bool = True
    binarize_at_median: bool = True     # median-split entropy -> {0,1}, per official notebook

    # PyTorch continuous-regression variant (src/sep_probe.py)
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 32
    val_split: float = 0.15

    # --- Improvements layered on top of the base paper architecture ---
    # 1. Token-position ablation: the paper compares TBG (token-before-
    #    generation, what Sub-step 3.1 specifies) against SLT (second-
    #    last-token / token-before-EOS of the model's OWN answer) and
    #    finds SLT is often stronger since it has seen the full answer.
    #    TBG is kept as the production default (matches the brief and
    #    allows a zero-generation-cost abstain-before-answering gate);
    #    SLT is exposed as an ablation researchers can run for comparison.
    compare_tbg_vs_slt: bool = True
    # 2. Layer-concatenation ensembling: the paper's own bootstrap
    #    analysis shows concatenating a small window of adjacent layers
    #    (instead of a single probe_layer_index) improves AUROC over any
    #    single layer. sklearn_probes.py exposes this as layer_window.
    layer_window: tuple = (14, 18)      # inclusive layer index range around config.probe_layer_index
