"""
This is the file Engineer 2 (Router Agent) and Engineer 4 (Evaluator
Agent / Pipeline Fusion) actually import — everything above this line
is training-time-only.

Runtime cost per query: exactly ONE forward pass through Phi-4-mini
(shared with whatever forward pass the pipeline already does), plus
two tiny matmuls (<10M and ~3K params). No extra generation calls.
"""

from __future__ import annotations

import torch

from config import Paths, ModelConfig, SEPConfig
from src.hooks import HiddenStateExtractor
from src.phase1_gate import Phase1GateMLP
from src.sep_probe import SemanticEntropyProbe


class LatentProbes:
    """
    Loads both trained probes once; call per-query at inference time.

    The SEP checkpoint (written by the updated train_sep.py) carries
    BOTH probe variants — the reference sklearn LogisticRegression
    (binarized SE, default) and the brief-literal PyTorch regressor
    (continuous SE). `use_reference_sep` picks which one
    `entropy_uncertainty_score()` actually calls.
    """

    def __init__(self, paths: Paths = Paths(), model_cfg: ModelConfig = ModelConfig(),
                 sep_cfg: SEPConfig = SEPConfig(), use_reference_sep: bool = True):
        self.model_cfg = model_cfg
        self.use_reference_sep = use_reference_sep and sep_cfg.use_reference_architecture
        self.extractor = HiddenStateExtractor(model_cfg)

        gate_ckpt = torch.load(paths.phase1_gate_ckpt, map_location="cpu")
        self.gate = Phase1GateMLP(model_cfg)
        self.gate.load_state_dict(gate_ckpt["state_dict"])
        self.gate.eval()
        self.gate_mean, self.gate_std = gate_ckpt["norm_mean"], gate_ckpt["norm_std"]

        sep_ckpt = torch.load(paths.sep_ckpt, map_location="cpu")
        self.sep_layer_indices = sep_ckpt.get("layer_indices", [model_cfg.probe_layer_index])

        if self.use_reference_sep:
            self.sklearn_sep = sep_ckpt["sklearn_model"]
            self.sep_median = sep_ckpt["median_threshold"]
        else:
            self.sep = SemanticEntropyProbe(model_cfg)
            self.sep.load_state_dict(sep_ckpt["pytorch_state_dict"])
            self.sep.eval()
            self.sep_mean, self.sep_std = sep_ckpt["pytorch_norm_mean"], sep_ckpt["pytorch_norm_std"]

    @torch.no_grad()
    def phase1_needs_retrieval(self, query: str, confidence_threshold: float = 0.5) -> bool:
        """
        Called by the Router Agent (Phase 1). Returns True if retrieval
        should proceed, False if the model is confident enough to
        answer directly -- the "Phase 1 exit" branch in the methodology
        PDF's workflow pseudocode (step 1).
        """
        h = self.extractor.extract(query)
        h_norm = (h - self.gate_mean) / self.gate_std
        p_correct = torch.sigmoid(self.gate(h_norm.unsqueeze(0))).item()
        return p_correct < confidence_threshold

    @torch.no_grad()
    def entropy_uncertainty_score(self, query: str) -> float:
        """
        Called by the Evaluator Agent (Phase 3) as its
        "entropy/uncertainty scorer" sub-tool (Table 1). One extra
        forward pass (or one pass over a small layer window), reusing
        the same tap point/layers as training.

        Reference path (default): returns P(high semantic uncertainty)
        in [0, 1] from the LogisticRegression SEP — this is the number
        the Evaluator's `adaptive_threshold` step should fuse, since a
        calibrated probability composes more naturally with the
        relevance/faithfulness scores than an unbounded nats value.

        Brief-literal path: returns the raw predicted entropy in nats,
        for call sites that specifically want the continuous score
        described in Sub-step 3.4's wording.
        """
        if self.use_reference_sep:
            h = (self.extractor.extract_multi_layer(query, self.sep_layer_indices)
                 if len(self.sep_layer_indices) > 1
                 else self.extractor.extract(query, self.sep_layer_indices[0]))
            p_high_uncertainty = self.sklearn_sep.predict_proba(h.numpy().reshape(1, -1))[0, 1]
            return float(p_high_uncertainty)

        h = self.extractor.extract(query, self.sep_layer_indices[0])
        h_norm = (h - self.sep_mean) / self.sep_std
        predicted_entropy = self.sep(h_norm.unsqueeze(0)).item()
        return max(predicted_entropy, 0.0)  # entropy is non-negative

    def close(self):
        self.extractor.close()
