"""
Sub-step 3.4 — Semantic Entropy Probe (SEP) Training.

"Train a second <10M parameter linear probe to predict the calculated
entropy scores from Sub-step 3.3 directly from the hidden states of a
single generation pass. This allows the production system to detect
hallucination risk without the latency penalty of running 5 to 10
separate generation cycles."

This is a linear regression head, following Kossen et al. 2024
("Semantic Entropy Probes: Robust and Cheap Hallucination Detection in
LLMs") — SEPs are deliberately linear (not an MLP like Sub-step 3.2's
gate) because the paper's central finding is that semantic-entropy
information is *already linearly decodable* from a single hidden
state, which is what makes the probe cheap and robust across domains.

Feeds the methodology PDF's Evaluator Agent (Phase 3): this is the
"entropy/uncertainty scorer" sub-tool listed in Table 1, called at
inference time with zero extra generation cost.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from config import ModelConfig, SEPConfig


class SemanticEntropyProbe(nn.Module):
    """Strictly linear: hidden_dim -> 1. hidden_dim=3072 => ~3.07K params."""

    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.linear = nn.Linear(cfg.hidden_dim, 1)
        n_params = sum(p.numel() for p in self.parameters())
        assert n_params < 10_000_000, f"SEP has {n_params:,} params, exceeds 10M budget"
        self.n_params = n_params

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h).squeeze(-1)  # predicted entropy, nats


class SEPDataset(Dataset):
    """
    hidden_states: (N, hidden_dim) — Sub-step 3.1 extraction from a
        SINGLE generation pass (not one per sampled answer).
    entropy_targets: (N,) — Sub-step 3.3's discrete_semantic_entropy()
        output, computed once offline per calibration query.
    Targets are NOT normalized (entropy has a meaningful zero: perfectly
    consistent answers), but inputs are, matching Sub-step 3.2's
    "normalize the hidden states" instruction and keeping both probes
    consistent.
    """

    def __init__(self, hidden_states: torch.Tensor, entropy_targets: torch.Tensor,
                 mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
        self.mean = hidden_states.mean(0) if mean is None else mean
        self.std = (hidden_states.std(0) + 1e-6) if std is None else std
        self.X = (hidden_states - self.mean) / self.std
        self.y = entropy_targets.float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_sep(hidden_states: torch.Tensor, entropy_targets: torch.Tensor,
              cfg: SEPConfig = SEPConfig(), model_cfg: ModelConfig = ModelConfig()
              ) -> tuple[SemanticEntropyProbe, dict]:
    full_ds = SEPDataset(hidden_states, entropy_targets)
    n_val = int(len(full_ds) * cfg.val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = SemanticEntropyProbe(model_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    history = {"train_mse": [], "val_mse": [], "val_mae": []}

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        model.eval()
        val_loss, val_mae = 0.0, 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * len(xb)
                val_mae += (pred - yb).abs().sum().item()
        val_loss /= max(n_val, 1)
        val_mae /= max(n_val, 1)

        history["train_mse"].append(train_loss)
        history["val_mse"].append(val_loss)
        history["val_mae"].append(val_mae)
        print(f"[sep_probe] epoch {epoch+1}/{cfg.epochs} "
              f"train_mse={train_loss:.4f} val_mse={val_loss:.4f} val_mae={val_mae:.4f}")

    history["norm_mean"] = full_ds.mean
    history["norm_std"] = full_ds.std
    return model, history
