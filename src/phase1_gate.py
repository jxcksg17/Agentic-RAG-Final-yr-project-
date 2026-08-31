"""
Sub-step 3.2 — Phase 1 Microscopic Gate.

"Route a batch of test queries through the LLM without any retrieval.
Track which queries were answered correctly and which failed. Flatten
and normalize the hidden states from Sub-step 3.1, and train a
lightweight (<10M parameter) PyTorch MLP on them using Binary
Cross-Entropy. This creates the gate that stops retrieval if the model
already knows the answer."

Architecturally this feeds the methodology PDF's Router Agent (Phase 1):
"uses a hidden-state probe to decide tier ... and whether retrieval is
needed at all" — this MLP *is* that hidden-state probe.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from config import ModelConfig, Phase1GateConfig


class Phase1GateMLP(nn.Module):
    """
    hidden_dim -> 512 -> 128 -> 1 (sigmoid logit)
    Predicts P(no-retrieval answer is correct) from a single hidden
    state vector. Param count is asserted at construction time against
    the <10M budget from the source doc.
    """

    def __init__(self, cfg: ModelConfig = ModelConfig(), gate_cfg: Phase1GateConfig = Phase1GateConfig()):
        super().__init__()
        dims = [cfg.hidden_dim, *gate_cfg.mlp_hidden_dims]
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.ReLU(), nn.Dropout(gate_cfg.dropout)]
        layers.append(nn.Linear(dims[-1], 1))  # raw logit; use BCEWithLogitsLoss
        self.net = nn.Sequential(*layers)

        n_params = sum(p.numel() for p in self.parameters())
        assert n_params < gate_cfg.max_params_budget, (
            f"Phase1GateMLP has {n_params:,} params, exceeds the "
            f"{gate_cfg.max_params_budget:,} budget from Sub-step 3.2"
        )
        self.n_params = n_params

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class HiddenStateLabelDataset(Dataset):
    """
    Wraps (hidden_state, correctness_label) pairs produced offline by
    running the no-retrieval arm of Engineer 2's Sub-step 2.2 simulation
    through HiddenStateExtractor.extract().
    hidden states are z-score normalized using train-split statistics
    only (per "normalize the hidden states" in the source doc).
    """

    def __init__(self, hidden_states: torch.Tensor, labels: torch.Tensor,
                 mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
        self.mean = hidden_states.mean(0) if mean is None else mean
        self.std = (hidden_states.std(0) + 1e-6) if std is None else std
        self.X = (hidden_states - self.mean) / self.std
        self.y = labels.float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_phase1_gate(hidden_states: torch.Tensor, correctness_labels: torch.Tensor,
                       cfg: Phase1GateConfig = Phase1GateConfig(),
                       model_cfg: ModelConfig = ModelConfig()) -> tuple[Phase1GateMLP, dict]:
    """
    hidden_states: (N, hidden_dim) float tensor, one row per query
                   (no-retrieval arm hidden state from Sub-step 3.1).
    correctness_labels: (N,) 1.0 if the no-retrieval answer matched
                   ground truth, else 0.0.
    Returns the trained probe plus the normalization stats needed at
    inference time (must be saved alongside the checkpoint).
    """
    full_ds = HiddenStateLabelDataset(hidden_states, correctness_labels)
    n_val = int(len(full_ds) * cfg.val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = Phase1GateMLP(model_cfg, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = loss_fn(logits, yb)
                val_loss += loss.item() * len(xb)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == yb).sum().item()
        val_loss /= max(n_val, 1)
        val_acc = correct / max(n_val, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"[phase1_gate] epoch {epoch+1}/{cfg.epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    history["norm_mean"] = full_ds.mean
    history["norm_std"] = full_ds.std
    return model, history
