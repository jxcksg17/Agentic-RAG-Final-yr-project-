"""
Entrypoint for Sub-step 3.4 (updated against the Metrics Deep-Dive PDF).

    python train_sep.py

For each calibration query that already has a true-entropy label AND a
ground-truth correctness label (from compute_true_entropy.py), extract
hidden states in a single pass, then train and validate TWO probe
architectures the way the Deep-Dive PDF's own "Metrics used at
evaluation / inference" section for Semantic Entropy Probes specifies:

  "AUROC of the probe's single-pass entropy estimate against
  ground-truth correctness — compared directly to the AUROC of full
  semantic entropy to show minimal quality loss."
  "Correlation coefficient between probe output and true semantic
  entropy, as a fidelity check."

That means training-target accuracy alone (how well the probe predicts
its own binarized-SE label) is NOT what gets reported here as the
headline number — AUROC-vs-correctness and correlation-vs-true-SE are,
because those are the two metrics the source paper and the team's own
literature survey actually report.

  A. Reference architecture (src/sklearn_probes.py) — Kossen et al.
     2024's actual published design: LogisticRegression on binarized
     entropy, with the paper's own layer-window-concatenation
     improvement (config.SEPConfig.layer_window).
  B. Brief-literal architecture (src/sep_probe.py) — a strict PyTorch
     linear layer regressing the raw continuous entropy value, per
     work_Dist.pdf Sub-step 3.4's wording ("predict the entropy scores").
     Also a valid reading per the Deep-Dive PDF: "regression loss (e.g.
     MSE) or classification loss ... depending on probe formulation."

If config.SEPConfig.compare_tbg_vs_slt is True, the same comparison is
repeated using SLT (token-before-EOS of the model's own answer) hidden
states, at the cost of a second forward pass per query.
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Paths, ModelConfig, SEPConfig
from src.hooks import HiddenStateExtractor
from src.sep_probe import train_sep
from src.sklearn_probes import train_sep_reference, layer_window_indices, evaluate_vs_ground_truth


def collect_hidden_states(entropy_df: pd.DataFrame, extractor: HiddenStateExtractor,
                           layer_indices: list[int], want_slt: bool):
    tbg_rows, slt_rows = [], []
    for row in tqdm(entropy_df.itertuples(), total=len(entropy_df), desc="extracting hidden states"):
        if want_slt:
            tbg, slt, _ = extractor.extract_tbg_and_slt(row.query, layer_index=layer_indices[0])
            tbg_rows.append(tbg)
            slt_rows.append(slt)
        else:
            h = (extractor.extract_multi_layer(row.query, layer_indices)
                 if len(layer_indices) > 1 else extractor.extract(row.query, layer_indices[0]))
            tbg_rows.append(h)
    return torch.stack(tbg_rows), (torch.stack(slt_rows) if slt_rows else None)


def run_comparison(tag: str, X: torch.Tensor, entropy_targets: np.ndarray, is_correct: np.ndarray,
                    sep_cfg: SEPConfig, model_cfg: ModelConfig):
    print(f"\n=== {tag} ===")

    print("-- A. Reference architecture (LogisticRegression, binarized SE) --")
    ref_model, ref_result, median = train_sep_reference(X.numpy(), entropy_targets, sep_cfg.val_split)
    ref_scores = ref_model.predict_proba(X.numpy())[:, 1]   # P(high uncertainty), full set for the AUROC check
    ref_gt = evaluate_vs_ground_truth(ref_scores, entropy_targets, is_correct)
    print(f"   vs. ground truth: AUROC={ref_gt['auroc_vs_correctness']:.4f}, "
          f"corr-to-true-SE={ref_gt['pearson_corr_vs_true_se']:.4f}")

    print("-- B. Brief-literal architecture (PyTorch linear, continuous SE) --")
    pt_model, pt_history = train_sep(X, torch.tensor(entropy_targets, dtype=torch.float32), sep_cfg, model_cfg)
    with torch.no_grad():
        X_norm = (X - pt_history["norm_mean"]) / pt_history["norm_std"]
        pt_scores = pt_model(X_norm).numpy()
    pt_gt = evaluate_vs_ground_truth(pt_scores, entropy_targets, is_correct)
    print(f"   vs. ground truth: AUROC={pt_gt['auroc_vs_correctness']:.4f}, "
          f"corr-to-true-SE={pt_gt['pearson_corr_vs_true_se']:.4f}")

    return {
        "tag": tag,
        "reference_model": ref_model, "reference_result": ref_result, "reference_median": median,
        "reference_gt": ref_gt,
        "pytorch_model": pt_model, "pytorch_history": pt_history, "pytorch_gt": pt_gt,
    }


def main():
    paths, model_cfg, sep_cfg = Paths(), ModelConfig(), SEPConfig()

    entropy_df = pd.read_csv(paths.entropy_labels_csv)
    entropy_targets = entropy_df["semantic_entropy"].to_numpy()
    is_correct = entropy_df["is_correct"].to_numpy()

    full_se_gt = evaluate_vs_ground_truth(entropy_targets, entropy_targets, is_correct)
    print(f"Full (expensive) semantic-entropy AUROC vs. correctness: "
          f"{full_se_gt['auroc_vs_correctness']:.4f}  <- ceiling both probes below are compared against")

    layer_indices = (layer_window_indices(model_cfg.probe_layer_index, sep_cfg.layer_window)
                      if sep_cfg.layer_window else [model_cfg.probe_layer_index])
    print(f"Using layer window {layer_indices} "
          f"({'ensembled/concatenated' if len(layer_indices) > 1 else 'single layer'})")

    with HiddenStateExtractor(model_cfg) as extractor:
        X_tbg, X_slt = collect_hidden_states(
            entropy_df, extractor, layer_indices, want_slt=sep_cfg.compare_tbg_vs_slt
        )

    results = [run_comparison("TBG (token-before-generation)", X_tbg, entropy_targets, is_correct, sep_cfg, model_cfg)]
    if X_slt is not None:
        results.append(run_comparison("SLT (token-before-EOS, ablation only)", X_slt, entropy_targets, is_correct, sep_cfg, model_cfg))

    print("\n=== Summary (compare against full-SE ceiling "
          f"AUROC={full_se_gt['auroc_vs_correctness']:.4f}) ===")
    for r in results:
        print(f"{r['tag']}:")
        print(f"  reference (LogReg):  AUROC={r['reference_gt']['auroc_vs_correctness']:.4f}  "
              f"corr-to-true-SE={r['reference_gt']['pearson_corr_vs_true_se']:.4f}")
        print(f"  pytorch (linear reg): AUROC={r['pytorch_gt']['auroc_vs_correctness']:.4f}  "
              f"corr-to-true-SE={r['pytorch_gt']['pearson_corr_vs_true_se']:.4f}")

    # Save whichever architecture the team decides to ship. Default:
    # ship the TBG reference architecture (matches the brief's tap
    # point AND the paper's actual design), keep the PyTorch regressor
    # as an alternative checkpoint for continuous-score use cases.
    primary = results[0]
    torch.save({
        "sklearn_model": primary["reference_model"],
        "median_threshold": primary["reference_median"],
        "layer_indices": layer_indices,
        "pytorch_state_dict": primary["pytorch_model"].state_dict(),
        "pytorch_norm_mean": primary["pytorch_history"]["norm_mean"],
        "pytorch_norm_std": primary["pytorch_history"]["norm_std"],
        "full_se_auroc_ceiling": full_se_gt["auroc_vs_correctness"],
        "reference_auroc_vs_correctness": primary["reference_gt"]["auroc_vs_correctness"],
        "pytorch_auroc_vs_correctness": primary["pytorch_gt"]["auroc_vs_correctness"],
    }, paths.sep_ckpt)
    print(f"\nSaved both SEP variants (TBG) -> {paths.sep_ckpt}")


if __name__ == "__main__":
    main()
