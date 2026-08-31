"""
Sub-step 3.1 — Tensor Interception.

Loads Phi-4-mini-instruct via plain `transformers` (NOT the vLLM server
Engineer 2 stands up — see requirements.txt) and registers forward hooks
on decoder layers to capture hidden states at:
  - TBG (token-before-generation): last prompt token, right before the
    model commits to a response. This is the exact tap point Sub-step
    3.1 specifies, and is what the Phase 1 gate uses (it must run
    BEFORE any answer exists).
  - SLT (second-last-token / token-before-EOS of the model's own
    answer): captured after a full generation pass. Not usable for the
    Phase 1 gate (defeats the purpose — the answer already exists by
    then), but the SEP literature (Kossen et al. 2024) finds it a
    stronger predictor of entropy when it IS available, so it's exposed
    here as an ablation for the SEP only (config.SEPConfig.compare_tbg_vs_slt).

Supports single-layer capture (the brief's default) and multi-layer
capture (the layer-concatenation ensemble improvement in
src/sklearn_probes.py), from ONE forward/generate pass either way.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import ModelConfig


class HiddenStateExtractor:
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        self.cfg = cfg
        self._captured: dict[int, torch.Tensor] = {}

        quant_config = None
        if cfg.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            quantization_config=quant_config,
            torch_dtype=getattr(torch, cfg.dtype),
            device_map="auto",
            output_hidden_states=False,  # we grab hidden states via hooks,
                                          # so the same hooks work whether
                                          # we call model() or model.generate()
        )
        self.model.eval()
        self.n_layers = len(self._decoder_layers())
        self._handles: list = []
        self._register(layer_indices=[cfg.probe_layer_index])

    def _decoder_layers(self):
        # Phi-4-mini-instruct follows the standard
        # model.model.layers[i] decoder-layer indexing used by most
        # HF causal LMs (Llama/Phi family).
        try:
            return self.model.model.layers
        except AttributeError as e:
            raise RuntimeError(
                "Could not resolve decoder layers — inspect "
                "`model.named_modules()` and update _decoder_layers() if "
                "Phi-4-mini-instruct's module tree differs."
            ) from e

    def _register(self, layer_indices: list[int]):
        for h in self._handles:
            h.remove()
        self._handles = []
        layers = self._decoder_layers()
        for idx in layer_indices:
            handle = layers[idx].register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)
        self._active_layers = list(layer_indices)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._captured[layer_idx] = hidden.detach()
        return hook_fn

    @torch.no_grad()
    def extract(self, prompt: str, layer_index: int | None = None) -> torch.Tensor:
        """
        TBG extraction (Sub-step 3.1's tap point): forward-only pass
        (no generation), hidden state at the last prompt token, single
        layer. This is what Sub-step 3.2 and the production
        `phase1_needs_retrieval` gate use.
        """
        idx = layer_index if layer_index is not None else self.cfg.probe_layer_index
        if [idx] != self._active_layers:
            self._register([idx])
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        self.model(**inputs)
        return self._captured[idx][0, -1, :].float().cpu()

    @torch.no_grad()
    def extract_multi_layer(self, prompt: str, layer_indices: list[int]) -> torch.Tensor:
        """
        TBG extraction across several adjacent layers in one forward
        pass, concatenated into a single feature vector. Backs the
        layer-window ensemble improvement in src/sklearn_probes.py —
        the paper's own ablation shows this beats any single layer.
        """
        if set(layer_indices) != set(self._active_layers):
            self._register(layer_indices)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        self.model(**inputs)
        vecs = [self._captured[idx][0, -1, :].float().cpu() for idx in layer_indices]
        return torch.cat(vecs, dim=-1)

    @torch.no_grad()
    def extract_tbg_and_slt(self, prompt: str, layer_index: int | None = None,
                             max_new_tokens: int = 64) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Runs ONE generation call and returns both tap points plus the
        generated text:
          - TBG: hidden state at the last prompt token (before any
            generation — same value `extract()` would give).
          - SLT: hidden state at the token immediately before the
            model's own EOS (i.e. after it has committed to a full
            answer). Ablation-only; see module docstring.
        """
        idx = layer_index if layer_index is not None else self.cfg.probe_layer_index
        if [idx] != self._active_layers:
            self._register([idx])

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Capture TBG on the prompt-only forward pass first (hook fires
        # once per forward call `generate` makes internally; grabbing it
        # from the very first step is equivalent to `extract()`'s own pass).
        self.model(**inputs)
        tbg = self._captured[idx][0, -1, :].float().cpu()

        out_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        answer_ids = out_ids[0][prompt_len:]
        answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=True)

        # Re-run a forward pass over the full (prompt + answer) sequence
        # to get a hidden state at the token before EOS -- generate()
        # doesn't expose intermediate hidden states through our hook
        # cheaply, so SLT costs one extra forward pass. This is the
        # explicit cost/accuracy trade the ablation is measuring.
        full_ids = out_ids[:, : out_ids.shape[1] - 1] if out_ids.shape[1] > prompt_len else out_ids
        self.model(input_ids=full_ids)
        slt = self._captured[idx][0, -1, :].float().cpu()

        return tbg, slt, answer_text

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
