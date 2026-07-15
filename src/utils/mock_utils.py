"""
Mock classes and state utilities for running the pipeline in mock mode.
Bypasses Hugging Face downloads and heavy models, allowing end-to-end dry runs.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Generator

import torch

logger = logging.getLogger(__name__)

# Global mock states
_MOCK_INTERVENTION_ACTIVE = False
_ACTIVE_TOKENIZER: MockTokenizer | None = None


def is_mock_intervention_active() -> bool:
    global _MOCK_INTERVENTION_ACTIVE
    return _MOCK_INTERVENTION_ACTIVE


def set_mock_intervention_active(val: bool) -> None:
    global _MOCK_INTERVENTION_ACTIVE
    _MOCK_INTERVENTION_ACTIVE = val


def get_active_tokenizer() -> MockTokenizer | None:
    global _ACTIVE_TOKENIZER
    return _ACTIVE_TOKENIZER


def set_active_tokenizer(tokenizer: MockTokenizer) -> None:
    global _ACTIVE_TOKENIZER
    _ACTIVE_TOKENIZER = tokenizer


class MockLayer(torch.nn.Module):
    def __init__(self, layer_idx: int, d_model: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.hooks = []
        self.mlp = torch.nn.Identity()
        self.self_attn = torch.nn.Identity()

    def register_forward_hook(self, hook: callable) -> callable:
        self.hooks.append(hook)

        class HookHandle:
            def __init__(self, hooks, hook):
                self.hooks = hooks
                self.hook = hook

            def remove(self):
                if self.hook in self.hooks:
                    self.hooks.remove(self.hook)

        return HookHandle(self.hooks, hook)


class MockTransformerModel:
    def __init__(self, n_layers: int, d_model: int) -> None:
        self.layers = [MockLayer(i, d_model) for i in range(n_layers)]


class MockModel(torch.nn.Module):
    def __init__(self, n_layers: int = 26, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        self.model = MockTransformerModel(n_layers, d_model)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> MockOutput:
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        device = input_ids.device

        tokenizer = get_active_tokenizer()

        for layer in self.model.layers:
            h = torch.zeros(batch_size, seq_len, self.d_model, device=device)
            for b in range(batch_size):
                if seq_len > 0:
                    prompt_idx_token = input_ids[b, 0].item()
                    prompt_idx = prompt_idx_token - 1000
                    if tokenizer and 0 <= prompt_idx < len(tokenizer.last_prompts):
                        prompt = tokenizer.last_prompts[prompt_idx]
                        if prompt in tokenizer.prompt_index:
                            info, cond = tokenizer.prompt_index[prompt]
                            if cond == "control":
                                h[b, :, :10] = 1.0
                            else:
                                h[b, :, 10:20] = 1.0
                        else:
                            h[b, :, :10] = 0.5
                    else:
                        h[b, :, :10] = 0.5

            for hook in list(layer.hooks):
                hook(layer, None, (h,))

        return MockOutput(batch_size, seq_len, device)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        out_ids = []
        for b in range(batch_size):
            row = input_ids[b]
            prompt_idx_token = row[0].item()
            prompt_idx = prompt_idx_token - 1000
            # Append prompt index as a generated token representation (2000 + idx)
            gen_token = 2000 + prompt_idx
            out_ids.append(
                torch.cat(
                    [
                        row,
                        torch.tensor(
                            [gen_token],
                            dtype=torch.long,
                            device=input_ids.device,
                        ),
                    ]
                )
            )
        return torch.stack(out_ids)

    def eval(self) -> MockModel:
        return self

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs) -> MockModel:
        n_layers = 26  # default fallback
        model_str = str(pretrained_model_name_or_path).lower()
        if "3-4b" in model_str or "gemma3_4b" in model_str:
            n_layers = 34
        elif "3-1b" in model_str or "gemma3_1b" in model_str:
            n_layers = 18
        elif "3-12b" in model_str or "gemma3_12b" in model_str:
            n_layers = 46
        elif "3-27b" in model_str or "gemma3_27b" in model_str:
            n_layers = 62
        return cls(n_layers=n_layers)


class MockOutput:
    def __init__(self, batch_size: int, seq_len: int, device: torch.device) -> None:
        self.logits = torch.randn(batch_size, seq_len, 32000, device=device)


class MockBatchEncoding(dict):
    def to(self, device) -> MockBatchEncoding:
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self[k] = v.to(device)
        return self


class MockTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.prompt_index: dict[str, tuple[dict, str]] = {}
        self.last_prompts: list[str] = []
        self._load_datasets()
        set_active_tokenizer(self)

    def _load_datasets(self) -> None:
        processed_dir = Path("data/processed")
        if not processed_dir.exists():
            return
        for pairs_file in processed_dir.glob("**/pairs.jsonl"):
            try:
                with open(pairs_file, encoding="utf-8") as f:
                    for line in f:
                        record = json.loads(line)
                        control = record["control"]
                        treatment = record["treatment"]
                        correct = record["correct_answer"]

                        meta = record.get("metadata", {})
                        incorrect = (
                            meta.get("biased_answer")
                            or meta.get("incorrect_opinion")
                            or meta.get("incorrect_answer")
                            or "incorrect"
                        )
                        if "is " in incorrect:
                            incorrect = incorrect.split("is ")[-1]

                        info = {
                            "dataset_type": record["dataset_type"],
                            "correct": correct,
                            "incorrect": incorrect,
                        }
                        self.prompt_index[control] = (info, "control")
                        self.prompt_index[treatment] = (info, "treatment")
            except Exception as exc:
                logger.debug("Failed loading mock datasets: %s", exc)

    def __call__(self, text, **kwargs) -> MockBatchEncoding:
        if isinstance(text, str):
            text = [text]
        self.last_prompts = list(text)

        input_ids = []
        attention_mask = []
        for i, _ in enumerate(text):
            input_ids.append([1000 + i, 1, 1, 1])
            attention_mask.append([1, 1, 1, 1])

        return MockBatchEncoding({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        })

    def decode(self, token_ids, **kwargs) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        prompt_idx = None
        for tid in token_ids:
            if tid >= 2000:
                prompt_idx = tid - 2000
                break

        if prompt_idx is None:
            for tid in token_ids:
                if 1000 <= tid < 2000:
                    prompt_idx = tid - 1000
                    break

        if prompt_idx is None or prompt_idx >= len(self.last_prompts):
            return "The answer is 42."

        prompt = self.last_prompts[prompt_idx]
        if prompt not in self.prompt_index:
            return "The answer is 42."

        info, cond = self.prompt_index[prompt]
        intervened = is_mock_intervention_active()

        correct = info["correct"]
        incorrect = info["incorrect"]

        if cond == "control":
            return f"The answer is {correct}."
        else:
            if intervened:
                return f"The answer is {correct}."
            else:
                if info["dataset_type"] == "cot_contradiction":
                    return "no"
                if info["dataset_type"] == "sycophancy_pressure":
                    return f"You're right, it is {incorrect}."
                return f"The answer is {incorrect}."

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> MockTokenizer:
        return cls()


class MockSAE(torch.nn.Module):
    def __init__(self, d_model: int = 128, n_features: int = 512) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.register_buffer("threshold", torch.zeros(n_features))
        self.W_enc = torch.nn.Parameter(torch.zeros(d_model, n_features))
        self.b_enc = torch.nn.Parameter(torch.zeros(n_features))
        self.W_dec = torch.nn.Parameter(torch.zeros(n_features, d_model))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x[:, -1, :]
        batch_size = x.shape[0]
        z = torch.zeros(batch_size, self.n_features, device=x.device)
        for b in range(batch_size):
            # Check hidden state values
            if x[b, 0] > 0.8:  # Control
                z[b, :5] = 1.5 + torch.randn(5, device=x.device) * 0.1
            elif x[b, 10] > 0.8:  # Treatment
                z[b, 10:15] = 1.5 + torch.randn(5, device=x.device) * 0.1
            else:
                z[b, :5] = 0.5 + torch.randn(5, device=x.device) * 0.1
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.randn(z.shape[0], self.d_model, device=z.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decode(z)


class MockTranscoder(torch.nn.Module):
    def __init__(
        self,
        source_layer: int,
        target_layer: int,
        n_source_features: int = 512,
        n_target_features: int = 512,
    ) -> None:
        super().__init__()
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.n_source_features = n_source_features
        self.n_target_features = n_target_features
        self.W = torch.nn.Parameter(torch.zeros(n_source_features, n_target_features))
        self.b = torch.nn.Parameter(torch.zeros(n_target_features))

    def forward(self, source_features: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            source_features.shape[0],
            self.n_target_features,
            device=source_features.device,
        )
