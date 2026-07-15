"""
Utilities for loading and interfacing with Gemma Scope 2 SAEs and transcoders.

Gemma Scope 2 provides JumpReLU SAEs trained on every layer of Gemma 3 IT models
(1B, 4B, 12B, 27B), with Matryoshka training for stable nested representations.
Transcoders (skip and cross-layer) are available for circuit tracing.

HuggingFace repo: google/gemma-scope-2
"""

from __future__ import annotations

import collections
import collections.abc
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import hf_hub_download, list_repo_files

logger = logging.getLogger(__name__)

HookPoint = Literal["resid_post", "mlp_out", "attn_out"]
TranscoderType = Literal["skip", "cross_layer"]

GEMMA_SCOPE_2_REPO = "google/gemma-scope-2"

# Map from model short-name to HuggingFace model ID
MODEL_HF_IDS = {
    "gemma3_1b": "google/gemma-3-1b-it",
    "gemma3_4b": "google/gemma-3-4b-it",
    "gemma3_12b": "google/gemma-3-12b-it",
    "gemma3_27b": "google/gemma-3-27b-it",
}

# Number of layers per model
N_LAYERS = {
    "gemma3_1b": 18,
    "gemma3_4b": 34,
    "gemma3_12b": 46,
    "gemma3_27b": 62,
}


@dataclass
class SAEConfig:
    model_name: str
    layer: int
    hook_point: HookPoint
    width_multiplier: int   # Expansion factor relative to d_model (e.g., 8, 16)
    repo_path: str          # Path within the HuggingFace repo


@dataclass
class TranscoderConfig:
    model_name: str
    source_layer: int
    target_layer: int       # Same as source_layer for skip-transcoders
    transcoder_type: TranscoderType
    repo_path: str


def build_sae_repo_path(
    model_name: str,
    layer: int,
    hook_point: HookPoint,
    width_multiplier: int,
) -> str:
    """
    Construct the file path within google/gemma-scope-2 for a given SAE.
    """
    # Map width: e.g., 16x or 16 -> "16k"
    width_str = f"{width_multiplier}k" if isinstance(width_multiplier, int) else str(width_multiplier).replace("x", "k")
    sparsity = "l0_big"
    return f"{hook_point}_all/layer_{layer}_width_{width_str}_{sparsity}/params.safetensors"


def build_transcoder_repo_path(
    model_name: str,
    source_layer: int,
    transcoder_type: TranscoderType,
    target_layer: int | None = None,
) -> str:
    width_str = "16k"
    sparsity = "l0_big"
    return f"transcoder_all/layer_{source_layer}_width_{width_str}_{sparsity}/params.safetensors"


def load_sae(
    model_name: str,
    layer: int,
    hook_point: HookPoint = "resid_post",
    width_multiplier: int = 16,
    cache_dir: str | Path | None = None,
    device: str = "cpu",
) -> "JumpReLUSAE":
    """
    Download and instantiate a Gemma Scope 2 SAE for the specified layer and hook point.

    Returns a JumpReLUSAE instance with loaded weights.
    """
    if os.environ.get("MOCK_PIPELINE") == "true":
        from src.utils.mock_utils import MockSAE
        return MockSAE()

    size = model_name.replace("gemma3_", "")
    repo_id = f"google/gemma-scope-2-{size}-it"
    repo_path = build_sae_repo_path(model_name, layer, hook_point, width_multiplier)

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=repo_path,
        cache_dir=cache_dir,
    )
    logger.info("Loaded SAE from %s", local_path)
    return JumpReLUSAE.from_file(local_path, device=device)


class LazySAEDict(collections.abc.Mapping):
    """
    A dictionary-like mapping that loads JumpReLU SAEs lazily on-demand.
    Uses an LRU cache to limit concurrent memory usage.
    """
    def __init__(
        self,
        model_name: str,
        hook_point: HookPoint,
        width_multiplier: int,
        cache_dir: str | Path | None,
        device: str,
        layer_range: tuple[int, int] | None = None,
        max_cache_size: int = 4,
    ):
        self.model_name = model_name
        self.hook_point = hook_point
        self.width_multiplier = width_multiplier
        self.cache_dir = cache_dir
        self.device = device
        self.max_cache_size = max_cache_size

        n_layers = N_LAYERS[model_name]
        start, end = layer_range if layer_range else (0, n_layers - 1)
        self._keys = list(range(start, end + 1))
        self._cache = collections.OrderedDict()

    def __getitem__(self, key: int) -> JumpReLUSAE:
        if key not in self._keys:
            raise KeyError(key)

        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        # Load SAE on CPU to conserve GPU VRAM
        sae = load_sae(
            model_name=self.model_name,
            layer=key,
            hook_point=self.hook_point,
            width_multiplier=self.width_multiplier,
            cache_dir=self.cache_dir,
            device="cpu",
        )

        self._cache[key] = sae

        if len(self._cache) > self.max_cache_size:
            old_key, old_sae = self._cache.popitem(last=False)
            del old_sae
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return sae

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def __contains__(self, key: int) -> bool:
        return key in self._keys

    def clear(self):
        self._cache.clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LazyTranscoderDict(collections.abc.Mapping):
    """
    A dictionary-like mapping that loads CrossLayerTranscoders lazily on-demand.
    Uses an LRU cache to limit concurrent memory usage.
    """
    def __init__(
        self,
        model_name: str,
        transcoder_type: TranscoderType,
        cache_dir: str | Path | None,
        device: str,
        n_layers: int,
        max_cache_size: int = 4,
    ):
        self.model_name = model_name
        self.transcoder_type = transcoder_type
        self.cache_dir = cache_dir
        self.device = device
        self.max_cache_size = max_cache_size

        # Keys are tuples of (source_layer, target_layer)
        self._keys = [(layer, layer + 1) for layer in range(n_layers - 1)]
        self._cache = collections.OrderedDict()

    def __getitem__(self, key: tuple[int, int]) -> CrossLayerTranscoder:
        if key not in self._keys:
            raise KeyError(key)

        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        src_layer, dest_layer = key
        try:
            tc = load_transcoder(
                model_name=self.model_name,
                source_layer=src_layer,
                transcoder_type=self.transcoder_type,
                target_layer=dest_layer,
                cache_dir=self.cache_dir,
                device="cpu",  # Load on CPU first
            )
        except Exception as exc:
            logger.warning("Could not load transcoder %d→%d: %s", src_layer, dest_layer, exc)
            raise KeyError(key) from exc

        self._cache[key] = tc

        if len(self._cache) > self.max_cache_size:
            old_key, old_tc = self._cache.popitem(last=False)
            del old_tc
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return tc

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self._keys

    def clear(self):
        self._cache.clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_all_layer_saes(
    model_name: str,
    hook_point: HookPoint = "resid_post",
    width_multiplier: int = 16,
    layer_range: tuple[int, int] | None = None,
    cache_dir: str | Path | None = None,
    device: str = "cpu",
) -> dict[int, "JumpReLUSAE"]:
    """
    Load SAEs for all (or a range of) layers in the model lazily.
    """
    return LazySAEDict(
        model_name=model_name,
        hook_point=hook_point,
        width_multiplier=width_multiplier,
        layer_range=layer_range,
        cache_dir=cache_dir,
        device=device,
    )



def load_transcoder(
    model_name: str,
    source_layer: int,
    transcoder_type: TranscoderType = "cross_layer",
    target_layer: int | None = None,
    cache_dir: str | Path | None = None,
    device: str = "cpu",
) -> "CrossLayerTranscoder":
    """
    Download and instantiate a Gemma Scope 2 transcoder.
    """
    if os.environ.get("MOCK_PIPELINE") == "true":
        from src.utils.mock_utils import MockTranscoder
        return MockTranscoder(source_layer, target_layer or (source_layer + 1))

    size = model_name.replace("gemma3_", "")
    repo_id = f"google/gemma-scope-2-{size}-it"
    repo_path = build_transcoder_repo_path(
        model_name, source_layer, transcoder_type, target_layer
    )
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=repo_path,
        cache_dir=cache_dir,
    )
    logger.info("Loaded transcoder from %s", local_path)
    return CrossLayerTranscoder.from_file(local_path, device=device)


class JumpReLUSAE(torch.nn.Module):
    """
    JumpReLU sparse autoencoder as used in Gemma Scope 2.
    """

    def __init__(
        self,
        d_model: int,
        n_features: int,
        threshold: torch.Tensor,
        W_enc: torch.Tensor,
        b_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_dec: torch.Tensor,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.register_buffer("threshold", threshold)
        self.W_enc = torch.nn.Parameter(W_enc)
        self.b_enc = torch.nn.Parameter(b_enc)
        self.W_dec = torch.nn.Parameter(W_dec)
        self.b_dec = torch.nn.Parameter(b_dec)

    @classmethod
    def from_file(cls, path: str | Path, device: str = "cpu") -> "JumpReLUSAE":
        path = Path(path)
        dtype = torch.bfloat16 if (torch.cuda.is_available() and device != "cpu") else torch.float32
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            tensors = load_file(path, device=device)
            return cls(
                d_model=tensors["w_enc"].shape[0],
                n_features=tensors["w_enc"].shape[1],
                threshold=tensors["threshold"].to(dtype=dtype),
                W_enc=tensors["w_enc"].to(dtype=dtype),
                b_enc=tensors["b_enc"].to(dtype=dtype),
                W_dec=tensors["w_dec"].to(dtype=dtype),
                b_dec=tensors["b_dec"].to(dtype=dtype),
            )
        else:
            import numpy as np
            data = np.load(path)
            return cls(
                d_model=data["W_enc"].shape[0],
                n_features=data["W_enc"].shape[1],
                threshold=torch.tensor(data["threshold"], device=device, dtype=dtype),
                W_enc=torch.tensor(data["W_enc"], device=device, dtype=dtype),
                b_enc=torch.tensor(data["b_enc"], device=device, dtype=dtype),
                W_dec=torch.tensor(data["W_dec"], device=device, dtype=dtype),
                b_dec=torch.tensor(data["b_dec"], device=device, dtype=dtype),
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.W_enc.device or x.dtype != self.W_enc.dtype:
            self.to(device=x.device, dtype=x.dtype)
        pre_act = x @ self.W_enc + self.b_enc
        # JumpReLU: zero out activations below per-feature threshold
        return torch.where(pre_act > self.threshold, pre_act, torch.zeros_like(pre_act))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.device != self.W_dec.device or z.dtype != self.W_dec.dtype:
            self.to(device=z.device, dtype=z.dtype)
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat

    @property
    def feature_directions(self) -> torch.Tensor:
        return self.W_dec


class CrossLayerTranscoder(torch.nn.Module):
    """
    Cross-layer transcoder for attribution graph construction.
    """

    def __init__(
        self,
        source_layer: int,
        target_layer: int,
        n_source_features: int,
        n_target_features: int,
        W: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        w_enc: torch.Tensor | None = None,
        w_dec: torch.Tensor | None = None,
        b_enc: torch.Tensor | None = None,
        b_dec: torch.Tensor | None = None,
        threshold: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.n_source_features = n_source_features
        self.n_target_features = n_target_features
        
        if W is not None:
            self.W = torch.nn.Parameter(W)
        else:
            self.W = None
            
        if b is not None:
            self.b = torch.nn.Parameter(b)
        else:
            self.b = None

        if w_enc is not None:
            self.w_enc = torch.nn.Parameter(w_enc)
            self.w_dec = torch.nn.Parameter(w_dec)
            self.b_enc = torch.nn.Parameter(b_enc)
            self.b_dec = torch.nn.Parameter(b_dec)
            self.threshold = torch.nn.Parameter(threshold)

    @classmethod
    def from_file(cls, path: str | Path, device: str = "cpu") -> "CrossLayerTranscoder":
        path = Path(path)
        dtype = torch.bfloat16 if (torch.cuda.is_available() and device != "cpu") else torch.float32
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            tensors = load_file(path, device=device)
            import re
            match = re.search(r"layer_(\d+)", str(path))
            source_layer = int(match.group(1)) if match else 0
            target_layer = source_layer + 1
            return cls(
                source_layer=source_layer,
                target_layer=target_layer,
                n_source_features=tensors["w_enc"].shape[0],
                n_target_features=tensors["w_enc"].shape[1],
                w_enc=tensors["w_enc"].to(dtype=dtype),
                w_dec=tensors["w_dec"].to(dtype=dtype),
                b_enc=tensors["b_enc"].to(dtype=dtype),
                b_dec=tensors["b_dec"].to(dtype=dtype),
                threshold=tensors["threshold"].to(dtype=dtype),
            )
        else:
            import numpy as np
            data = np.load(path)
            return cls(
                source_layer=int(data["source_layer"]),
                target_layer=int(data["target_layer"]),
                n_source_features=data["W"].shape[0],
                n_target_features=data["W"].shape[1],
                W=torch.tensor(data["W"], device=device, dtype=dtype),
                b=torch.tensor(data["b"], device=device, dtype=dtype),
            )

    def forward(self, source_features: torch.Tensor) -> torch.Tensor:
        if self.W is not None:
            if source_features.device != self.W.device or source_features.dtype != self.W.dtype:
                self.to(device=source_features.device, dtype=source_features.dtype)
            return source_features @ self.W + self.b
        # JumpReLU transcoder forward pass mapping hidden to hidden
        if source_features.device != self.w_enc.device or source_features.dtype != self.w_enc.dtype:
            self.to(device=source_features.device, dtype=source_features.dtype)
        pre_act = source_features @ self.w_enc + self.b_enc
        z = torch.where(pre_act > self.threshold, pre_act, torch.zeros_like(pre_act))
        return z @ self.w_dec + self.b_dec
