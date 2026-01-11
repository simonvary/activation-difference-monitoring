# utils.py
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch

@dataclass(frozen=True)
class FeatureKey:
    layer: int
    pos: int
    feature_idx: int

    def to_str(self) -> str:
        return f"L{self.layer}:P{self.pos}:F{self.feature_idx}"

    @staticmethod
    def from_any(x: Any) -> "FeatureKey":
        """
        circuit-tracer demos treat Feature as a tuple (layer, pos, feature_idx).
        Sometimes it's a namedtuple / dataclass-ish object with attrs.
        """
        if hasattr(x, "layer") and hasattr(x, "pos") and hasattr(x, "feature_idx"):
            return FeatureKey(int(x.layer), int(x.pos), int(x.feature_idx))
        if isinstance(x, (tuple, list)) and len(x) == 3:
            return FeatureKey(int(x[0]), int(x[1]), int(x[2]))
        raise TypeError(f"Unsupported Feature key type: {type(x)} / {x!r}")


def pick_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(dtype: str | None, device: torch.device) -> torch.dtype:
    if dtype:
        return getattr(torch, dtype)
    # Good defaults: bf16 on CUDA if available; otherwise fp16 on CUDA; else fp32
    if device.type == "cuda":
        # bf16 is typically best if supported
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def filter_supported_kwargs(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    supported = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in supported}


def load_replacement_model(
    model_name: str,
    transcoders: str,
    device: torch.device,
    dtype: torch.dtype,
    backend: str | None = None,
) -> Any:
    # circuit-tracer exports ReplacementModel at top-level in many installs
    try:
        from circuit_tracer import ReplacementModel  # type: ignore
    except Exception:
        from circuit_tracer.replacement_model import ReplacementModel  # type: ignore

    if backend is None:
        backend = "transformerlens"

    kwargs = {
        "dtype": dtype,
        "backend": backend,
    }
    kwargs = filter_supported_kwargs(ReplacementModel.from_pretrained, kwargs)

    model = ReplacementModel.from_pretrained(model_name, transcoders, **kwargs)
    model.eval()

    # Best-effort move to device (ReplacementModel usually handles this,
    # but we keep it robust across backends)
    try:
        model.to(device)
    except Exception:
        pass

    return model


def topk_next_token(model: Any, prompt: str, k: int = 10) -> list[tuple[str, float]]:
    """
    Returns [(token_str, prob), ...] for the next token after the prompt.
    """
    with torch.inference_mode():
        logits = model(prompt)

    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"Expected logits tensor, got: {type(logits)}")

    if logits.ndim == 2:
        # (seq, vocab)
        next_logits = logits[-1]
    elif logits.ndim == 3:
        # (batch, seq, vocab)
        next_logits = logits[0, -1]
    else:
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

    probs = torch.softmax(next_logits.float(), dim=-1)
    top = torch.topk(probs, k=min(k, probs.shape[-1]))
    ids = top.indices.tolist()
    vals = top.values.tolist()

    tok = model.tokenizer
    tokens = []
    for tid, p in zip(ids, vals):
        # convert_ids_to_tokens gives raw token pieces; decode gives nicer strings for many tokenizers
        try:
            s = tok.decode([tid])
        except Exception:
            s = str(tok.convert_ids_to_tokens(tid))
        tokens.append((s, float(p)))
    return tokens


def get_sparse_feature_activations(model: Any, prompt: str) -> Dict[FeatureKey, float]:
    """
    Uses circuit-tracer's ReplacementModel.get_activations(prompt, sparse=True),
    which demos rely on. Returns a dict keyed by (layer,pos,feature_idx).
    """
    out = model.get_activations(prompt, sparse=True)
    if not (isinstance(out, tuple) and len(out) == 2):
        raise TypeError(f"Expected (something, activations_dict), got: {type(out)} / {out!r}")

    _ignored, acts = out

    # Older / alternative circuit-tracer versions may return a dict.
    if isinstance(acts, dict):
        sparse: Dict[FeatureKey, float] = {}
        for k, v in acts.items():
            fk = FeatureKey.from_any(k)
            if isinstance(v, torch.Tensor):
                v = v.detach().float().item()
            sparse[fk] = float(v)
        return sparse

    # circuit-tracer 0.1.0 (transformer-lens backend) returns an activation cache tensor.
    # With sparse=True this is typically a sparse COO tensor with shape (layers, pos, feature).
    if isinstance(acts, torch.Tensor):
        cache = acts

        # Normalize to sparse COO if possible.
        try:
            if getattr(cache, "layout", None) != torch.sparse_coo and getattr(cache, "is_sparse", False) is False:
                # Dense tensor; avoid iterating huge tensors.
                raise TypeError(
                    "Activation cache was returned as a dense tensor; expected a sparse tensor when sparse=True. "
                    f"Got shape={tuple(cache.shape)} dtype={cache.dtype}."
                )
            if getattr(cache, "layout", None) != torch.sparse_coo:
                cache = cache.to_sparse_coo()
        except Exception:
            # If conversion isn't available, we still try to proceed if it's already sparse.
            pass

        if getattr(cache, "is_sparse", False) and getattr(cache, "layout", None) == torch.sparse_coo:
            cache = cache.coalesce()
            idx = cache.indices()
            vals = cache.values()

            if idx.ndim != 2:
                raise TypeError(f"Unexpected sparse indices shape: {tuple(idx.shape)}")

            if idx.shape[0] == 3:
                layer_idx, pos_idx, feat_idx = idx
            elif idx.shape[0] == 2:
                # Fallback: (pos, feature) with a single implicit layer.
                pos_idx, feat_idx = idx
                layer_idx = torch.zeros_like(pos_idx)
            else:
                raise TypeError(f"Unexpected number of index dims: {idx.shape[0]}")

            # Build sparse dict.
            sparse: Dict[FeatureKey, float] = {}
            for l, p, f, v in zip(
                layer_idx.tolist(),
                pos_idx.tolist(),
                feat_idx.tolist(),
                vals.detach().float().tolist(),
            ):
                sparse[FeatureKey(int(l), int(p), int(f))] = float(v)
            return sparse

        raise TypeError(
            "Unsupported activation cache tensor layout. "
            f"Got layout={getattr(cache, 'layout', None)} shape={tuple(cache.shape)}"
        )

    raise TypeError(f"Expected activations dict or tensor, got: {type(acts)}")


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def save_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def sparse_dict_to_serializable(d: Dict[FeatureKey, float]) -> Dict[str, float]:
    return {k.to_str(): v for k, v in d.items()}


def compute_delta(
    it_acts: Dict[FeatureKey, float],
    base_acts: Dict[FeatureKey, float],
) -> Dict[FeatureKey, float]:
    keys = set(it_acts.keys()) | set(base_acts.keys())
    return {k: it_acts.get(k, 0.0) - base_acts.get(k, 0.0) for k in keys}


def topk_by_abs(
    d: Dict[FeatureKey, float],
    k: int = 20,
) -> list[tuple[FeatureKey, float]]:
    return sorted(d.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
