# scripts/collect_feature_matrices.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


@dataclass(frozen=True)
class FeatureKey:
    layer: int
    pos: int
    feature_idx: int

    @staticmethod
    def from_any(x: Any) -> "FeatureKey":
        # circuit-tracer Feature is often tuple-like (layer, pos, feature_idx)
        if hasattr(x, "layer") and hasattr(x, "pos") and hasattr(x, "feature_idx"):
            return FeatureKey(int(x.layer), int(x.pos), int(x.feature_idx))
        if isinstance(x, (tuple, list)) and len(x) == 3:
            return FeatureKey(int(x[0]), int(x[1]), int(x[2]))
        raise TypeError(f"Unsupported feature key: {type(x)} / {x!r}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=str, default="data/prompts_v0.jsonl")
    ap.add_argument("--outdir", type=str, default="outputs/step2")
    ap.add_argument("--transcoders", type=str, default="gemma")
    ap.add_argument("--base_model", type=str, default="google/gemma-2-2b")
    ap.add_argument("--it_model", type=str, default="google/gemma-2-2b-it")
    ap.add_argument("--layer", type=int, default=25, help="Which layer to slice")
    ap.add_argument("--pos_mode", type=str, default="last", choices=["last"], help="Token position selection")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_prompts", type=int, default=0, help="0 = all")
    return ap.parse_args()


def read_prompts(path: Path, max_prompts: int = 0) -> List[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_prompts and len(rows) >= max_prompts:
                break
    return rows


def load_replacement_model(model_name: str, transcoders: str, dtype: torch.dtype, device: torch.device) -> Any:
    try:
        from circuit_tracer import ReplacementModel  # type: ignore
    except Exception:
        from circuit_tracer.replacement_model import ReplacementModel  # type: ignore

    model = ReplacementModel.from_pretrained(model_name, transcoders, dtype=dtype)
    model.eval()
    try:
        model.to(device)
    except Exception:
        pass
    return model


def last_token_pos(model: Any, prompt: str) -> int:
    # Prefer circuit-tracer/ReplacementModel tokenization if available.
    # Some backends prepend a BOS/PAD token; using ensure_tokenized keeps positions aligned
    # with get_activations() cache indexing.
    if hasattr(model, "ensure_tokenized"):
        ids = model.ensure_tokenized(prompt)
        return int(ids.shape[0] - 1)

    # Fallback: tokenize without special tokens and treat the last position as (len).
    # This approximates the common behavior of prepending a single special token.
    ids = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    return int(ids.shape[0])


def slice_layer_pos_from_activations(
    acts: Any,
    *,
    layer: int,
    pos: int,
) -> Dict[int, float]:
    """Return mapping feature_idx -> activation for a specific (layer,pos) slice.

    Supports both:
    - dict-like activations keyed by (layer,pos,feature_idx)
    - circuit_tracer 0.1.0 tensor cache (sparse COO when sparse=True)
    """
    # Dict-based (older API)
    if isinstance(acts, dict):
        out: Dict[int, float] = {}
        for k, v in acts.items():
            fk = FeatureKey.from_any(k)
            if fk.layer == layer and fk.pos == pos:
                if isinstance(v, torch.Tensor):
                    v = v.detach().float().item()
                out[fk.feature_idx] = float(v)
        return out

    # Tensor cache (circuit_tracer 0.1.0 transformer-lens backend)
    if isinstance(acts, torch.Tensor):
        cache = acts

        # Expect sparse COO when sparse=True.
        if getattr(cache, "layout", None) != torch.sparse_coo:
            try:
                cache = cache.to_sparse_coo()
            except Exception as e:
                raise TypeError(
                    "Activation cache tensor is not sparse COO and could not be converted. "
                    f"layout={getattr(acts, 'layout', None)} shape={tuple(cache.shape)}"
                ) from e

        cache = cache.coalesce()
        idx = cache.indices()  # (ndim, nnz)
        vals = cache.values().detach().float()

        if idx.ndim != 2:
            raise TypeError(f"Unexpected sparse indices shape: {tuple(idx.shape)}")

        if idx.shape[0] == 3:
            layer_idx, pos_idx, feat_idx = idx
            mask = (layer_idx == int(layer)) & (pos_idx == int(pos))
            if not bool(mask.any()):
                return {}
            feat_sel = feat_idx[mask].tolist()
            val_sel = vals[mask].tolist()
            return {int(f): float(v) for f, v in zip(feat_sel, val_sel)}

        if idx.shape[0] == 2:
            # Fallback: (pos, feature) with implicit layer=0
            pos_idx, feat_idx = idx
            if layer != 0:
                return {}
            mask = pos_idx == int(pos)
            if not bool(mask.any()):
                return {}
            feat_sel = feat_idx[mask].tolist()
            val_sel = vals[mask].tolist()
            return {int(f): float(v) for f, v in zip(feat_sel, val_sel)}

        raise TypeError(f"Unexpected number of index dims: {idx.shape[0]}")

    raise TypeError(f"Unsupported activations type: {type(acts)}")


def get_layer_pos_slice(model: Any, prompt: str, *, layer: int, pos: int) -> Dict[int, float]:
    _ignored, acts = model.get_activations(prompt, sparse=True)
    return slice_layer_pos_from_activations(acts, layer=layer, pos=pos)


def build_dense_matrix(
    per_prompt: List[Dict[int, float]], feature_index: Dict[int, int], device: torch.device
) -> torch.Tensor:
    n = len(per_prompt)
    d = len(feature_index)
    M = torch.zeros((n, d), dtype=torch.float32, device=device)
    for i, row in enumerate(per_prompt):
        for feat_idx, val in row.items():
            j = feature_index[feat_idx]
            M[i, j] = val
    return M


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    prompts = read_prompts(Path(args.prompts), args.max_prompts)
    texts = [r["text"] for r in prompts]
    ids = [r.get("id", i) for i, r in enumerate(prompts)]
    print(f"Loaded {len(texts)} prompts from {args.prompts}")

    # Load base and IT models
    print(f"Loading BASE: {args.base_model}")
    base = load_replacement_model(args.base_model, args.transcoders, dtype=dtype, device=device)
    print(f"Loading IT:   {args.it_model}")
    it = load_replacement_model(args.it_model, args.transcoders, dtype=dtype, device=device)

    layer = args.layer

    base_rows: List[Dict[int, float]] = []
    it_rows: List[Dict[int, float]] = []
    feature_union: set[int] = set()

    # Collect per-prompt, sliced dicts
    for i, prompt in enumerate(texts):
        pos = last_token_pos(base, prompt)

        base_slice = get_layer_pos_slice(base, prompt, layer=layer, pos=pos)
        it_slice = get_layer_pos_slice(it, prompt, layer=layer, pos=pos)

        # If slice empty (rare), try backing up a couple tokens (keeps script robust)
        if len(base_slice) == 0 and pos >= 1:
            base_slice = get_layer_pos_slice(base, prompt, layer=layer, pos=pos - 1)
        if len(it_slice) == 0 and pos >= 1:
            it_slice = get_layer_pos_slice(it, prompt, layer=layer, pos=pos - 1)

        base_rows.append(base_slice)
        it_rows.append(it_slice)
        feature_union.update(base_slice.keys())
        feature_union.update(it_slice.keys())

        if (i + 1) % 5 == 0 or i == 0:
            print(f"[{i+1}/{len(texts)}] pos={pos} base_feats={len(base_slice)} it_feats={len(it_slice)}")

    # Build feature index (columns)
    feature_list = sorted(feature_union)
    feat_to_col = {f: j for j, f in enumerate(feature_list)}
    print(f"Union features in slice: {len(feature_list)} (layer={layer}, pos=last token)")

    # Build dense matrices on CPU for stability, then save
    # (If you prefer GPU, set device to cuda, but CPU is fine for 30–60 prompts.)
    cpu = torch.device("cpu")
    F_base = build_dense_matrix(base_rows, feat_to_col, device=cpu)
    F_it = build_dense_matrix(it_rows, feat_to_col, device=cpu)
    Delta = F_it - F_base

    payload = {
        "meta": {
            "base_model": args.base_model,
            "it_model": args.it_model,
            "transcoders": args.transcoders,
            "layer": layer,
            "pos_mode": args.pos_mode,
            "n_prompts": len(texts),
            "n_features": len(feature_list),
            "dtype": args.dtype,
        },
        "prompt_ids": ids,
        "prompts": texts,
        "feature_list": feature_list,  # column j corresponds to feature_list[j]
        "F_base": F_base,
        "F_it": F_it,
        "Delta": Delta,
    }

    out_path = outdir / "feature_matrices.pt"
    torch.save(payload, out_path)
    print(f"Saved matrices to {out_path.resolve()}")

if __name__ == "__main__":
    main()
