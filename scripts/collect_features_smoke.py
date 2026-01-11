# test_install.py
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from utils import (
    compute_delta,
    get_sparse_feature_activations,
    load_replacement_model,
    pick_device,
    pick_dtype,
    save_json,
    save_jsonl,
    sparse_dict_to_serializable,
    topk_by_abs,
    topk_next_token,
)


DEFAULT_PROMPTS = [
    "2 + 2 =",
    "Write a short haiku about rain.",
    "Translate to French: 'The cat sits on the mat.'",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="outputs/step1")
    p.add_argument("--transcoders", type=str, default="gemma", help="e.g. 'gemma' preset or a HF repo like 'mntss/gemma-scope-transcoders'")
    p.add_argument("--base_model", type=str, default="google/gemma-2-2b")
    p.add_argument("--it_model", type=str, default="google/gemma-2-2b-it")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default=None, help="e.g. bfloat16, float16, float32")
    p.add_argument(
        "--backend",
        type=str,
        default="transformerlens",
        help="ReplacementModel backend: 'transformerlens' or 'nnsight'",
    )
    p.add_argument("--k_tokens", type=int, default=10)
    p.add_argument("--k_delta", type=int, default=30)
    p.add_argument("--prompts_file", type=str, default=None, help="Optional text file: one prompt per line")
    return p.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts_file:
        lines = Path(args.prompts_file).read_text().splitlines()
        prompts = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        return prompts
    return DEFAULT_PROMPTS


def free_model(model) -> None:
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    prompts = load_prompts(args)

    run_meta = {
        "device": str(device),
        "dtype": str(dtype),
        "transcoders": args.transcoders,
        "base_model": args.base_model,
        "it_model": args.it_model,
        "backend": args.backend,
        "n_prompts": len(prompts),
    }
    save_json(outdir / "run_meta.json", run_meta)

    # ---- Pass 1: Base model ----
    print(f"\n[1/2] Loading BASE model: {args.base_model}")
    base = load_replacement_model(
        model_name=args.base_model,
        transcoders=args.transcoders,
        device=device,
        dtype=dtype,
        backend=args.backend,
    )

    base_rows = []
    base_sparse_by_prompt = []

    for i, prompt in enumerate(prompts):
        print(f"\n[BASE] Prompt {i}: {prompt!r}")
        top = topk_next_token(base, prompt, k=args.k_tokens)
        print("  Top next-token preds:")
        for tok, p in top:
            print(f"   - {tok!r}: {p:.4f}")

        acts = get_sparse_feature_activations(base, prompt)
        base_sparse_by_prompt.append(acts)
        row = {
            "i": i,
            "prompt": prompt,
            "top_next_tokens": top,
            "n_active_features": len(acts),
            "sparse_acts": sparse_dict_to_serializable(acts),
        }
        base_rows.append(row)

    save_jsonl(outdir / "base.jsonl", base_rows)
    free_model(base)

    # ---- Pass 2: Instruction-tuned model ----
    print(f"\n[2/2] Loading IT model: {args.it_model}")
    it = load_replacement_model(
        model_name=args.it_model,
        transcoders=args.transcoders,
        device=device,
        dtype=dtype,
        backend=args.backend,
    )

    it_rows = []
    delta_rows = []

    for i, prompt in enumerate(prompts):
        print(f"\n[IT] Prompt {i}: {prompt!r}")
        top = topk_next_token(it, prompt, k=args.k_tokens)
        print("  Top next-token preds:")
        for tok, p in top:
            print(f"   - {tok!r}: {p:.4f}")

        it_acts = get_sparse_feature_activations(it, prompt)
        base_acts = base_sparse_by_prompt[i]

        delta = compute_delta(it_acts, base_acts)
        top_delta = topk_by_abs(delta, k=args.k_delta)

        print(f"  Active features (IT): {len(it_acts)} | (BASE): {len(base_acts)} | union: {len(delta)}")
        print("  Top |delta| features (FeatureKey -> delta):")
        for fk, dv in top_delta[: min(10, len(top_delta))]:
            print(f"   - {fk.to_str():>18} -> {dv:+.4f}")

        it_rows.append(
            {
                "i": i,
                "prompt": prompt,
                "top_next_tokens": top,
                "n_active_features": len(it_acts),
                "sparse_acts": sparse_dict_to_serializable(it_acts),
            }
        )
        delta_rows.append(
            {
                "i": i,
                "prompt": prompt,
                "n_union_features": len(delta),
                "top_abs_delta": [(fk.to_str(), dv) for fk, dv in top_delta],
            }
        )

    save_jsonl(outdir / "it.jsonl", it_rows)
    save_jsonl(outdir / "delta_topk.jsonl", delta_rows)
    free_model(it)

    print(f"\nDone. Wrote outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
