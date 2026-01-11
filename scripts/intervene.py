# scripts/intervene.py
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=str, default="outputs/step4/case.json")
    ap.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    ap.add_argument("--transcoders", type=str, default="gemma")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--k_features", type=int, default=3)
    return ap.parse_args()


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


def import_feature_intervention_generate() -> Callable[..., Any]:
    candidates = [
        ("circuit_tracer", "feature_intervention_generate"),
        ("circuit_tracer.intervention", "feature_intervention_generate"),
        ("circuit_tracer.interventions", "feature_intervention_generate"),
        ("circuit_tracer.feature_intervention", "feature_intervention_generate"),
    ]
    for mod, name in candidates:
        try:
            m = __import__(mod, fromlist=[name])
            fn = getattr(m, name)
            return fn
        except Exception:
            continue
    raise ImportError(
        "Could not import feature_intervention_generate from circuit_tracer. "
        "Check the intervention_demo.ipynb in the circuit-tracer repo for the correct import path."
    )


def call_with_supported_kwargs(fn: Callable[..., Any], **kwargs):
    sig = inspect.signature(fn)
    supported = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    return fn(**filtered)


def prompt_last_pos(model: Any, prompt: str) -> int:
    # Prefer circuit-tracer's tokenization (it may prepend a BOS/PAD token).
    if hasattr(model, "ensure_tokenized"):
        ids = model.ensure_tokenized(prompt)
        return int(ids.shape[0] - 1)
    ids = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    # Many transformer models effectively have an implicit BOS when analyzing activations.
    return int(ids.shape[0])


def extract_text_from_generation_result(result: Any) -> str:
    # circuit_tracer 0.1.0 returns (generated_text, logits_cache, activations_cache_or_None)
    if isinstance(result, tuple) and len(result) >= 1 and isinstance(result[0], str):
        return result[0]
    if isinstance(result, str):
        return result
    return repr(result)


def run_generation(
    model: Any,
    *,
    prompt: str,
    interventions: Sequence[tuple[Any, Any, Any, Any]],
    max_new_tokens: int,
    do_sample: bool,
) -> str:
    # Newer circuit-tracer: method on the model
    if hasattr(model, "feature_intervention_generate"):
        result = call_with_supported_kwargs(
            model.feature_intervention_generate,
            inputs=prompt,
            interventions=interventions,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            # Required by circuit_tracer 0.1.0 (asserts True)
            use_past_kv_cache=True,
            # We only need the generated text for this script
            return_activations=False,
        )
        return extract_text_from_generation_result(result)

    # Older circuit-tracer demos: top-level helper function
    fn = import_feature_intervention_generate()
    result = call_with_supported_kwargs(
        fn,
        model=model,
        prompt=prompt,
        intervention_tuples=interventions,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        use_past_kv_cache=True,
    )
    return extract_text_from_generation_result(result)


def main():
    args = parse_args()
    case = json.loads(Path(args.case).read_text())
    prompt = case["prompt"]
    layer = int(case["layer"])
    top_feats = case["top_sparse_features"][: args.k_features]
    feat_idxs = [int(x["feature_idx"]) for x in top_feats]

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    model = load_replacement_model(args.model, args.transcoders, dtype=dtype, device=device)

    pos = prompt_last_pos(model, prompt)
    open_ended = slice(pos, None)

    # intervention tuples: (layer, position_slice, feature_idx, value)
    clamp_to_zero = [(layer, open_ended, fidx, 0.0) for fidx in feat_idxs]

    print("\n=== CASE ===")
    print(prompt)
    print(f"layer={layer}, last_prompt_pos={pos}, features={feat_idxs}")

    print("\n=== BASELINE (no intervention) ===")
    baseline_text = run_generation(
        model,
        prompt=prompt,
        interventions=[],
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    print(baseline_text)

    print("\n=== INTERVENED (clamp features to 0.0) ===")
    intervened_text = run_generation(
        model,
        prompt=prompt,
        interventions=clamp_to_zero,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    print(intervened_text)


if __name__ == "__main__":
    main()
