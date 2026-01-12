from __future__ import annotations

import argparse
import csv
import json
import inspect
from pathlib import Path
from typing import Any, Callable, List, Sequence

import torch


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


def call_with_supported_kwargs(fn: Callable[..., Any], **kwargs):
    sig = inspect.signature(fn)
    # If the callable accepts **kwargs, don't filter (generation kwargs are forwarded via **kwargs).
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)

    supported = set(sig.parameters.keys())
    return fn(**{k: v for k, v in kwargs.items() if k in supported})


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
    # circuit_tracer 0.1.0: method on ReplacementModel (TransformerLens backend)
    if hasattr(model, "feature_intervention_generate"):
        result = call_with_supported_kwargs(
            model.feature_intervention_generate,
            inputs=prompt,
            interventions=interventions,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_past_kv_cache=True,
            return_activations=False,
        )
        return extract_text_from_generation_result(result)

    # Older circuit-tracer demos: top-level helper function
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
        except Exception:
            continue

    raise ImportError(
        "Could not find feature intervention generation API in circuit_tracer. "
        "Expected model.feature_intervention_generate(...) on newer versions."
    )


def soft_contains_disclaimer(text: str) -> int:
    t = text.lower()
    return int(("disclaimer" in t) or ("## disclaimer" in t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", type=str, default="outputs/step2_300/feature_matrices.pt")
    ap.add_argument("--decomp", type=str, default="outputs/step3_300/decomp.pt")
    ap.add_argument("--out", type=str, default="outputs/step4/batch_results.csv")
    ap.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    ap.add_argument("--transcoders", type=str, default="gemma")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--topk_prompts", type=int, default=15)
    ap.add_argument("--k_features", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--min_abs_s", type=float, default=0.0, help="Skip prompts where top features are all ~0")
    args = ap.parse_args()

    fm = torch.load(args.fm, map_location="cpu")
    decomp = torch.load(args.decomp, map_location="cpu")

    prompts: List[str] = fm["prompts"]
    feature_list: List[int] = fm["feature_list"]
    layer = int(fm["meta"]["layer"])
    S: torch.Tensor = decomp["S"]
    scores: torch.Tensor = decomp["scores"]

    order = torch.argsort(scores, descending=True).tolist()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    model = load_replacement_model(args.model, args.transcoders, dtype=dtype, device=device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "prompt_index", "score_L1", "top_features",
            "baseline_has_disclaimer", "intervened_has_disclaimer",
            "baseline_preview", "intervened_preview"
        ])

        n_done = 0
        seen = set()

        for rank, idx in enumerate(order, start=1):
            prompt = prompts[idx]
            if prompt in seen:
                continue
            seen.add(prompt)

            row = S[idx]
            abs_row = torch.abs(row)
            cols = torch.argsort(abs_row, descending=True).tolist()

            chosen = []
            for j in cols:
                if abs_row[j].item() == 0:
                    break
                chosen.append((feature_list[j], float(row[j].item())))
                if len(chosen) >= args.k_features:
                    break

            if not chosen or max(abs(v) for _, v in chosen) < args.min_abs_s:
                continue

            pos = prompt_last_pos(model, prompt)
            itv = [(layer, slice(pos, None), int(feat), 0.0) for feat, _ in chosen]

            btxt = run_generation(
                model,
                prompt=prompt,
                interventions=[],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            itxt = run_generation(
                model,
                prompt=prompt,
                interventions=itv,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

            w.writerow([
                n_done + 1, idx, float(scores[idx].item()),
                json.dumps(chosen),
                soft_contains_disclaimer(btxt),
                soft_contains_disclaimer(itxt),
                btxt[:160].replace("\n", "\\n"),
                itxt[:160].replace("\n", "\\n"),
            ])

            n_done += 1
            if n_done >= args.topk_prompts:
                break

    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
