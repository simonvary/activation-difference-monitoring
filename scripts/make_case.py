# scripts/make_case.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_dir", type=str, default="outputs")
    ap.add_argument("--k_features", type=int, default=3)
    ap.add_argument("--out", type=str, default="outputs/step4/case.json")
    return ap.parse_args()


def main():
    args = parse_args()
    outputs = Path(args.outputs_dir)

    fm = torch.load(outputs / "step2" / "feature_matrices.pt", map_location="cpu")
    decomp = torch.load(outputs / "step3" / "decomp.pt", map_location="cpu")

    prompts = fm["prompts"]
    feature_list = fm["feature_list"]
    layer = int(fm["meta"]["layer"])
    scores = decomp["scores"]
    S = decomp["S"]

    # pick top UNIQUE prompt by text
    order = torch.argsort(scores, descending=True).tolist()
    seen = set()
    picked = None
    for idx in order:
        txt = prompts[idx]
        if txt not in seen:
            picked = idx
            break
        seen.add(txt)

    if picked is None:
        raise RuntimeError("Could not pick a prompt (unexpected).")

    row = S[picked]
    abs_row = torch.abs(row)
    feat_cols = torch.argsort(abs_row, descending=True).tolist()

    chosen = []
    for j in feat_cols:
        if abs_row[j].item() == 0:
            break
        chosen.append(
            {
                "feature_idx": int(feature_list[j]),
                "s_value": float(row[j].item()),
            }
        )
        if len(chosen) >= args.k_features:
            break

    case = {
        "prompt_index": int(picked),
        "prompt": prompts[picked],
        "layer": layer,
        "top_sparse_features": chosen,
        "score_L1": float(scores[picked].item()),
        "note": "pos is computed at runtime as last prompt token index; intervention uses slice(pos, None).",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(case, indent=2))
    print(f"Wrote {out.resolve()}")
    print(json.dumps(case, indent=2))


if __name__ == "__main__":
    main()
