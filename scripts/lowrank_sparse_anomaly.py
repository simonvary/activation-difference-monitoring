# scripts/lowrank_sparse_anomaly.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inpt", type=str, default="outputs/step2/feature_matrices.pt")
    ap.add_argument("--outdir", type=str, default="outputs/step3")
    ap.add_argument("--rank", type=int, default=10, help="Low-rank dimension r")
    ap.add_argument("--tau_percentile", type=float, default=99.0, help="Percentile for |R| threshold (e.g., 95–99.5)")
    ap.add_argument("--topk_prompts", type=int, default=10)
    ap.add_argument("--topk_features", type=int, default=20)
    return ap.parse_args()


def pca_lowrank_recon(X: torch.Tensor, r: int) -> torch.Tensor:
    """
    Uses torch.pca_lowrank to get a rank-r approximation.
    X: [n, d] float32 on CPU is fine for n~60, d~few thousand.
    """
    # center rows? For Delta you can optionally center columns; keep simple first:
    # Xc = X - X.mean(dim=0, keepdim=True)
    Xc = X
    q = min(r + 5, min(Xc.shape) - 1) if min(Xc.shape) > 1 else r
    U, S, V = torch.pca_lowrank(Xc, q=q, center=False)
    # Take first r
    Ur = U[:, :r]
    Sr = S[:r]
    Vr = V[:, :r]
    L = (Ur * Sr) @ Vr.T
    return L


def soft_threshold(R: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.sign(R) * torch.clamp(torch.abs(R) - tau, min=0.0)


def percentile_tau(abs_vals: torch.Tensor, pct: float) -> float:
    k = int((pct / 100.0) * (abs_vals.numel() - 1))
    flat = abs_vals.flatten()
    # kthvalue is 1-indexed
    v = torch.kthvalue(flat, k + 1).values.item()
    return float(v)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.inpt, map_location="cpu")
    Delta: torch.Tensor = payload["Delta"].float()
    prompts = payload["prompts"]
    prompt_ids = payload["prompt_ids"]
    feature_list = payload["feature_list"]

    print(f"Loaded Delta with shape {tuple(Delta.shape)}")

    # Low-rank
    r = min(args.rank, min(Delta.shape) - 1) if min(Delta.shape) > 1 else 1
    L = pca_lowrank_recon(Delta, r=r)
    R = Delta - L

    # Sparse proxy
    tau = percentile_tau(torch.abs(R), args.tau_percentile)
    S = soft_threshold(R, tau=tau)

    # Prompt anomaly scores
    scores = torch.sum(torch.abs(S), dim=1)  # L1 norm per prompt
    top_idx = torch.argsort(scores, descending=True)[: args.topk_prompts].tolist()

    # Save prompt ranking
    out_prompts = outdir / "top_prompts.csv"
    with out_prompts.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "prompt_id", "score_L1", "prompt"])
        for rank, i in enumerate(top_idx, start=1):
            w.writerow([rank, prompt_ids[i], float(scores[i].item()), prompts[i]])

    # Save top features for each top prompt
    out_details = outdir / "top_prompt_features.txt"
    with out_details.open("w") as f:
        for rank, i in enumerate(top_idx, start=1):
            row = S[i]
            vals = torch.abs(row)
            feat_idx = torch.argsort(vals, descending=True)[: args.topk_features].tolist()
            f.write(f"\n=== Rank {rank} | prompt_id={prompt_ids[i]} | score={float(scores[i]):.4f} ===\n")
            f.write(prompts[i] + "\n")
            for j in feat_idx:
                if vals[j].item() == 0:
                    break
                feat = feature_list[j]
                f.write(f"  feature {feat:>6}  S={float(row[j]):+.4f}\n")

    # Save tensors for later (optional)
    torch.save(
        {"L": L, "R": R, "S": S, "scores": scores, "tau": tau, "rank": r},
        outdir / "decomp.pt",
    )

    print(f"tau={tau:.6f}, rank={r}")
    print(f"Wrote: {out_prompts.resolve()}")
    print(f"Wrote: {out_details.resolve()}")
    print(f"Wrote: {(outdir / 'decomp.pt').resolve()}")

if __name__ == "__main__":
    main()
