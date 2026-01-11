# scripts/step4_trace_cli.py
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=str, default="outputs/step4/case.json")
    ap.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    ap.add_argument("--transcoder_set", type=str, default="gemma")
    ap.add_argument("--outdir", type=str, default="outputs/step4/graphs")
    ap.add_argument("--slug", type=str, default=None)
    ap.add_argument("--server", action="store_true")
    ap.add_argument("--port", type=int, default=8041)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--max_n_logits", type=int, default=10)
    ap.add_argument("--desired_logit_prob", type=float, default=0.95)
    ap.add_argument("--node_threshold", type=float, default=0.8)
    ap.add_argument("--edge_threshold", type=float, default=0.98)
    return ap.parse_args()


def main():
    args = parse_args()
    case = json.loads(Path(args.case).read_text())
    prompt = case["prompt"]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    slug = args.slug or f"anomaly_prompt_{case['prompt_index']}"
    graph_pt = outdir / f"{slug}.pt"
    graph_files = outdir / f"{slug}_graph_files"

    cmd = [
        "circuit-tracer",
        "attribute",
        "--model", args.model,
        "--transcoder_set", args.transcoder_set,
        "--prompt", prompt,
        "--graph_output_path", str(graph_pt),
        "--slug", slug,
        "--graph_file_dir", str(graph_files),
        "--dtype", args.dtype,
        "--max_n_logits", str(args.max_n_logits),
        "--desired_logit_prob", str(args.desired_logit_prob),
        "--node_threshold", str(args.node_threshold),
        "--edge_threshold", str(args.edge_threshold),
    ]
    if args.server:
        cmd += ["--server", "--port", str(args.port)]

    print("Running:\n", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)
    print(f"\nSaved raw graph: {graph_pt.resolve()}")
    print(f"Saved graph files: {graph_files.resolve()}")
    if args.server:
        print(f"Server should be on port {args.port} (use SSH port forwarding if remote).")


if __name__ == "__main__":
    main()
