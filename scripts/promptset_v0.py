# scripts/promptset_v0.py
from __future__ import annotations

import argparse
import json
from pathlib import Path


SAFE_PROMPTS = [
    # Format-following
    "Answer with exactly 5 words: What is the capital of France?",
    'Return JSON only with keys {"answer": ...}: What is 7*8?',
    "Give a 3-bullet list of benefits of sleep.",
    "Write one sentence, no commas: Describe a sunset.",
    "Respond in YAML with keys: summary, steps. Topic: making tea.",

    # Style constraints
    "Be extremely concise: explain what a mutex is.",
    "Be friendly and encouraging: give study tips for an exam.",
    "Be formal: write a short note thanking someone for their help.",
    "Explain like I'm 10: what is electricity?",
    "Use a witty tone: explain why backups matter.",

    # Instruction-y wrappers vs plain query
    "You are a helpful assistant. Provide a short answer: What is photosynthesis?",
    "Please answer clearly and step-by-step: How do you compute an average?",
    "Give a structured plan with numbered steps: learn basic Python in a week.",
    "Answer as a brief checklist: preparing for a job interview.",
    "Give two options and a recommendation: choosing a laptop for university.",

    # Slightly conflicting (benign) constraints
    "Answer in exactly one sentence, and also give a 2-item bullet list: What is gravity?",
    "Provide JSON only, but also include a short explanation: What is recursion?",
    "Give exactly 3 words, and also be detailed: Define 'entropy'.",

    # “assistant-y” meta prompts (still safe)
    "As an AI assistant, explain what you can do in one paragraph.",
    "Write a short disclaimer then answer: What is machine learning?",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/prompts_v0.jsonl")
    ap.add_argument(
        "--n",
        type=int,
        default=40,
        help="Target count (will truncate or repeat templates)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    prompts = SAFE_PROMPTS.copy()
    # If user requests more than we have, just cycle (fine for quick prototype)
    while len(prompts) < args.n:
        prompts.extend(SAFE_PROMPTS)
    prompts = prompts[: args.n]

    with out.open("w") as f:
        for i, p in enumerate(prompts):
            f.write(json.dumps({"id": i, "text": p}) + "\n")

    print(f"Wrote {len(prompts)} prompts to {out.resolve()}")


if __name__ == "__main__":
    main()
