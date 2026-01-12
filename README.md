# Model-divergence activation anomaly detection (Gemma-2-2B) via low-rank + sparse feature decomposition

This repo is a compact prototype of **activation anomaly detection** built on **interpretable latent features** (transcoder/SAE-like). The core idea is to compare a **base** model vs an **instruction-tuned** model on the *same prompts*, compute **feature-activation deltas** Δ, then decompose Δ into:

- **Low-rank** structure: shared, global “instruction-tuning drift”
- **Sparse** structure: prompt-specific “something unusual happened” residuals (anomaly signal)

Finally, the repo demonstrates **causal validation** by intervening directly on the implicated transcoder features and measuring changes in both **text** and **logit-space** diagnostics.

> Models used: `google/gemma-2-2b` vs `google/gemma-2-2b-it` (open weights)  
> Feature slice used in the notebook/pipeline: **layer 25** (configurable)

---

## Why this project (and why it’s relevant to monitoring)

Many practical safety/robustness problems are about **detecting distribution shift / unusual internal computation**, not interpretability “for its own sake.” This prototype treats “weird prompts” as **outliers in model-difference activation space** and uses a structured decomposition to separate:
- common, systematic differences (low-rank), from
- rare, prompt-specific differences (sparse).

The sparse component provides:
1) an **anomaly score** per prompt (‖Sᵢ‖₁) and  
2) a **short list of interpretable features** (the support of Sᵢ) to trace / intervene on.

---

## TL;DR: What to look at first

- 📓 **Notebook:** `notebooks/Transcoder_Anomaly_Detection_FellowshipReady.ipynb`  
  End-to-end methodology + single-prompt and multi-prompt intervention evaluation.
- 🌐 **Qualitative report:** `report_top10.html`  
  Side-by-side outputs for Baseline vs Low-rank-removed vs Sparse-removed on top prompts.
- 🧰 **Scripts:** `scripts/`  
  A reproducible pipeline: prompt set → feature matrices → decomposition → interventions.

---

## Repository layout


```
notebooks/
Transcoder_Anomaly_Detection_FellowshipReady.ipynb

scripts/
promptset_v0.py # small safe prompt generator (JSONL)
collect_feature_matrices.py # collect feature activations for base + IT, build Δ
lowrank_sparse_anomaly.py # decompose Δ into low-rank + sparse + rank/τ sweeps
make_case.py # select top anomaly prompt + top sparse features
intervene.py # single-prompt intervention demo
batch_intervene_topk.py # multi-prompt quantitative comparison -> CSV
trace_cli.py # (optional) wrapper for circuit-tracing graphs
utils.py

outputs/
step2*/feature_matrices.pt # cached matrices (F_base, F_it, Δ, feature list, prompts, meta)
step3*/decomp.pt # decomposition (L, S, scores, thresholds, etc.)
step4*/... # cases, CSVs, graphs
```

---

## Setup

### Environment
- Python 3.10+ recommended
- One GPU strongly recommended (tested on CUDA); CPU may be slow.

### Install dependencies
You need:
- `torch`, `numpy`, `pandas`, `tqdm`, `matplotlib`
- **`circuit-tracer==0.1.0`** (this repo assumes circuit-tracer v0.1.0)
- HuggingFace model access (Gemma weights)

Example:
```bash
pip install torch numpy pandas tqdm matplotlib
pip install circuit-tracer==0.1.0
```