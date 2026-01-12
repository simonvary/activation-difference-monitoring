# Model-divergence activation anomaly detection via low-rank + sparse feature decomposition

This repository contains a compact proof-of-concept for **activation anomaly detection** built on **interpretable latent features** (transcoder / SAE-like features). The core idea is to compare a **base** model and an **instruction-tuned** model on the *same prompts*, decompose their internal activation differences into **low-rank** and **sparse** structure, and then **causally validate** those components via targeted feature interventions.

> **Models used:** `google/gemma-2-2b` vs `google/gemma-2-2b-it` (open weights)  
> **Tooling:** `circuit-tracer==0.1.0`  
> **Feature slice (default):** layer 25 (configurable)

The repo is to learn interpretability workflow: detecting *when* internal computation is unusual, localizing *which features* are responsible, and validating that signal with controlled interventions.

## Why this project

This project aims to monitor difference between two models:

1. Run the same prompt through two closely related models (base vs instruction-tuned).
2. Compute **feature-level activation deltas** \(\Delta = F_{IT} - F_{base}\).
3. Decompose \(\Delta\) into:
   - **Low-rank structure**: shared, global shifts (e.g. instruction-following style or planning drift).
   - **Sparse structure**: prompt-specific residuals (“something unusual happened”).
4. Use the sparse residual both as:
   - an **anomaly score** (\(\|S_i\|_1\)), and
   - a short list of **interpretable features** to intervene on.
5. **Causally validate** both low-rank and sparse components by intervening on the implicated features and measuring changes in text *and* logit-space behavior.

To goal is to understand activation monitoring and see if the sparse component can act as anomaly detection.


## Files

- `notebooks/Transcoder_Anomaly_Detection.ipynb`  
  Single-prompt test and multi-prompt eval.
- `report_top10.html`  
  Side-by-side baseline vs low-rank/sparse interventions
- `scripts/`  
  A reproducible pipeline from prompt sets → feature matrices → decomposition → interventions.




## Reproducing the pipeline (scripts)

1. Generate prompt set:
```bash
python scripts/promptset_v0.py --out data/prompts_v0.jsonl --n 40
```

2. Collect feature matrices and Δ:
```bash
python scripts/collect_feature_matrices.py   --prompts data/prompts_v0.jsonl   --outdir outputs/step2   --layer 25   --base_model google/gemma-2-2b   --it_model google/gemma-2-2b-it
```
Outputs `feature_matrices.pt` containing `F_base`, `F_it`, and `Delta`.).

3. Low-rank + sparse decomposition:
```bash
python scripts/lowrank_sparse_anomaly.py   --inpt outputs/step2/feature_matrices.pt   --outdir outputs/step3   --rank 4   --tau_percentile 95
```

4. Select a case and run interventions:
```bash
python scripts/make_case.py   --outputs_dir outputs   --k_features 3   --out outputs/step4/case.json

python scripts/intervene.py   --case outputs/step4/case.json   --model google/gemma-2-2b-it   --k_features 3
```

5. Multi-prompt quantitative evaluation (recommended)
```bash
python scripts/batch_intervene_topk.py   --fm outputs/step2/feature_matrices.pt   --decomp outputs/step3/decomp.pt   --out outputs/step4/batch_results.csv   --topk_prompts 15
```

## Typical results and next steps

In some cases:
- **Low-rank removal** produces broad, early shifts in generation (often changing response trajectory quickly).
- **Sparse removal** is more prompt-selective, often acting as subthreshold pressure that only causes divergence later, especially on prompts with strict formatting or structural constraints.

Next steps:
- Using full Robust PCA / PCP solvers instead of PCA + thresholding.
- Aggregating across layers or generated-token windows.
- Explicit distribution-shift splits across prompt families.
- Tighter integration with attribution graphs for feature-to-circuit tracing.

## Data format for prompts

Scripts expect a JSONL file where each line is:
```json
{"id": 0, "text": "Answer in exactly two sentences. Explain containerization."}
```