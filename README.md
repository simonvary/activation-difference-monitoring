# Robust model-difference anomaly detection on transcoder/SAE-like features (Gemma-2-2B)
This notebook is a **self-contained demo** of the core methodology:

1. **Collect interpretable feature activations** (transcoder features) for the *same prompts* under two models  
   - `Gemma-2-2B` (base)  
   - `Gemma-2-2B-it` (instruction-tuned)

2. Compute a **model-difference matrix**:
\[ \Delta = F_{it} - F_{base} \]

3. Decompose differences into **low-rank + sparse** components (Robust-PCA style):
\[ \Delta \approx L + S \]
- **Low-rank `L`**: shared, correlated shifts (often “global instruction-tuning style”)  
- **Sparse `S`**: prompt-specific spikes (candidate anomaly signal)

4. Pick one anomaly prompt and **intervene** on the top sparse features to causally test their role.

This is designed to be readable (job-application friendly) and tweakable: you can play with **layer**, **rank**, **threshold**, **intervention mode**, and **prompt selection**.
