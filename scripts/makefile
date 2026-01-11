all:
	python scripts/promptset_v0.py --out data/prompts_v0.jsonl --n 40
	python scripts/collect_feature_matrices.py   --prompts data/prompts_v0.jsonl   --outdir outputs/step2   --layer 25
	python scripts/lowrank_sparse_anomaly.py \
  --inpt outputs/step2/feature_matrices.pt \
  --outdir outputs/step3 \
  --rank 10 \
  --tau_percentile 99