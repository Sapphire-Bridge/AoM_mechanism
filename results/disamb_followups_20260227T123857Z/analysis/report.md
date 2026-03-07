# DISAMB Follow-up Analysis

## Bootstrap Stability (L12, k=20)
- n=4; Jaccard mean=0.484; min=0.379; max=0.538
- top-k effect mean across resamples=0.0164; sd=0.0428

## Split-Seed Stability (L12, k=20)
- seeds=[0, 1, 2]
- topk-random margin mean=0.0720; variance=0.000436
- pairwise Jaccard mean=0.673; min=0.600; max=0.818

## Outputs
- Bootstrap table: `results/disamb_followups_20260227T123857Z/analysis/bootstrap_stability_l12_k20.csv`
- Split-seed table: `results/disamb_followups_20260227T123857Z/analysis/split_seed_stability_l12_k20.csv`
- k-sweep table: `results/disamb_followups_20260227T123857Z/analysis/k_sweep_separation_l4_l8_l12.csv`
- Scale sweep table: `results/disamb_followups_20260227T123857Z/analysis/scale_sweep_l12_k20.csv`
- Cross-task table: `results/disamb_followups_20260227T123857Z/analysis/cross_task_synthesis_crr_c_over_a.csv`
- Plots directory: `results/disamb_followups_20260227T123857Z/plots`
