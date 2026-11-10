# Summary of the controlled leakage experiment (MSLD v2.0, six classes, five-fold CV)

Accuracy %, mean over all folds and seeds ± mean SD over folds; macro-F1 in brackets. Three seeds for every protocol; the 20-epoch ResNet-18 check uses one seed. All protocols train on all 14 versions (original + 13 fixed augmented copies) of their training originals, so the training budget is identical. Patients are grouped by the 524 identifiers parsed from the file names (the dataset authors report 541 patients; the lower count makes grouping stricter).

| Network | A: augment, then random split | AG: copies of each original kept in one fold | B: image-level split, originals tested | C: patient-grouped split |
|---|---|---|---|---|
| ResNet-18 | 99.76 ± 0.07 (99.75) | 85.45 ± 2.22 (83.59) | 88.21 ± 2.95 (86.68) | 80.04 ± 3.57 (77.48) |
| ResNet-50 | 99.85 ± 0.10 (99.86) | 85.75 ± 2.49 (84.28) | 86.58 ± 2.69 (85.32) | 79.72 ± 3.44 (77.91) |
| EfficientNet-B0 | 99.98 ± 0.02 (99.99) | 86.86 ± 1.67 (85.37) | 88.12 ± 2.24 (86.58) | 81.00 ± 3.66 (78.76) |
| Swin-T | 99.99 ± 0.02 (99.99) | 89.30 ± 2.07 (87.64) | 90.02 ± 1.97 (88.03) | 83.82 ± 3.08 (81.24) |
| ResNet-18, 20 epochs (1 seed) | 99.75 ± 0.09 (99.74) | 85.22 ± 2.60 (83.26) | 85.56 ± 3.98 (84.05) | 79.85 ± 2.96 (77.21) |

Gaps (percentage points): leakage A − AG = 10.7–14.3; patient grouping B − C = 6.2–8.2. Within every network no seed of protocol C reaches any seed of protocol B (seed means 86.2–90.6% under B, 78.4–84.1% under C).

Leakage check (fold 1, seed 0): A 100.0% of test images have a copy of the same original in training; B 40.4% of test images share a patient with training; C 0.0%.

Settings: 10 epochs, AdamW (weight decay 1e-2), batch 64, cosine schedule, mixed precision, 224×224, lr 3e-4 for the CNNs and 1e-4 for Swin-T, fixed per architecture before any run; 13 fixed copies per original (rotation ±30°, h/v flip, random-resized-crop 0.7–1.0, colour jitter 0.2/0.2/0.2/0.02); last-epoch model, no model selection. Kaggle, 2× NVIDIA T4. First run (CNNs, A seed 0; AG/B/C seeds 0–2): 5 h 35 min. Revision run (A seeds 1–2 for the CNNs, Swin-T, 20-epoch check; `notebooks/kaggle_mpox_leakage_experiment_v3.ipynb`, one process per GPU): 4 h 45 min.

Files: `per_fold_results.csv` (all 260 fold results), `per_fold_results_revision_run.csv` (the 110 results of the revision run).
