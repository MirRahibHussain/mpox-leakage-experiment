# Summary of the controlled leakage experiment (MSLD v2.0, six classes, five-fold CV)

Accuracy %, mean over all folds and seeds ± mean SD over folds; macro-F1 in brackets. A: one seed; AG, B, C: three seeds. All protocols train on all 14 versions (original + 13 fixed augmented copies) of their training originals, so the training budget is identical.

| Network | A: augment, then random split | AG: copies of each original kept in one fold | B: image-level split, originals tested | C: patient-grouped split |
|---|---|---|---|---|
| resnet18 | 99.74 ± 0.04 (99.73) | 85.45 ± 2.22 (83.59) | 88.21 ± 2.95 (86.68) | 80.04 ± 3.57 (77.48) |
| resnet50 | 99.81 ± 0.09 (99.82) | 85.75 ± 2.49 (84.28) | 86.58 ± 2.69 (85.32) | 79.72 ± 3.44 (77.91) |
| efficientnet_b0 | 99.98 ± 0.03 (99.98) | 86.86 ± 1.67 (85.37) | 88.12 ± 2.24 (86.58) | 81.00 ± 3.66 (78.76) |

Leakage check (fold 1, seed 0): A 100.0% of test images have a copy of the same original in training; B 40.4% of test images share a patient with training; C 0.0%.

Settings: 10 epochs, AdamW (lr 3e-4, weight decay 1e-2), batch 64, cosine schedule, mixed precision, 224×224, 13 fixed copies per original (rotation ±30°, h/v flip, random-resized-crop 0.7–1.0, colour jitter 0.2/0.2/0.2/0.02), last-epoch model, no model selection. Kaggle, 2× NVIDIA T4, run time 5 h 35 min.
