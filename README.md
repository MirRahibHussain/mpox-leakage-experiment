# Mpox leakage experiment and protocol audit

Supplementary code and data for the review *AI-Based Mpox Detection From Skin Lesion Images, From CNNs to Foundation Models: A Critical Review of Data Leakage, Clade Drift, and Skin-Tone Fairness* (Mir Rahib Hussain Talpur and Ghulam Ali Rahu, Sindh Agriculture University, Tandojam).

The repository has two parts:

1. **`leakage_experiment.py`**: the controlled experiment of Section VII-D and Table IV.
2. **`audit/`**: the search log, screening decisions and random-sample coding sheet behind Section III-E, Section VI and Fig. 5.

## 1. Controlled leakage experiment

The same networks are trained on the same images under four protocols. Every original image gets 13 fixed augmented copies (14 versions, the expansion factor of the augmented MSLD release), and every protocol trains on all 14 versions of its training originals, so **all protocols take the same number of optimisation steps**.

| Protocol | What it does |
|---|---|
| **A** augment, then split | The pooled 10,570 images are split at random, so copies of one original fall on both sides |
| **AG** control for A | The same pooled images, but all 14 versions of an original stay in one fold |
| **B** image-level split | Same models as AG, tested on the held-out originals only |
| **C** patient-grouped | `StratifiedGroupKFold` by patient (IDs from file names), tested on originals only |

A minus AG isolates leakage with the training budget held constant; B minus C isolates patient grouping.

### Results

MSLD v2.0 (755 images, six classes, 524 patients parsed from file names), ImageNet weights, five-fold CV. Accuracy %, mean over folds and seeds ± mean SD over folds; macro-F1 in brackets. A: one seed; AG, B and C: three seeds.

| Network | A | AG | B | C |
|---|---|---|---|---|
| ResNet-18 | 99.74 ± 0.04 (99.73) | 85.45 ± 2.22 (83.59) | 88.21 ± 2.95 (86.68) | 80.04 ± 3.57 (77.48) |
| ResNet-50 | 99.81 ± 0.09 (99.82) | 85.75 ± 2.49 (84.28) | 86.58 ± 2.69 (85.32) | 79.72 ± 3.44 (77.91) |
| EfficientNet-B0 | 99.98 ± 0.03 (99.98) | 86.86 ± 1.67 (85.37) | 88.12 ± 2.24 (86.58) | 81.00 ± 3.66 (78.76) |

Per-fold values: `results/per_fold_results.csv`. Summary and settings: `results/summary.md`.

### Run

The experiment was run on Kaggle (GPU T4 ×2, about 5.5 hours) with the dataset [Mpox Skin Lesion Dataset Version 2.0 (MSLD v2.0)](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20) attached:

```bash
python leakage_experiment.py
```

or import `notebooks/kaggle_mpox_leakage_experiment.ipynb` into Kaggle. Settings can be changed through environment variables (`EPOCHS`, `NETWORKS`, `SEEDS_CLEAN`, `DATA_ROOT`, `OUT_DIR`, ...); see the top of the script. The script resumes from `results.json` if interrupted.

Download MSLD v2.0 from its official sources and do not redistribute it (CC BY-NC 4.0):
https://github.com/mHealthBuet/Mpox-Skin-Lesion-Dataset-v2

## 2. Protocol audit (`audit/`)

| File | Content |
|---|---|
| `search_strategy.csv` | Exact search strings, dates, deduplication and sampling rules |
| `screening_log.csv` | Every record returned by PubMed (91), IEEE Xplore (186), arXiv (44) and Europe PMC preprints (17) on 10 November 2026 (338 records), with the screening decision and reason; 274 relevant after deduplication and screening |
| `coding_sheet.csv` | The 50 sampled records; 45 coded from full text with group (A/B/C/U), reporting items, headline accuracy and the supporting evidence for each assignment |
| `Supplementary_search_log_and_coding_sheet.xlsx` | The same, as one workbook |
| `trend_test.py` | Jonckheere–Terpstra and Mann–Whitney tests behind Section VI |
| `Fig2_records_by_year.png`, `Fig_S1_record_flow.png`, `Fig5_audit.png` | Figures built from these files |

Full texts are not redistributed; every record is identified by PMID, PMCID, DOI or arXiv ID.

## Earlier version

`legacy/` holds the first version of the experiment (three protocols, protocol A with 14× more optimisation steps) and its logs, kept for transparency.

## License

Code: MIT. MSLD v2.0 remains under its authors' terms.
