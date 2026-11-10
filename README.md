# Mpox leakage experiment

Supplementary code for the review *AI-Based Mpox Detection From Skin Lesion Images, From CNNs to Foundation Models: A Critical Review of Data Leakage, Clade Drift, and Skin-Tone Fairness* (Mir Rahib Hussain Talpur et al., under review).

The script trains the same network on the same images under three evaluation protocols, so that the effect of the protocol alone can be measured:

| Protocol | What it does | Corresponds to |
|---|---|---|
| **A** augment-then-split | Each original image is expanded into 13 fixed augmented copies (14× in total, as in the MSLD release); the pooled set is split at random at image level | Studies that split or cross-validate a pre-augmented dataset |
| **B** image-level split | Originals are split at image level; augmentation only on the training fold; test folds hold originals only | Studies with clean augmentation but no patient grouping |
| **C** patient-grouped | Originals are split so that all images of one patient fall in the same fold (`StratifiedGroupKFold`); augmentation only on the training fold | Recommended protocol |

Model, epochs, optimiser, image size, seed and number of folds are identical across protocols. The model after the last epoch is evaluated once on each test fold; no test-fold model selection is done.

## Data

MSLD v2.0 (Ali *et al.*, 2024): 755 original images in six classes. Download it from the official sources and do not redistribute it:

- https://github.com/mHealthBuet/Mpox-Skin-Lesion-Dataset-v2
- https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20

Patient IDs are read from file names (`MKP_17_01` = class MKP, patient 17, image 1). Only the `Original Images` tree is used; augmented images are ignored.

## Run

macOS (Apple Silicon GPU), one step: double-click `run_experiment.command`. It creates a virtual environment, installs the requirements, downloads MSLD v2.0 and runs the quick configuration (ResNet-18, 10 epochs, 5 folds).

Any platform:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python leakage_experiment.py --data "path/to/Original Images" --arch resnet18 --epochs 10 --out results_resnet18.csv
```

The script uses CUDA, Apple MPS or CPU, whichever is available. Per-fold results are written to the CSV; a summary (mean ± SD over folds) is printed at the end.

## Results

See `results_resnet18.csv` and `run_log.txt`.

## License

Code: MIT. The dataset remains under its authors' terms.
