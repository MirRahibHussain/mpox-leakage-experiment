#!/bin/bash
# Double-click this file in Finder (or run: bash run_experiment.command).
# It installs PyTorch in a private folder, downloads MSLD v2.0 from Kaggle, and runs the
# leakage experiment on the Mac's GPU. Progress is written to run_log.txt next to this file.
cd "$(dirname "$0")" || exit 1
exec > >(tee -a run_log.txt) 2>&1
echo "== start $(date)"
command -v python3 >/dev/null || { echo "python3 not found. Install it from https://www.python.org/downloads/ and run again."; exit 1; }
python3 --version
[ -d .venv ] || python3 -m venv .venv || exit 1
source .venv/bin/activate
python -m pip install --upgrade pip -q
python -m pip install -q torch torchvision scikit-learn pillow kagglehub || { echo "pip install failed"; exit 1; }
python -c "import torch; print('torch', torch.__version__, 'MPS available:', torch.backends.mps.is_available())"

if [ ! -d data ]; then
  echo "== downloading MSLD v2.0"
  curl -fL -o msld_v2.zip "https://www.kaggle.com/api/v1/datasets/download/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20" \
    && mkdir -p data && unzip -q msld_v2.zip -d data && rm -f msld_v2.zip
  if [ ! -d data ]; then
    python - <<'PY'
import kagglehub, os
p = kagglehub.dataset_download("joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20")
os.symlink(p, "data"); print("downloaded to", p)
PY
  fi
fi
DATA=$(find -L data -type d -iname "Original Images" | head -1)
[ -n "$DATA" ] || { echo "Could not find the 'Original Images' folder inside data/"; exit 1; }
echo "DATA=$DATA"
echo "== training (quick run: ResNet-18, 10 epochs, 3 protocols x 5 folds)"
caffeinate -i python leakage_experiment_v1.py --data "$DATA" --arch resnet18 --epochs 10 --out results_resnet18.csv
echo "== done $(date)"
