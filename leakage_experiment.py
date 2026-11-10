"""
Leakage experiment for mpox skin-lesion classification (supplementary material).

Trains the same network on the same images under three evaluation protocols and reports
accuracy for each, so that the effect of the protocol alone can be measured:

  A  "augment-then-split"  every original image is expanded into K augmented copies first,
                            and the pooled set is split at random at image level
                            (what many published studies do with pre-augmented releases)
  B  "image-level split"    originals are split at image level; augmentation is applied
                            to the training fold only; test folds hold originals only
  C  "patient-grouped"      originals are split so that all images of one patient fall in
                            the same fold; augmentation on the training fold only

Everything else (model, epochs, optimiser, image size, seeds, folds) is identical.

-------------------------------------------------------------------------------------------
How to run in Google Colab (GPU runtime):

  1. Download MSLD v2.0 (Ali et al., 2024) from its official page:
       https://github.com/mHealthBuet/Mpox-Skin-Lesion-Dataset-v2   (links to the Kaggle copy)
     Use the folder of ORIGINAL images (one sub-folder per class), not the augmented folders.
  2. Upload/unzip it, e.g. to /content/MSLDv2/Original Images/
  3. !pip install timm==1.0.9  (optional; torchvision models are used by default)
  4. !python leakage_experiment.py --data "/content/MSLDv2/Original Images" --out results.csv

  A T4 GPU needs roughly 1-2 h for 3 protocols x 5 folds x 15 epochs with ResNet-50.
  Use --arch resnet18 --epochs 10 for a quick first run.

Patient IDs: MSLD v2.0 file names encode the patient (e.g. MKP_17_02.jpg = class MKP,
patient 17, image 2). The default --group-regex extracts "<class>_<patient>". CHECK THIS on
your copy: the script prints the number of images, groups and images per group before
training. If your file names differ, change --group-regex. If no patient ID can be parsed,
the file name itself is used as the group (protocol C then equals protocol B).

Report in the paper: mean +/- SD accuracy and macro-F1 across folds for each protocol, the
number of images and patients, K, the architecture and the number of epochs.
-------------------------------------------------------------------------------------------
"""
import argparse, csv, os, re, random, time, zlib
from collections import Counter

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ------------------------------------------------------------------------------- data
def list_images(root, group_regex):
    """Two layouts are supported:
    (1) one sub-folder per class holding the original images;
    (2) the official MSLD v2.0 'Original Images/FOLDS/foldX/{train,valid,test}/...' tree, in which
        every image appears once per fold: files are collected recursively, de-duplicated by file
        name, and the class is read from the file-name prefix (MKP, CHP, CWP, MSL, HFMD, HEALTHY)."""
    rx = re.compile(group_regex) if group_regex else None
    subdirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    flat = subdirs and all(
        any(f.lower().endswith(IMG_EXT) for f in os.listdir(os.path.join(root, d))) for d in subdirs)
    found = {}
    if flat:
        for c in subdirs:
            for f in sorted(os.listdir(os.path.join(root, c))):
                if f.lower().endswith(IMG_EXT):
                    found.setdefault(f, (os.path.join(root, c, f), c))
    else:
        for dp, _, fs in os.walk(root):
            if "aug" in dp.lower():          # never pick up augmented copies
                continue
            for f in sorted(fs):
                if f.lower().endswith(IMG_EXT):
                    found.setdefault(f, (os.path.join(dp, f), f.split("_")[0].upper()))
    classes = sorted({c for _, c in found.values()})
    paths, labels, groups = [], [], []
    for f in sorted(found):
        p, c = found[f]
        stem = os.path.splitext(f)[0]
        m = rx.search(stem) if rx else None
        paths.append(p); labels.append(classes.index(c))
        groups.append(f"{c}/{m.group(1)}" if m else f"{c}/{stem}")
    return classes, np.array(paths), np.array(labels), np.array(groups)


def make_tf(size, train):
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.7, 1.0)), T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(), T.RandomRotation(25),
            T.ColorJitter(0.2, 0.2, 0.2, 0.02), T.ToTensor(), norm])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), norm])


class ImageSet(Dataset):
    """items: list of (path, label, aug_id). aug_id = -1 -> original image (eval transform);
    aug_id >= 0 -> a fixed augmented copy, regenerated deterministically from (path, aug_id),
    which is equivalent to a pre-augmented release written to disk. online=True applies a
    fresh random augmentation every epoch (training folds of protocols B and C)."""

    def __init__(self, items, size, online=False, base_seed=0):
        self.items, self.online, self.base_seed = items, online, base_seed
        self.tf_aug, self.tf_eval = make_tf(size, True), make_tf(size, False)
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def _load(self, p):
        if p not in self.cache:
            self.cache[p] = Image.open(p).convert("RGB")
        return self.cache[p]

    def __getitem__(self, i):
        p, y, a = self.items[i]
        img = self._load(p)
        if self.online:
            x = self.tf_aug(img)
        elif a >= 0:
            seed = (zlib.crc32(os.path.basename(p).encode()) % 100003) * 1000 + a + self.base_seed
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                x = self.tf_aug(img)
        else:
            x = self.tf_eval(img)
        return x, y


# ------------------------------------------------------------------------------ model
def build_model(arch, n_classes, pretrained=True):
    w = "DEFAULT" if pretrained else None
    m = getattr(torchvision.models, arch)(weights=w)
    if hasattr(m, "fc"):
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif hasattr(m, "classifier"):
        last = m.classifier[-1] if isinstance(m.classifier, nn.Sequential) else m.classifier
        new = nn.Linear(last.in_features, n_classes)
        if isinstance(m.classifier, nn.Sequential):
            m.classifier[-1] = new
        else:
            m.classifier = new
    return m


def run_fold(train_items, test_items, n_classes, args, online_train, dev):
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    tr = DataLoader(ImageSet(train_items, args.size, online=online_train), batch_size=args.bs,
                    shuffle=True, num_workers=args.workers, drop_last=False)
    te = DataLoader(ImageSet(test_items, args.size, online=False), batch_size=args.bs,
                    shuffle=False, num_workers=args.workers)
    model = build_model(args.arch, n_classes, not args.no_pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = lossf(model(x), y); loss.backward(); opt.step()
        sched.step()
    # No model selection on the test fold: the model after the last epoch is evaluated once.
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for x, y in te:
            P.append(model(x.to(dev)).argmax(1).cpu().numpy()); Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    return accuracy_score(Y, P), f1_score(Y, P, average="macro"), len(Y)


# ---------------------------------------------------------------------------- protocols
def folds_A(paths, labels, K, k_folds, seed):
    """Augment-then-split: pool = originals + K fixed augmented copies of each, split at random."""
    pool = [(p, int(y), -1) for p, y in zip(paths, labels)]
    pool += [(p, int(y), a) for p, y in zip(paths, labels) for a in range(K)]
    yy = np.array([t[1] for t in pool])
    skf = StratifiedKFold(k_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.zeros(len(pool)), yy):
        yield [pool[i] for i in tr], [pool[i] for i in te], False   # copies are fixed, no online aug


def folds_B(paths, labels, k_folds, seed):
    skf = StratifiedKFold(k_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(paths, labels):
        yield ([(paths[i], int(labels[i]), -1) for i in tr],
               [(paths[i], int(labels[i]), -1) for i in te], True)


def folds_C(paths, labels, groups, k_folds, seed):
    sgk = StratifiedGroupKFold(k_folds, shuffle=True, random_state=seed)
    for tr, te in sgk.split(paths, labels, groups):
        assert not set(groups[tr]) & set(groups[te]), "patient overlap between folds"
        yield ([(paths[i], int(labels[i]), -1) for i in tr],
               [(paths[i], int(labels[i]), -1) for i in te], True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder of ORIGINAL images (class sub-folders, or the MSLD v2.0 Original Images tree)")
    ap.add_argument("--out", default="leakage_results.csv")
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--K", type=int, default=13, help="augmented copies per original in protocol A "
                    "(13 copies + original = 14x, as in the MSLD release)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=0 if os.name != "nt" and os.uname().sysname == "Darwin" else 2)
    ap.add_argument("--group-regex", default=r"^[A-Za-z]+_(\d+)", help="regex whose group(1) is the patient ID")
    ap.add_argument("--protocols", default="A,B,C")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    classes, paths, labels, groups = list_images(args.data, args.group_regex)
    gcount = Counter(groups)
    print(f"classes: {classes}")
    print(f"images: {len(paths)}  groups (patients): {len(gcount)}  "
          f"images per group: max {max(gcount.values())}, mean {len(paths) / len(gcount):.2f}")
    print("per class:", dict(Counter(classes[i] for i in labels)))
    if len(gcount) == len(paths):
        print("WARNING: every image is its own group, so no patient ID was parsed; protocol C = protocol B.")

    if torch.cuda.is_available():
        dev = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = "mps"          # Apple Silicon GPU (MacBook M1/M2/M3/M4)
    else:
        dev = "cpu"
    print(f"device: {dev}")
    rows = []
    for proto in args.protocols.split(","):
        if proto == "A":
            gen = folds_A(paths, labels, args.K, args.folds, args.seed)
        elif proto == "B":
            gen = folds_B(paths, labels, args.folds, args.seed)
        else:
            gen = folds_C(paths, labels, groups, args.folds, args.seed)
        for f, (tr, te, online) in enumerate(gen, 1):
            t0 = time.time()
            acc, f1, n = run_fold(tr, te, len(classes), args, online, dev)
            rows.append(dict(protocol=proto, fold=f, n_train=len(tr), n_test=n, accuracy=acc, macro_f1=f1))
            print(f"[{proto}] fold {f}: acc {acc:.4f}  macro-F1 {f1:.4f}  (train {len(tr)}, test {n}, "
                  f"{time.time() - t0:.0f}s)", flush=True)
            with open(args.out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print("\nSummary (mean +/- SD over folds)")
    for proto in args.protocols.split(","):
        a = np.array([r["accuracy"] for r in rows if r["protocol"] == proto])
        f = np.array([r["macro_f1"] for r in rows if r["protocol"] == proto])
        print(f"  {proto}: accuracy {100 * a.mean():.2f} +/- {100 * a.std(ddof=1):.2f}   "
              f"macro-F1 {100 * f.mean():.2f} +/- {100 * f.std(ddof=1):.2f}")
    print(f"\nPer-fold results written to {args.out}")


if __name__ == "__main__":
    main()
