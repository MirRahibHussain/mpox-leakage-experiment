"""
Controlled leakage experiment on MSLD v2.0  (replaces Table IV, Section VII-D)
==============================================================================
Run on Kaggle (GPU T4, Internet ON) with the dataset
"Mpox Skin Lesion Dataset Version 2.0 (MSLD v2.0)" attached:

    !python leakage_experiment.py

Design. Every original image gets 13 FIXED augmented copies (14 versions in total, the
expansion factor of the augmented MSLD release). Every protocol trains on all 14 versions
of its training originals, so all protocols take the SAME number of optimisation steps.
The only thing that changes is how images are assigned to folds and what is tested:

  A    augment, then random split of the pooled 10,570 images (copies leak across folds)
  AG   same pooled images, but all 14 versions of an original stay in one fold
       (no leakage; test fold = originals + their copies)      <- the control for A
  B    image-level split of the 755 originals; test on originals only
  C    patient-grouped split (StratifiedGroupKFold); test on originals only
  D    official MSLD v2.0 folds; test on originals only

Networks: ResNet-18, ResNet-50, EfficientNet-B0 (lr 3e-4) and Swin-T (lr 1e-4), ImageNet weights.
All protocols use three seeds. "resnet18@e20" = ResNet-18 trained for 20 epochs (sensitivity check).

AG and B share the same trained models (same folds), they differ only in the test images.
A - AG isolates leakage with training budget held constant. B - C isolates patient grouping.

Outputs (in OUT_DIR): results.json (every fold of every run), summary.csv, summary.md.
The script resumes from results.json if the session is interrupted: just run it again.
"""
import os, re, json, glob, time, math, zlib
from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import f1_score, balanced_accuracy_score

# ----------------------------------------------------------------------------- settings
DATA_ROOT   = os.environ.get("DATA_ROOT", "/kaggle/input")
OUT_DIR     = os.environ.get("OUT_DIR", "/kaggle/working")
NETWORKS    = os.environ.get("NETWORKS", "resnet18,resnet50,efficientnet_b0,swin_t").split(",")
EPOCHS      = int(os.environ.get("EPOCHS", 10))
BATCH       = int(os.environ.get("BATCH", 64))
LR          = float(os.environ.get("LR", 3e-4))                # CNNs
LR_TRANSFORMER = float(os.environ.get("LR_TRANSFORMER", 1e-4))  # Swin-T (standard fine-tuning rate)
WEIGHT_DECAY= float(os.environ.get("WEIGHT_DECAY", 1e-2))
IMG         = int(os.environ.get("IMG", 224))
N_COPIES    = int(os.environ.get("N_COPIES", 13))        # augmented copies per original
N_FOLDS     = 5
SEEDS_LEAKY = [int(s) for s in os.environ.get("SEEDS_LEAKY", "0,1,2").split(",")]  # protocol A
SEEDS_CLEAN = [int(s) for s in os.environ.get("SEEDS_CLEAN", "0,1,2").split(",")]  # protocols B/AG and C
SEEDS_OFFIC = [int(s) for s in os.environ.get("SEEDS_OFFIC", "0").split(",")]      # protocol D
PRETRAINED  = os.environ.get("PRETRAINED", "1") == "1"
AUG_SEED    = 2026
TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", 10.0))   # stop starting new runs after this
RES_NAME    = os.environ.get("RES_NAME", "results.json")
# Optional explicit job list "net|proto|seed;net|proto|seed;..." (proto B also yields AG).
# A network may carry an epoch override, e.g. "resnet18@e20" trains ResNet-18 for 20 epochs.
JOBS        = os.environ.get("JOBS", "")
IMG_EXT     = (".jpg", ".jpeg", ".png", ".bmp")
NAME_RE     = re.compile(r"^([A-Za-z]+)[_-](\d+)[_-](\d+)")   # CLASS_patient_image, e.g. MKP_17_02.jpg


# ----------------------------------------------------------------------------- data index
def index_dataset(root):
    """Find the 755 originals (each appears in several official folds; keep one path per file
    name), their class (parent folder), patient id (from the file name) and official test fold."""
    paths = [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
             if p.lower().endswith(IMG_EXT)]
    orig = [p for p in paths if "aug" not in p.lower() and NAME_RE.match(os.path.basename(p))]
    skipped = [p for p in paths if "aug" not in p.lower() and not NAME_RE.match(os.path.basename(p))]
    if skipped:
        print(f"skipped {len(skipped)} non-dataset image(s), e.g. {skipped[0]}")
    assert orig, f"No original images found under {root}. Attach MSLD v2.0 or set DATA_ROOT."
    by_name, official_test = {}, {}
    for p in sorted(orig):
        name = os.path.basename(p)
        by_name.setdefault(name, p)
        m = re.search(r"fold[_ ]?(\d)", p, flags=re.I)
        if m and os.path.basename(os.path.dirname(os.path.dirname(p))).lower().startswith("test"):
            official_test[name] = int(m.group(1)) - 1
    names = sorted(by_name)
    classes = sorted({os.path.basename(os.path.dirname(by_name[n])) for n in names})
    y = np.array([classes.index(os.path.basename(os.path.dirname(by_name[n]))) for n in names])
    patients = []
    for n in names:
        m = NAME_RE.match(n)
        assert m, f"File name {n} does not follow CLASS_patient_image; cannot derive a patient id."
        patients.append(f"{m.group(1).upper()}_{int(m.group(2))}")
    uniq = sorted(set(patients))
    g = np.array([uniq.index(p) for p in patients])
    off = np.array([official_test.get(n, -1) for n in names])
    return names, [by_name[n] for n in names], y, g, off, classes


def make_folds(y, g, off, seed):
    """Return {protocol: [(train_idx, test_idx), ...]} in units described in run_protocol()."""
    n = len(y)
    folds = {}
    # A: random stratified split of the pooled (original + copies) images
    V = N_COPIES + 1
    src = np.repeat(np.arange(n), V)          # source original of every pooled image
    ver = np.tile(np.arange(V), n)            # 0 = original, 1..13 = augmented copy
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    folds["A"] = [(tr, te) for tr, te in skf.split(src, y[src])]
    # B (and AG): image-level split of the originals
    folds["B"] = [(tr, te) for tr, te in skf.split(np.arange(n), y)]
    # C: patient-grouped split of the originals
    sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
    folds["C"] = [(tr, te) for tr, te in sgkf.split(np.arange(n), y, g)]
    # D: official folds
    if (off >= 0).all():
        folds["D"] = [(np.where(off != k)[0], np.where(off == k)[0]) for k in range(N_FOLDS)]
    return folds, src, ver


def leakage_report(folds, src, g):
    """Share of test images whose source image / patient also occurs in the training fold."""
    rep = {}
    tr, te = folds["A"][0]
    rep["A: test images with a copy of the same original in training (%)"] = \
        100 * np.isin(src[te], src[tr]).mean()
    for k in ("B", "C", "D"):
        if k in folds:
            tr, te = folds[k][0]
            rep[f"{k}: test images whose patient is also in training (%)"] = \
                100 * np.isin(g[te], g[tr]).mean()
    return rep


# ----------------------------------------------------------------------------- images
def build_cache(paths):
    """uint8 array [n, 14, 3, IMG, IMG]: version 0 = resized original, 1..13 = fixed copies."""
    import torch
    from PIL import Image
    from torchvision import transforms as T
    f = os.path.join(OUT_DIR, f"cache_{len(paths)}x{N_COPIES+1}_{IMG}.npy")
    if os.path.exists(f):
        return np.load(f, mmap_mode="r")
    aug = T.Compose([
        T.RandomRotation(30),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomResizedCrop(IMG, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        T.ColorJitter(0.2, 0.2, 0.2, 0.02),
    ])
    out = np.lib.format.open_memmap(f, mode="w+", dtype=np.uint8,
                                    shape=(len(paths), N_COPIES + 1, 3, IMG, IMG))
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB").resize((256, 256), Image.BILINEAR)
        out[i, 0] = np.asarray(im.resize((IMG, IMG), Image.BILINEAR)).transpose(2, 0, 1)
        for k in range(1, N_COPIES + 1):
            torch.manual_seed(AUG_SEED * 100003 + i * 101 + k)    # fixed copy, same in every protocol
            out[i, k] = np.asarray(aug(im)).transpose(2, 0, 1)
        if i % 100 == 0:
            print(f"  building augmented copies {i}/{len(paths)}", flush=True)
    out.flush()
    return np.load(f, mmap_mode="r")


# ----------------------------------------------------------------------------- training
def parse_net(spec):
    """'resnet18@e20' -> ('resnet18', 20); 'swin_t' -> ('swin_t', EPOCHS)."""
    name, _, opt = spec.partition("@")
    return name, (int(opt[1:]) if opt.startswith("e") else EPOCHS)


def build_model(name, n_classes):
    import torch.nn as nn
    from torchvision import models
    w = "IMAGENET1K_V1" if PRETRAINED else None
    m = getattr(models, name)(weights=w)
    if hasattr(m, "head"):                      # Swin / ViT-style transformers
        m.head = nn.Linear(m.head.in_features, n_classes)
    elif hasattr(m, "fc"):
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    else:
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    return m


def train_eval(spec, X, tr_s, tr_v, y_tr, tests, n_classes, seed):
    """Train on images X[tr_s, tr_v]; evaluate the last-epoch model once on every test set.
    tests = {label: (src_idx, ver_idx, y)}. No model selection of any kind."""
    import torch, torch.nn.functional as F
    name, epochs = parse_net(spec)
    lr = LR_TRANSFORMER if name.startswith(("swin", "vit")) else LR
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def batch(s, v):
        order = np.argsort(s * 100 + v)                       # sorted read from the memmap
        xb = np.stack([X[a, b] for a, b in zip(s[order], v[order])])
        xb = xb[np.argsort(order)]
        return ((torch.from_numpy(xb).to(dev).float() / 255) - mean) / std

    model = build_model(name, n_classes).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    steps = epochs * math.ceil(len(tr_s) / BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    scaler = torch.amp.GradScaler(enabled=(dev == "cuda"))
    rng = np.random.default_rng(seed)
    yt = torch.from_numpy(y_tr).to(dev)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(tr_s))
        for b in range(0, len(perm), BATCH):
            idx = perm[b:b + BATCH]
            xb = batch(tr_s[idx], tr_v[idx]).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=dev, enabled=(dev == "cuda")):
                loss = F.cross_entropy(model(xb), yt[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    model.eval(); res = {}
    with torch.no_grad():
        for label, (s, v, yy) in tests.items():
            pred = []
            for b in range(0, len(s), 256):
                xb = batch(s[b:b + 256], v[b:b + 256]).contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=dev, enabled=(dev == "cuda")):
                    pred.append(model(xb).argmax(1).cpu().numpy())
            pred = np.concatenate(pred)
            res[label] = dict(acc=100 * float((pred == yy).mean()),
                              macro_f1=100 * float(f1_score(yy, pred, average="macro")),
                              bal_acc=100 * float(balanced_accuracy_score(yy, pred)),
                              n_test=int(len(yy)))
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return res


def expand(idx):
    """All 14 versions of the given originals -> (source index, version index)."""
    V = N_COPIES + 1
    return np.repeat(idx, V), np.tile(np.arange(V), len(idx))


def run_protocol(proto, fold, folds, src, ver, y):
    """Return training (source, version, label) and the dict of test sets for one fold."""
    tr, te = folds[proto][fold]
    if proto == "A":                                   # pooled images, copies leak
        return src[tr], ver[tr], y[src[tr]], {"A": (src[te], ver[te], y[src[te]])}
    s, v = expand(tr)                                  # train: all versions of training originals
    zeros = np.zeros(len(te), dtype=int)
    tests = {proto: (te, zeros, y[te])}                # test: originals only
    if proto == "B":                                   # control AG: same model, test incl. copies
        ts, tv = expand(te)
        tests["AG"] = (ts, tv, y[ts])
    return s, v, y[s], tests


# ----------------------------------------------------------------------------- summary
def summarise(results):
    """Per network and protocol: mean over all folds and seeds, SD over folds, SD of seed means."""
    cell = defaultdict(lambda: defaultdict(list))
    for r in results.values():
        cell[(r["network"], r["protocol"])][r["seed"]].append(r)
    rows = []
    for (net, proto), by_seed in sorted(cell.items()):
        for metric in ("acc", "macro_f1", "bal_acc"):
            seed_means = [np.mean([r[metric] for r in rs]) for rs in by_seed.values()]
            allv = [r[metric] for rs in by_seed.values() for r in rs]
            fold_sd = np.mean([np.std([r[metric] for r in rs], ddof=1) for rs in by_seed.values()])
            rows.append(dict(network=net, protocol=proto, metric=metric,
                             mean=np.mean(allv), sd_folds=fold_sd,
                             sd_seeds=np.std(seed_means, ddof=1) if len(seed_means) > 1 else float("nan"),
                             n_seeds=len(seed_means), n_folds=len(allv) // len(seed_means),
                             n_test_per_fold=int(np.mean([r["n_test"] for rs in by_seed.values() for r in rs]))))
    return rows


def write_summary(results, leak):
    rows = summarise(results)
    with open(os.path.join(OUT_DIR, "summary.csv"), "w") as f:
        f.write(",".join(rows[0].keys()) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")
    order = ["A", "AG", "B", "C", "D"]
    acc = {(r["network"], r["protocol"]): r for r in rows if r["metric"] == "acc"}
    f1 = {(r["network"], r["protocol"]): r for r in rows if r["metric"] == "macro_f1"}
    nets = sorted({k[0] for k in acc})
    lines = ["Accuracy %, mean +/- SD over folds (macro-F1). [SD of seed means, number of seeds]", "",
             "| Network | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    for n in nets:
        cells = []
        for p in order:
            if (n, p) in acc:
                a, m = acc[(n, p)], f1[(n, p)]
                extra = f" [{a['sd_seeds']:.2f}, {a['n_seeds']}]" if a["n_seeds"] > 1 else ""
                cells.append(f"{a['mean']:.2f} +/- {a['sd_folds']:.2f} ({m['mean']:.2f}){extra}")
            else:
                cells.append("-")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += ["", "Gaps in accuracy (percentage points):"]
    for n in nets:
        g = lambda a, b: acc[(n, a)]["mean"] - acc[(n, b)]["mean"] if (n, a) in acc and (n, b) in acc else float("nan")
        lines.append(f"- {n}: leakage A-AG = {g('A','AG'):.2f}; copies vs originals in test AG-B = {g('AG','B'):.2f}; "
                     f"patient grouping B-C = {g('B','C'):.2f}; official folds D = "
                     f"{acc[(n,'D')]['mean']:.2f}" if (n, 'D') in acc else f"- {n}: A-AG = {g('A','AG'):.2f}; B-C = {g('B','C'):.2f}")
    lines += ["", "Leakage check (fold 1):"] + [f"- {k}: {v:.1f}" for k, v in leak.items()]
    lines += ["", f"Settings: epochs={EPOCHS}, batch={BATCH}, lr={LR}, weight_decay={WEIGHT_DECAY}, AdamW, cosine schedule, "
              f"{IMG}x{IMG}, {N_COPIES} fixed copies per original (rotation +/-30, h/v flip, "
              f"random-resized-crop 0.7-1.0, colour jitter 0.2/0.2/0.2/0.02), last-epoch model, no model selection."]
    open(os.path.join(OUT_DIR, "summary.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    names, paths, y, g, off, classes = index_dataset(DATA_ROOT)
    print(f"{len(names)} original images, {len(set(g))} patients (from file names), classes: {classes}")
    print("images per class:", dict(zip(classes, np.bincount(y).tolist())))
    if (off < 0).any():
        print(f"WARNING: official test fold not found for {(off < 0).sum()} images; protocol D is skipped.")
    folds0, src, ver = make_folds(y, g, off, 0)
    leak = leakage_report(folds0, src, g)
    for k, v in leak.items():
        print(f"  {k}: {v:.1f}")
    X = build_cache(paths)

    if os.environ.get("CACHE_ONLY") == "1":
        return
    res_file = os.path.join(OUT_DIR, RES_NAME)
    results = json.load(open(res_file)) if os.path.exists(res_file) else {}
    plan = [("A", SEEDS_LEAKY), ("B", SEEDS_CLEAN), ("C", SEEDS_CLEAN), ("D", SEEDS_OFFIC)]
    # first seed of every protocol first, extra seeds afterwards, so a time-out still leaves a full table
    jobs = sorted(((i, proto, seed) for proto, seeds in plan for i, seed in enumerate(seeds)),
                  key=lambda j: (j[0], "ABCD".index(j[1])))
    if JOBS:                                   # explicit list: one (net, proto, seed) per entry
        jobs = [(0, j.split("|")[1], int(j.split("|")[2]), j.split("|")[0]) for j in JOBS.split(";") if j]
    else:
        jobs = [(i, p, sd, net) for i, p, sd in jobs for net in NETWORKS]
    t0 = time.time()
    for _, proto, seed, net in jobs:
        if True:
            folds, src, ver = make_folds(y, g, off, seed)
            if proto not in folds:
                continue
            if True:
                for k in range(N_FOLDS):
                    key = f"{net}|{proto}|{seed}|{k}"
                    if key in results:
                        continue
                    if time.time() - t0 > TIME_BUDGET_H * 3600:
                        print(f"time budget reached; skipping {key}", flush=True)
                        continue
                    s, v, yt, tests = run_protocol(proto, k, folds, src, ver, y)
                    run_seed = seed * 1000 + k * 10 + zlib.crc32(net.encode()) % 7
                    out = train_eval(net, X, s, v, yt, tests, len(classes), run_seed)
                    for label, m in out.items():
                        kk = key if label == proto else f"{net}|{label}|{seed}|{k}"
                        results[kk] = dict(network=net, protocol=label, seed=seed, fold=k,
                                           n_train=int(len(s)), **m)
                    json.dump(results, open(res_file, "w"), indent=1)
                    print(f"[{(time.time()-t0)/60:6.1f} min] {key}  " +
                          "  ".join(f"{l}: {m['acc']:.2f}" for l, m in out.items()), flush=True)
    if not JOBS:
        write_summary(results, leak)


if __name__ == "__main__":
    main()
