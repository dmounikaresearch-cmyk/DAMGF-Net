# -*- coding: utf-8 -*-
"""
DAMGF-Net -- controlled evaluation pipeline
===========================================

Reproduces the controlled experiments of the paper (Sections 4.7 and 5.6-5.9):

  STEP 1  Image audit: per-source/per-class counts, perceptual-hash (pHash)
          de-duplication and near-duplicate leakage of a random split.
  STEP 2  Feature caching: frozen Swin-Tiny image features and frozen BERT [CLS]
          features for the original and keyword-masked posts.
  EXPORT  Writes the de-duplication lists, data splits and cross-modal pairing
          indices for every seed to data/ (the files released with the paper).
  STEP 3  Frozen-backbone baselines and ablations over 3 seeds and 3 text
          pairings (keyword, keyword-masked, random), test-time interventions
          and adjacency diagnostics (Tables 10, 11 and 13).
  STEP 4  Leave-one-source-out evaluation (Table 12).
  STEP 5  Optional end-to-end fine-tuning (set DAMGF_FINETUNE=1; several hours).
  STEP 6  Computational cost (Table 14).
  STEP 7  Writes RESULTS_FOR_PAPER.md with all tables.

USAGE
  python scripts/download_data.py --out data/MultiHazard      # once
  python src/damgf_experiments.py                             # all steps
  DAMGF_SMOKE=1 python src/damgf_experiments.py               # 2-minute check on synthetic data

ENVIRONMENT VARIABLES
  DAMGF_DATA      root of the downloaded data   (default: data/MultiHazard)
  DAMGF_OUT       output folder for results     (default: results/run)
  DAMGF_EXPORT    folder for split/pairing files (default: data)
  DAMGF_FINETUNE  1 = also run step 5
  DAMGF_SMOKE     1 = synthetic smoke test

Every step is resumable: finished results are stored as JSON in DAMGF_OUT and
skipped on the next run. Steps 1-4 and 6 take about 1-2 h on a Colab T4 GPU.
"""

import os, re, json, math, time, random, itertools, warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             cohen_kappa_score, recall_score, roc_auc_score,
                             confusion_matrix)

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
SMOKE_TEST = os.environ.get("DAMGF_SMOKE", "0") == "1"   # synthetic data, tiny run

# ---- PATHS: defaults match the original notebook (Drive zips unzipped to /content/MultiHazard)
MH_ROOT        = os.environ.get("DAMGF_DATA", "data/MultiHazard")
IMAGE_ROOT     = f"{MH_ROOT}/RemoteSensing"           # <source>/.../<class folder>/<image>
OUT_DIR        = os.environ.get("DAMGF_OUT", "results/run")
EXPORT_DIR     = os.environ.get("DAMGF_EXPORT", "data")
IMAGE_MANIFEST = os.path.join(OUT_DIR, "image_manifest.csv")  # built automatically from IMAGE_ROOT
TEXT_CSV       = f"{MH_ROOT}/SocialMedia/Tweets.csv"
TEXT_COL       = "text"
IOT_CSV        = f"{MH_ROOT}/IoT/Data.csv"
IOT_COLS       = ["aX", "aY", "aZ", "gX", "gY", "gMZ"]
SEIS_CSV       = f"{MH_ROOT}/Seismic/earthquake_alert_balanced_dataset.csv"
SEIS_COLS      = ["magnitude", "depth", "cdi", "mmi", "sig", "alert"]
# Folder-name -> harmonised class, copied from the original notebook (CELL 7)
CLASS_MAP = {"earthquake": "earthquake", "Earthquake": "earthquake",
             "flood": "flood", "Flood": "flood", "Flood Images": "flood", "Water_Disaster": "flood",
             "fire": "fire", "FIRE": "fire", "wildfire": "fire", "Wildfire": "fire", "Wild_Fire": "fire",
             "Urban_Fire": "fire", "Both_smoke_and_fire": "fire", "Smoke_from_fires": "smoke",
             "NON_FIRE": "normal", "nofire": "normal", "Non_Damage_Buildings_Street": "normal",
             "Non_Damage_Wildlife_Forest": "normal"}
TEXT_PREPROCESS_LIKE_PAPER = False  # original notebook feeds raw tweets to BERT (no stop-word removal)
NORMALIZE_AUX  = False              # original notebook uses raw IoT / seismic values
TEXT_POOL_FROM_TRAIN_ONLY = False   # original drew val/test tweets from the TRAIN pool; False = stricter
ALERT_ORDER    = {"green": 0, "yellow": 1, "orange": 2, "red": 3}  # MATCH ORIGINAL

# ---- Experimental settings --------------------------------------------------
CLASSES   = ["earthquake", "flood", "fire", "smoke", "normal"]
SEVERITY  = {"earthquake": 0.95, "flood": 0.90, "fire": 0.80, "smoke": 0.50, "normal": 0.10}
KEYWORDS  = {                                  # copied from the original notebook (CELL 17)
    "earthquake": ["earthquake", "quake", "seismic", "tremor"],
    "flood":      ["flood", "rain", "storm", "water"],
    "fire":       ["fire", "wildfire", "smoke", "burn"],
}
POOL_OF   = {"earthquake": "earthquake", "flood": "flood", "fire": "fire",
             "smoke": "fire", "normal": "normal"}      # smoke images get fire tweets (original)
MASK_REGEX = re.compile(r"\w*(earthquake|quake|seismic|tremor|flood|rain|storm|water|fire|smoke|burn)\w*",
                        re.IGNORECASE)
SEEDS          = [42, 43, 44]       # use 5 seeds if time allows
SPLIT          = (0.70, 0.15, 0.15)
IOT_WINDOW     = 1                  # original: one 6-value reading per sample (no sequence)
HASH_THRESH    = 4                  # pHash Hamming distance <= 4 -> near-duplicate
D_MODEL        = 512
LAMBDA_SEV     = 0.30
LABEL_SMOOTH   = 0.10
DROPOUT        = 0.30
GT_LAYERS, GT_HEADS = 2, 8
HEAD_EPOCHS, HEAD_LR, HEAD_BS, PATIENCE = 30, 1e-3, 256, 5
MODDROP_P      = 0.25               # modality-dropout probability (robustness variant)

VARIANTS = ["image_only", "text_only", "iot_only", "seis_only", "concat",
            "acmaf_only", "static_graph", "full", "full_no_sev", "full_moddrop"]
PAIRINGS = ["keyword", "keyword_masked", "random_text"]

RUN_STEP = {"audit": True, "export": True, "cache": True, "grid": True, "loso": True,
            "finetune": os.environ.get("DAMGF_FINETUNE", "0") == "1", "efficiency": True}
FINETUNE_EPOCHS, FINETUNE_BS, FINETUNE_LR = 10, 16, 1e-4
FINETUNE_SEEDS = [42]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SWIN_NAME = "swin_tiny_patch4_window7_224"
BERT_NAME = "bert-base-uncased"
MOD = ["img", "txt", "iot", "seis"]

if SMOKE_TEST:
    OUT_DIR = "/content/results_smoke" if os.path.isdir("/content") else "./results_smoke"
    EXPORT_DIR = os.path.join(OUT_DIR, "export")
    SEEDS = [42]
    HEAD_EPOCHS = 2
    FINETUNE_EPOCHS = 1
    RUN_STEP["finetune"] = True

os.makedirs(OUT_DIR, exist_ok=True)
CACHE = os.path.join(OUT_DIR, "cache"); os.makedirs(CACHE, exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def save_json(obj, name):
    with open(os.path.join(OUT_DIR, name), "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(name):
    p = os.path.join(OUT_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else None


# =============================================================================
# DATA LOADING
# =============================================================================
def build_manifest_from_folders(root, out_csv=IMAGE_MANIFEST,
                                class_map=None, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    """ROOT/<source>/<class>/<image>  ->  CSV(path,label,source).
    class_map maps folder names to the 5 harmonised labels, e.g.
    {"Wildfire": "fire", "Flood_Images": "flood", "Non_Damage": "normal"}."""
    rows = []
    for src in sorted(os.listdir(root)):
        sdir = os.path.join(root, src)
        if not os.path.isdir(sdir):
            continue
        for cls in sorted(os.listdir(sdir)):
            cdir = os.path.join(sdir, cls)
            if not os.path.isdir(cdir):
                continue
            lab = (class_map or {}).get(cls, cls.lower())
            if lab not in CLASSES:
                print(f"  skipping folder {src}/{cls} (label '{lab}' not in CLASSES)")
                continue
            for dp, _, fs in os.walk(cdir):
                for fn in fs:
                    if fn.lower().endswith(exts):
                        rows.append((os.path.join(dp, fn), lab, src))
    df = pd.DataFrame(rows, columns=["path", "label", "source"])
    df.to_csv(out_csv, index=False)
    print(f"manifest: {len(df)} images, {df.source.nunique()} sources -> {out_csv}")
    return df


def build_manifest_original(root=IMAGE_ROOT, out_csv=IMAGE_MANIFEST):
    """Re-creates the original notebook's image list (CELL 7) and adds the source dataset.
    Source = first folder under RemoteSensing; for 'neutral_images' it is the next folder."""
    from pathlib import Path
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    rows = []
    root = Path(root)
    for folder in root.rglob("*"):
        if folder.is_dir() and folder.name in CLASS_MAP:
            parts = folder.relative_to(root).parts
            src = parts[1] if parts[0] == "neutral_images" and len(parts) > 1 else parts[0]
            for f in folder.iterdir():
                if f.suffix.lower() in exts:
                    rows.append((str(f), f.relative_to(root).as_posix(),
                                 CLASS_MAP[folder.name], src, folder.name))
    # Sort by relative path so that the image order (and therefore every split)
    # is identical on every machine, independent of file-system listing order.
    rows.sort(key=lambda r: r[1])
    df = pd.DataFrame(rows, columns=["path", "image_id", "label", "source", "original_class"])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"manifest: {len(df)} images from {df.source.nunique()} sources -> {out_csv}")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0))
    return df


def make_smoke_data():
    """Tiny synthetic corpus so the whole pipeline can be checked in minutes."""
    from PIL import Image
    d = os.path.join(OUT_DIR, "smoke_data"); os.makedirs(d, exist_ok=True)
    rng = np.random.RandomState(0)
    rows = []
    for src in ["srcA", "srcB", "srcC"]:
        for ci, c in enumerate(CLASSES):
            for k in range(12):
                arr = (rng.rand(64, 64, 3) * 60 + ci * 40).astype(np.uint8)
                p = os.path.join(d, f"{src}_{c}_{k}.png")
                Image.fromarray(arr).save(p)
                rows.append((p, c, src))
    img = pd.DataFrame(rows, columns=["path", "label", "source"])
    words = ["earthquake hits city", "flood water rising", "forest fire spreading",
             "smoke over hills", "nice sunny day", "storm warning", "quake felt"]
    txt = pd.DataFrame({TEXT_COL: [words[i % len(words)] + f" {i}" for i in range(400)]})
    iot = pd.DataFrame(rng.randn(500, 6), columns=IOT_COLS)
    seis = pd.DataFrame(rng.rand(120, 6), columns=SEIS_COLS)
    return img, txt, iot, seis


def load_raw():
    if SMOKE_TEST:
        return make_smoke_data()
    img = pd.read_csv(IMAGE_MANIFEST) if os.path.exists(IMAGE_MANIFEST) else None
    if img is None or len(img) == 0:
        img = build_manifest_original()
    assert len(img) > 0, f"no images found under {IMAGE_ROOT}; run scripts/download_data.py first"
    if "image_id" in img.columns:          # rebuild absolute paths for the current data root
        img["path"] = [os.path.join(IMAGE_ROOT, i) for i in img["image_id"]]
    img["label"] = img["label"].str.lower()
    assert set(img.label) <= set(CLASSES), set(img.label) - set(CLASSES)
    txt = pd.read_csv(TEXT_CSV)
    txt = txt.dropna(subset=[TEXT_COL])
    if TEXT_PREPROCESS_LIKE_PAPER:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        def prep(s):
            s = re.sub(r"https?://\S+|www\.\S+", " ", str(s).lower())
            s = re.sub(r"[^a-z0-9\s]", " ", s)
            return " ".join(t for t in s.split() if t not in ENGLISH_STOP_WORDS)
        txt[TEXT_COL] = txt[TEXT_COL].map(prep)
    txt = txt[txt[TEXT_COL].astype(str).str.len() > 0].reset_index(drop=True)   # original: no de-duplication
    iot = pd.read_csv(IOT_CSV)
    iot = iot.loc[:, ~iot.columns.str.contains("^Unnamed")].dropna(subset=IOT_COLS).reset_index(drop=True)
    seis = pd.read_csv(SEIS_CSV)
    if not pd.api.types.is_numeric_dtype(seis["alert"]):   # text labels green/yellow/orange/red
        seis["alert"] = seis["alert"].str.lower().map(ALERT_ORDER)
    seis = seis.dropna(subset=SEIS_COLS).reset_index(drop=True)
    return img, txt, iot, seis


def split_idx(n, seed, strat=None):
    idx = np.arange(n)
    tr, rest = train_test_split(idx, train_size=SPLIT[0], random_state=seed, stratify=strat)
    rs = None if strat is None else strat[rest]
    va, te = train_test_split(rest, test_size=SPLIT[2] / (SPLIT[1] + SPLIT[2]),
                              random_state=seed, stratify=rs)
    return {"train": tr, "val": va, "test": te}


# =============================================================================
# STEP 1 - IMAGE AUDIT, DE-DUPLICATION, NEAR-DUPLICATE LEAKAGE
# =============================================================================
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_matrix(a, b):
    """a: (n,) uint64, b: (m,) uint64 -> (n,m) Hamming distances."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    return _POP[x.view(np.uint8).reshape(x.shape[0], x.shape[1], 8)].sum(-1)


def phash_all(paths):
    import imagehash
    from PIL import Image
    out = np.zeros(len(paths), dtype=np.uint64)
    bad = []
    for i, p in enumerate(paths):
        try:
            h = imagehash.phash(Image.open(p).convert("RGB"))
            out[i] = np.uint64(int(str(h), 16))
        except Exception:
            bad.append(i)
        if i % 5000 == 0:
            print(f"  hashed {i}/{len(paths)}")
    return out, bad


def step_audit(img):
    print("\n=== STEP 1: image audit ===")
    hp = os.path.join(CACHE, "phash.npy")
    if os.path.exists(hp):
        h = np.load(hp); bad = load_json("unreadable.json") or []
    else:
        h, bad = phash_all(img.path.tolist()); np.save(hp, h); save_json(bad, "unreadable.json")

    res = {"n_images_before": int(len(img)), "unreadable": len(bad)}
    res["per_source_class_before"] = img.groupby(["source", "label"]).size().unstack(fill_value=0).to_dict()

    # (a) leakage of the ORIGINAL protocol: random split of the non-deduplicated corpus
    y = img.label.map(CLASSES.index).values
    sp = split_idx(len(img), 42, y)
    tr_h, te_h = h[sp["train"]], h[sp["test"]]
    min_d = np.full(len(te_h), 64)
    for s in range(0, len(te_h), 512):
        min_d[s:s + 512] = hamming_matrix(te_h[s:s + 512], tr_h).min(1)
    res["orig_split_test_with_exact_dup_in_train"] = int((min_d == 0).sum())
    res["orig_split_test_with_near_dup_in_train"] = int((min_d <= HASH_THRESH).sum())
    res["orig_split_test_size"] = int(len(te_h))
    res["orig_split_near_dup_fraction"] = float((min_d <= HASH_THRESH).mean())
    te_lab = img.label.values[sp["test"]]
    res["orig_split_near_dup_fraction_per_class"] = {
        c: float((min_d[te_lab == c] <= HASH_THRESH).mean()) if (te_lab == c).any() else None
        for c in CLASSES}

    # (b) de-duplicate the corpus (keep first occurrence of each near-duplicate group)
    keep = np.ones(len(h), bool); keep[bad] = False
    dup_of = np.full(len(h), -1); dup_dist = np.full(len(h), -1)
    for s in range(0, len(h), 512):
        blk = hamming_matrix(h[s:s + 512], h[:s + 512])
        for r in range(blk.shape[0]):
            i = s + r
            if not keep[i]:
                continue
            prev = np.where(blk[r, :i] <= HASH_THRESH)[0]
            if len(prev) and keep[prev].any():
                keep[i] = False
                j = prev[keep[prev]][0]             # first kept image it duplicates
                dup_of[i], dup_dist[i] = j, blk[r, j]
    ded = img[keep].reset_index(drop=True)
    ded.to_csv(os.path.join(OUT_DIR, "image_manifest_dedup.csv"), index=False)
    # Release files: pHash of every image and the list of removed near-duplicates
    ids = img["image_id"] if "image_id" in img.columns else img["path"]
    os.makedirs(os.path.join(EXPORT_DIR, "dedup"), exist_ok=True)
    pd.DataFrame({"image_id": ids, "source": img.source, "label": img.label,
                  "phash": [f"{int(x):016x}" for x in h], "kept": keep}
                 ).to_csv(os.path.join(EXPORT_DIR, "dedup", "phash.csv"), index=False)
    rem = np.where(~keep)[0]
    pd.DataFrame({"removed_image_id": ids.values[rem], "removed_source": img.source.values[rem],
                  "duplicate_of_image_id": [ids.values[j] if j >= 0 else "" for j in dup_of[rem]],
                  "duplicate_of_source": [img.source.values[j] if j >= 0 else "" for j in dup_of[rem]],
                  "hamming_distance": dup_dist[rem],
                  "reason": ["unreadable" if i in set(bad) else "near_duplicate" for i in rem]}
                 ).to_csv(os.path.join(EXPORT_DIR, "dedup", "removed.csv"), index=False)
    np.save(os.path.join(CACHE, "phash_dedup.npy"), h[keep])
    res["n_images_after_dedup"] = int(len(ded))
    res["n_removed_as_near_duplicates"] = int(len(img) - len(ded) - len(bad))
    res["per_class_after"] = ded.label.value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict()
    res["per_source_class_after"] = ded.groupby(["source", "label"]).size().unstack(fill_value=0).to_dict()
    res["cross_source_near_dups_removed"] = None  # filled below
    # cross-source duplicates: removed images whose kept twin is from another source
    removed = np.where(~keep)[0]
    cross = 0
    for i in removed:
        if i in bad:
            continue
        d = hamming_matrix(h[i:i + 1], h[:i])[0]
        j = np.where((d <= HASH_THRESH) & keep[:i])[0]
        if len(j) and img.source.values[j[0]] != img.source.values[i]:
            cross += 1
    res["cross_source_near_dups_removed"] = int(cross)
    save_json(res, "audit.json")
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))
    example_grid(ded)
    return ded


def example_grid(ded, per=4):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(len(CLASSES), per, figsize=(per * 2.2, len(CLASSES) * 2.2))
    for r, c in enumerate(CLASSES):
        sub = ded[ded.label == c]
        pick = sub.sample(min(per, len(sub)), random_state=rng) if len(sub) else sub
        for k in range(per):
            a = ax[r, k]; a.axis("off")
            if k < len(pick):
                a.imshow(Image.open(pick.path.values[k]).convert("RGB").resize((224, 224)))
                a.set_title(f"{c}\n[{pick.source.values[k]}]", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_examples_per_class.png"), dpi=200)
    plt.close()


# =============================================================================
# EXPORT - splits, pairing indices and class mapping released with the paper
# =============================================================================
def step_export(ded, txt, iot_np, seis_np):
    """Write, for every seed, the image splits and the cross-modal pairing indices
    exactly as they are built inside Bundle (same functions, same seeds)."""
    print("\n=== EXPORT: splits and pairing indices ===")
    ids = ded["image_id"].values if "image_id" in ded.columns else ded["path"].values
    y = ded.label.map(CLASSES.index).values
    sev = ded.label.map(SEVERITY).values
    pools = text_pools(txt)
    os.makedirs(os.path.join(EXPORT_DIR, "class_mappings"), exist_ok=True)
    pd.DataFrame(sorted(CLASS_MAP.items()), columns=["source_folder", "harmonised_class"]).to_csv(
        os.path.join(EXPORT_DIR, "class_mappings", "class_mapping.csv"), index=False)
    pd.DataFrame([{"class": c, "keywords": " ".join(KEYWORDS.get(POOL_OF[c], [])) or "(random post)",
                   "pool_size": int(len(pools[c])), "severity_target": SEVERITY[c]} for c in CLASSES]
                 ).to_csv(os.path.join(EXPORT_DIR, "class_mappings", "text_pairing_keywords.csv"), index=False)
    global TEXT_TRAIN_SPLIT
    for seed in SEEDS:
        sdir = os.path.join(EXPORT_DIR, "splits", f"seed{seed}")
        pdir = os.path.join(EXPORT_DIR, "pairing", f"seed{seed}")
        os.makedirs(sdir, exist_ok=True); os.makedirs(pdir, exist_ok=True)
        img_sp = split_idx(len(ded), seed, y)
        tsp = split_idx(len(txt), seed)
        TEXT_TRAIN_SPLIT = tsp["train"]
        isp = split_idx(len(iot_np) - IOT_WINDOW + 1, seed)
        ssp = split_idx(len(seis_np), seed)
        for k, part in img_sp.items():
            pd.DataFrame({"image_id": ids[part], "source": ded.source.values[part],
                          "label": ded.label.values[part]}).to_csv(os.path.join(sdir, f"{k}.csv"), index=False)
            # keyword-masked pairing uses the keyword indices with masked text features
            for pairing in ["keyword", "random_text"]:
                t, i, s = make_pairs(ded.label.values, part, tsp[k], isp[k], ssp[k], pools, pairing,
                                     seed + {"train": 0, "val": 1, "test": 2}[k])
                pd.DataFrame({"image_id": ids[part], "label": ded.label.values[part],
                              "post_row": t, "sensor_row": i, "seismic_row": s,
                              "severity_target": sev[part]}
                             ).to_csv(os.path.join(pdir, f"{pairing}_{k}.csv"), index=False)
        print(f"  seed {seed}: " + ", ".join(f"{k}={len(v)}" for k, v in img_sp.items()))
    # leave-one-source-out splits (seed 42), as built in step_loso
    for src in sorted(ded.source.unique()):
        te = np.where(ded.source.values == src)[0]
        rest = np.where(ded.source.values != src)[0]
        if set(ded.label.values[te]) - set(ded.label.values[rest]):
            continue
        tr, va = train_test_split(rest, test_size=0.15, random_state=42, stratify=y[rest])
        ldir = os.path.join(EXPORT_DIR, "splits", "loso", src)
        os.makedirs(ldir, exist_ok=True)
        for k, part in {"train": tr, "val": va, "test": te}.items():
            pd.DataFrame({"image_id": ids[part], "source": ded.source.values[part],
                          "label": ded.label.values[part]}).to_csv(os.path.join(ldir, f"{k}.csv"), index=False)
    print(f"  written to {EXPORT_DIR}/")


# =============================================================================
# STEP 2 - FEATURE CACHING (frozen pretrained backbones)
# =============================================================================
def image_tf(train=False):
    from torchvision import transforms as T
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return T.Compose([T.Resize((256, 256)), T.RandomCrop(224), T.RandomHorizontalFlip(),
                          T.RandomRotation(10), T.ColorJitter(.2, .2, .2), T.ToTensor(), norm])
    return T.Compose([T.Resize((256, 256)), T.CenterCrop(224), T.ToTensor(), norm])


class ImgDS(Dataset):
    def __init__(self, paths, tf):
        self.p, self.tf = list(paths), tf
    def __len__(self):
        return len(self.p)
    def __getitem__(self, i):
        from PIL import Image
        return self.tf(Image.open(self.p[i]).convert("RGB"))


def make_swin(pretrained=True):
    import timm
    return timm.create_model(SWIN_NAME, pretrained=pretrained and not SMOKE_TEST, num_classes=0)


def make_bert():
    from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel
    tok = AutoTokenizer.from_pretrained(BERT_NAME)
    if SMOKE_TEST:
        return tok, BertModel(BertConfig(num_hidden_layers=1, num_attention_heads=12))
    return tok, AutoModel.from_pretrained(BERT_NAME)


def mask_text(s):
    return MASK_REGEX.sub("[MASK]", s)


@torch.no_grad()
def step_cache(ded, txt):
    print("\n=== STEP 2: caching frozen features ===")
    fi = os.path.join(CACHE, "img_feat.npy")
    if not os.path.exists(fi):
        m = make_swin().to(DEVICE).eval()
        dl = DataLoader(ImgDS(ded.path, image_tf(False)), batch_size=64, num_workers=2)
        feats = []
        for k, x in enumerate(dl):
            with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                feats.append(m(x.to(DEVICE)).float().cpu())
            if k % 50 == 0:
                print(f"  images {k * 64}/{len(ded)}")
        np.save(fi, torch.cat(feats).numpy()); del m
    for tag, fn in [("orig", lambda s: s), ("masked", mask_text)]:
        ft = os.path.join(CACHE, f"txt_feat_{tag}.npy")
        if os.path.exists(ft):
            continue
        tok, bert = make_bert(); bert = bert.to(DEVICE).eval()
        texts = [fn(str(s)) for s in txt[TEXT_COL]]
        out = []
        for s in range(0, len(texts), 128):
            enc = tok(texts[s:s + 128], padding=True, truncation=True, max_length=128,
                      return_tensors="pt").to(DEVICE)
            with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                out.append(bert(**enc).last_hidden_state[:, 0].float().cpu())
        np.save(ft, torch.cat(out).numpy()); del bert
        print(f"  text features '{tag}' done")
    return (np.load(fi), np.load(os.path.join(CACHE, "txt_feat_orig.npy")),
            np.load(os.path.join(CACHE, "txt_feat_masked.npy")))


# =============================================================================
# PAIRING (Section 4.2 of the paper)
# =============================================================================
def text_pools(txt):
    low = txt[TEXT_COL].astype(str).str.lower()
    pools = {c: np.where(low.str.contains("|".join(kw)).values)[0] for c, kw in KEYWORDS.items()}
    pools["normal"] = np.arange(len(txt))          # normal -> random post (as in original)
    return {c: pools[POOL_OF[c]] for c in CLASSES}


def make_pairs(img_labels, img_split, txt_split, iot_split, seis_split, pools, pairing, seed):
    """Return per-image indices into text/IoT/seismic for one split.
    pairing: keyword | keyword_masked (same indices, masked features) | random_text"""
    rng = np.random.RandomState(seed)
    t_set = set((TEXT_TRAIN_SPLIT if TEXT_POOL_FROM_TRAIN_ONLY else txt_split).tolist())
    pool_s = {c: np.array([i for i in p if i in t_set]) for c, p in pools.items()}
    t_idx = np.empty(len(img_split), int)
    for k, i in enumerate(img_split):
        c = img_labels[i]
        if pairing == "random_text" or len(pool_s[c]) == 0:
            t_idx[k] = rng.choice(txt_split)
        elif TEXT_POOL_FROM_TRAIN_ONLY:
            t_idx[k] = rng.choice(pool_s[c])
        else:
            t_idx[k] = rng.choice(pool_s[c])
    i_idx = rng.choice(iot_split, len(img_split))      # independent of label (as in paper)
    s_idx = rng.choice(seis_split, len(img_split))
    return t_idx, i_idx, s_idx


TEXT_TRAIN_SPLIT = None   # set by Bundle when TEXT_POOL_FROM_TRAIN_ONLY


def iot_tensor(iot_np, idx):
    """(N, IOT_WINDOW, 6): IOT_WINDOW consecutive rows starting at idx."""
    T = IOT_WINDOW
    idx = np.minimum(idx, len(iot_np) - T)
    return np.stack([iot_np[i:i + T] for i in idx]).astype(np.float32)


# =============================================================================
# MODEL (same components as the paper; backbones optional)
# =============================================================================
class IoTEncoder(nn.Module):
    """Original notebook (CELL 13): the 6 values of one reading are treated as a
    length-6, 1-channel sequence -> Conv1d x2 -> BiLSTM -> last step -> projection."""
    def __init__(self, d=D_MODEL, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 64, 3, padding=1), nn.BatchNorm1d(64), nn.GELU(),
                                  nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU())
        self.lstm = nn.LSTM(128, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(2 * hidden, 1024), nn.LayerNorm(1024), nn.GELU(),
                                  nn.Dropout(0.30), nn.Linear(1024, d))
    def forward(self, x):                       # x: (B, 1, 6)
        x = x.reshape(x.size(0), -1)[:, :6].unsqueeze(1)
        h = self.conv(x).permute(0, 2, 1)
        out, _ = self.lstm(h)
        return self.proj(out[:, -1])


class SeisEncoder(nn.Module):
    def __init__(self, d=D_MODEL):              # original CELL 15
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.30),
                                 nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, d))
    def forward(self, x):
        return self.net(x)


def proj_head(din, d=D_MODEL):                  # original image/text projection heads
    return nn.Sequential(nn.Linear(din, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.30),
                         nn.Linear(1024, d))


class ACMAF(nn.Module):
    def __init__(self, d=D_MODEL):              # original CELL 14
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 1))
        self.proj = nn.Sequential(nn.Linear(4 * d, 1024), nn.LayerNorm(1024), nn.GELU(),
                                  nn.Dropout(0.30), nn.Linear(1024, d))
    def forward(self, Z):                       # Z: (B,4,d)
        alpha = self.score(Z).squeeze(-1).softmax(-1)          # Eqs. 5-6
        z = self.proj((alpha.unsqueeze(-1) * Z).flatten(1))   # Eq. 7
        return z, alpha


class GraphModule(nn.Module):
    """Original CELL 15: A = softmax(QK^T/sqrt(d)), H = A V (V = Z W_V), 2-layer transformer
    (8 heads, FFN 2048, dropout 0.1), mean pooling.
    dynamic=False: one learned static 4x4 adjacency shared by all samples (ablation)."""
    def __init__(self, dynamic=True, d=D_MODEL):
        super().__init__()
        self.dynamic, self.d = dynamic, d
        self.Wq, self.Wk, self.Wv = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        if not dynamic:
            self.A_logits = nn.Parameter(torch.zeros(4, 4))
        layer = nn.TransformerEncoderLayer(d, GT_HEADS, 2048, 0.1, activation="gelu", batch_first=True)
        self.tf = nn.TransformerEncoder(layer, GT_LAYERS)
    def forward(self, Z):
        if self.dynamic:
            A = (self.Wq(Z) @ self.Wk(Z).transpose(1, 2) / math.sqrt(self.d)).softmax(-1)
        else:
            A = self.A_logits.softmax(-1).unsqueeze(0).expand(Z.size(0), 4, 4)
        H = self.tf(A @ self.Wv(Z))
        return H.mean(1), A


class FusionModel(nn.Module):
    """variant in VARIANTS. If backbones=True the image/text inputs are raw
    (pixels / token ids) and Swin-T / BERT are fine-tuned; otherwise inputs
    are cached 768-d backbone features."""
    def __init__(self, variant, backbones=False, img_dim=768, txt_dim=768):
        super().__init__()
        self.variant, self.backbones = variant, backbones
        d = D_MODEL
        if backbones:
            self.swin = make_swin()
            if variant != "image_only":
                _, self.bert = make_bert()
        self.p_img = proj_head(img_dim)
        if variant != "image_only":
            self.p_txt = proj_head(txt_dim)
            self.e_iot, self.e_seis = IoTEncoder(), SeisEncoder()
        if variant == "concat":
            self.cat = nn.Sequential(nn.Linear(4 * d, 1024), nn.LayerNorm(1024), nn.GELU(),
                                     nn.Dropout(0.30), nn.Linear(1024, d))
        if variant.startswith("full") or variant in ("acmaf_only", "static_graph"):
            self.acmaf = ACMAF()
        if variant.startswith("full") or variant == "static_graph":
            self.graph = GraphModule(dynamic=(variant != "static_graph"))
        self.shared = nn.Sequential(nn.Linear(d, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(DROPOUT))
        self.cls = nn.Linear(256, len(CLASSES))
        self.sev = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))   # no sigmoid (original)

    def embed(self, img, txt, iot, seis):
        if self.backbones:
            img = self.swin(img)
            if txt is not None and hasattr(self, "bert"):
                txt = self.bert(input_ids=txt[0], attention_mask=txt[1]).last_hidden_state[:, 0]
        e = {"img": self.p_img(img)}
        if self.variant != "image_only":
            e["txt"] = self.p_txt(txt)
            e["iot"] = self.e_iot(iot)
            e["seis"] = self.e_seis(seis)
        return e

    def forward(self, img, txt, iot, seis, drop=(), moddrop_p=0.0):
        e = self.embed(img, txt, iot, seis)
        aux = {}
        single = {"image_only": "img", "text_only": "txt", "iot_only": "iot", "seis_only": "seis"}
        if self.variant in single:
            z = e[single[self.variant]]
        else:
            Z = torch.stack([e[m] for m in MOD], 1)                  # (B,4,d) Eq. 8
            keep = torch.ones(Z.shape[:2], device=Z.device)
            for m in drop:                                           # test-time removal
                keep[:, MOD.index(m)] = 0
            if moddrop_p > 0 and self.training:                      # train-time dropout
                r = (torch.rand_like(keep) > moddrop_p).float()
                r[r.sum(1) == 0, 0] = 1                              # keep >=1 modality
                keep = keep * r
            Z = Z * keep.unsqueeze(-1)
            if self.variant == "concat":
                z = self.cat(Z.flatten(1))
            else:
                z, aux["alpha"] = self.acmaf(Z)
                if hasattr(self, "graph"):
                    zg, aux["A"] = self.graph(Z)
                    z = z + zg                                       # Eq. 17
        h = self.shared(z)
        return self.cls(h), self.sev(h).squeeze(-1), aux


def loss_fn(variant, logits, sev, y, s):
    l = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTH)
    if variant != "full_no_sev":
        l = l + LAMBDA_SEV * F.smooth_l1_loss(sev, s)
    return l


# =============================================================================
# METRICS
# =============================================================================
def metrics(y, p, prob=None):
    m = {"acc": accuracy_score(y, p),
         # macro-F1 over the classes present in the test set (matters for leave-one-source-out)
         "macro_f1": f1_score(y, p, average="macro", labels=sorted(set(np.asarray(y).tolist())), zero_division=0),
         "weighted_f1": f1_score(y, p, average="weighted", zero_division=0),
         "mcc": matthews_corrcoef(y, p), "kappa": cohen_kappa_score(y, p)}
    rec = recall_score(y, p, average=None, labels=list(range(len(CLASSES))), zero_division=0)
    for c, r in zip(CLASSES, rec):
        m[f"recall_{c}"] = float(r)
    if prob is not None:
        try:
            present = sorted(set(y))
            if len(present) > 1:
                m["mean_auc_ovr"] = roc_auc_score(np.eye(len(CLASSES))[y][:, present],
                                                  prob[:, present], average="macro")
        except Exception:
            pass
    return {k: float(v) for k, v in m.items()}


# =============================================================================
# STEP 3 - FROZEN-BACKBONE ABLATION GRID
# =============================================================================
class Bundle:
    """Holds all tensors for one (seed, pairing) setting, per split."""
    def __init__(self, ded, txt, iot_np, seis_np, F_img, F_txt_o, F_txt_m, pools,
                 pairing, seed, img_split=None):
        y = ded.label.map(CLASSES.index).values
        sev = ded.label.map(SEVERITY).values.astype(np.float32)
        self.img_split = img_split or split_idx(len(ded), seed, y)
        tsp = split_idx(len(txt), seed)
        global TEXT_TRAIN_SPLIT
        TEXT_TRAIN_SPLIT = tsp["train"]
        isp = split_idx(len(iot_np) - IOT_WINDOW + 1, seed)
        ssp = split_idx(len(seis_np), seed)
        F_txt = F_txt_m if pairing == "keyword_masked" else F_txt_o
        self.data = {}
        for k, part in self.img_split.items():
            t, i, s = make_pairs(ded.label.values, part, tsp[k], isp[k], ssp[k], pools,
                                 "keyword" if pairing == "keyword_masked" else pairing,
                                 seed + {"train": 0, "val": 1, "test": 2}[k])
            self.data[k] = TensorDataset(
                torch.tensor(F_img[part]), torch.tensor(F_txt[t]),
                torch.tensor(iot_tensor(iot_np, i)), torch.tensor(seis_np[s]),
                torch.tensor(y[part]), torch.tensor(sev[part]))


def run_epoch(model, dl, opt=None, variant="full", moddrop=0.0, drop=(), shuffle_aux=False):
    train = opt is not None
    model.train(train)
    tot, n, Y, P, PR, As, Al = 0.0, 0, [], [], [], [], []
    for xi, xt, xo, xs, y, s in dl:
        xi, xt, xo, xs, y, s = [a.to(DEVICE) for a in (xi, xt, xo, xs, y, s)]
        if shuffle_aux:                       # break any IoT/seismic <-> sample association
            perm = torch.randperm(len(y), device=DEVICE)
            xo, xs = xo[perm], xs[perm]
        with torch.set_grad_enabled(train):
            lo, sv, aux = model(xi, xt, xo, xs, drop=drop, moddrop_p=moddrop)
            loss = loss_fn(variant, lo, sv, y, s)
        if train:
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot += loss.item() * len(y); n += len(y)
        if not train:
            Y.append(y.cpu()); P.append(lo.argmax(1).cpu()); PR.append(lo.softmax(1).cpu())
            if "A" in aux: As.append(aux["A"].detach().cpu())
            if "alpha" in aux: Al.append(aux["alpha"].detach().cpu())
    out = {"loss": tot / max(n, 1)}
    if not train:
        out.update(y=torch.cat(Y).numpy(), p=torch.cat(P).numpy(), prob=torch.cat(PR).numpy(),
                   A=torch.cat(As).numpy() if As else None,
                   alpha=torch.cat(Al).numpy() if Al else None)
    return out


def train_head(variant, bundle, seed):
    set_seed(seed)
    model = FusionModel(variant).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=HEAD_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, HEAD_EPOCHS)
    g = torch.Generator().manual_seed(seed)
    dl_tr = DataLoader(bundle.data["train"], HEAD_BS, shuffle=True, generator=g, drop_last=True)
    dl_va = DataLoader(bundle.data["val"], 1024)
    md = MODDROP_P if variant == "full_moddrop" else 0.0
    best, bad, state, hist = 1e9, 0, None, []
    for ep in range(HEAD_EPOCHS):
        tr = run_epoch(model, dl_tr, opt, variant, md)
        va = run_epoch(model, dl_va, None, variant)
        sched.step()
        hist.append((ep + 1, tr["loss"], va["loss"], float((va["y"] == va["p"]).mean())))
        if va["loss"] < best - 1e-4:
            best, bad = va["loss"], 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(state)
    return model, hist


def adjacency_diag(A, alpha):
    """A: (N,4,4) per-sample adjacency. Entropy of uniform row = ln4 = 1.386."""
    if A is None:
        return None
    ent = -(A * np.log(A + 1e-12)).sum(-1)                    # (N,4)
    d = {"row_entropy_mean": float(ent.mean()), "row_entropy_std": float(ent.std()),
         "row_entropy_uniform": float(np.log(4)),
         "frac_rows_max_gt_0.9": float((A.max(-1) > 0.9).mean()),
         "frac_rows_max_gt_0.99": float((A.max(-1) > 0.99).mean()),
         "mean_A": A.mean(0).round(4).tolist(), "std_A": A.std(0).round(4).tolist(),
         "row_entropy_by_source": dict(zip(MOD, ent.mean(0).round(4).tolist())),
         "argmax_target_freq": {MOD[i]: np.bincount(A[:, i].argmax(-1), minlength=4).tolist()
                                for i in range(4)}}
    if alpha is not None:
        d["alpha_mean"] = dict(zip(MOD, alpha.mean(0).round(4).tolist()))
        d["alpha_std"] = dict(zip(MOD, alpha.std(0).round(4).tolist()))
    return d


def step_grid(ded, txt, iot_np, seis_np, F_img, F_to, F_tm):
    print("\n=== STEP 3: frozen-backbone ablation grid ===")
    pools = text_pools(txt)
    save_json({c: int(len(p)) for c, p in pools.items()}, "text_pool_sizes.json")
    results = load_json("grid.json") or {}
    for seed in SEEDS:
        for pairing in PAIRINGS:
            b = Bundle(ded, txt, iot_np, seis_np, F_img, F_to, F_tm, pools, pairing, seed)
            dl_te = DataLoader(b.data["test"], 1024)
            for variant in VARIANTS:
                key = f"{variant}|{pairing}|{seed}"
                if key in results:
                    continue
                if pairing != "keyword" and variant in ("iot_only", "seis_only", "image_only"):
                    continue    # unaffected by text pairing
                t0 = time.time()
                model, hist = train_head(variant, b, seed)
                te = run_epoch(model, dl_te, None, variant)
                r = {"test": metrics(te["y"], te["p"], te["prob"]), "epochs": hist,
                     "params_head": sum(p.numel() for p in model.parameters()),
                     "confusion": confusion_matrix(te["y"], te["p"],
                                                   labels=list(range(len(CLASSES)))).tolist()}
                if variant.startswith("full") or variant in ("acmaf_only", "static_graph"):
                    r["adjacency"] = adjacency_diag(te["A"], te["alpha"])
                if variant in ("full", "full_moddrop"):
                    ctrl = {}
                    set_seed(seed)
                    s = run_epoch(model, dl_te, None, variant, shuffle_aux=True)
                    ctrl["iot_seis_shuffled"] = metrics(s["y"], s["p"])
                    for m in MOD:
                        o = run_epoch(model, dl_te, None, variant, drop=(m,))
                        ctrl[f"drop_{m}"] = metrics(o["y"], o["p"])
                    o = run_epoch(model, dl_te, None, variant, drop=("iot", "seis"))
                    ctrl["drop_iot_seis"] = metrics(o["y"], o["p"])
                    o = run_epoch(model, dl_te, None, variant, drop=("txt",  "iot", "seis"))
                    ctrl["image_node_only"] = metrics(o["y"], o["p"])
                    r["controls"] = ctrl
                results[key] = r
                save_json(results, "grid.json")
                print(f"  {key:40s} acc={r['test']['acc']:.4f} macroF1={r['test']['macro_f1']:.4f}"
                      f"  ({time.time() - t0:.0f}s)")
    return results


# =============================================================================
# STEP 4 - LEAVE-ONE-SOURCE-OUT
# =============================================================================
def step_loso(ded, txt, iot_np, seis_np, F_img, F_to, F_tm, seed=42):
    print("\n=== STEP 4: leave-one-source-out ===")
    pools = text_pools(txt)
    results = load_json("loso.json") or {}
    y = ded.label.map(CLASSES.index).values
    for src in sorted(ded.source.unique()):
        te = np.where(ded.source.values == src)[0]
        rest = np.where(ded.source.values != src)[0]
        missing = set(ded.label.values[te]) - set(ded.label.values[rest])
        if missing:
            print(f"  skip {src}: classes {missing} occur only in this source")
            results[src] = {"skipped": f"classes {sorted(missing)} only in this source"}
            continue
        tr, va = train_test_split(rest, test_size=0.15, random_state=seed, stratify=y[rest])
        sp = {"train": tr, "val": va, "test": te}
        for pairing in ["keyword", "keyword_masked"]:
            b = Bundle(ded, txt, iot_np, seis_np, F_img, F_to, F_tm, pools, pairing, seed, sp)
            for variant in ["image_only", "full"]:
                key = f"{src}|{variant}|{pairing}"
                if key in results or (variant == "image_only" and pairing != "keyword"):
                    continue
                model, _ = train_head(variant, b, seed)
                o = run_epoch(model, DataLoader(b.data["test"], 1024), None, variant)
                results[key] = {"n_test": int(len(te)),
                                "classes": sorted(set(ded.label.values[te])),
                                "test": metrics(o["y"], o["p"])}
                save_json(results, "loso.json")
                print(f"  {key:40s} acc={results[key]['test']['acc']:.4f}")
    return results


# =============================================================================
# STEP 5 - OPTIONAL END-TO-END FINE-TUNING
# =============================================================================
class E2EDS(Dataset):
    def __init__(self, paths, texts, iot, seis, y, s, tf, tok):
        self.paths, self.texts, self.iot, self.seis = list(paths), list(texts), iot, seis
        self.y, self.s, self.tf, self.tok = y, s, tf, tok
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        from PIL import Image
        return (self.tf(Image.open(self.paths[i]).convert("RGB")), self.texts[i],
                self.iot[i], self.seis[i], int(self.y[i]), float(self.s[i]))


def step_finetune(ded, txt, iot_np, seis_np):
    global TEXT_TRAIN_SPLIT
    print("\n=== STEP 5: end-to-end fine-tuning ===")
    tok, _ = make_bert()
    pools = text_pools(txt)
    y = ded.label.map(CLASSES.index).values
    sev = ded.label.map(SEVERITY).values.astype(np.float32)
    results = load_json("finetune.json") or {}

    def collate(batch):
        im, tx, io, se, yy, ss = zip(*batch)
        enc = tok(list(tx), padding=True, truncation=True, max_length=128, return_tensors="pt")
        return (torch.stack(im), (enc["input_ids"], enc["attention_mask"]),
                torch.tensor(np.stack(io)), torch.tensor(np.stack(se)),
                torch.tensor(yy), torch.tensor(ss, dtype=torch.float32))

    for seed in FINETUNE_SEEDS:
        sp = split_idx(len(ded), seed, y)
        tsp, isp, ssp = (split_idx(len(txt), seed), split_idx(len(iot_np) - IOT_WINDOW + 1, seed),
                         split_idx(len(seis_np), seed))
        TEXT_TRAIN_SPLIT = tsp["train"]
        dls = {}
        for k in sp:
            t, i, s = make_pairs(ded.label.values, sp[k], tsp[k], isp[k], ssp[k], pools, "keyword",
                                 seed + {"train": 0, "val": 1, "test": 2}[k])
            texts = [mask_text(str(x)) for x in txt[TEXT_COL].values[t]]
            ds = E2EDS(ded.path.values[sp[k]], texts, iot_tensor(iot_np, i), seis_np[s],
                       y[sp[k]], sev[sp[k]], image_tf(k == "train"), tok)
            dls[k] = DataLoader(ds, FINETUNE_BS, shuffle=(k == "train"), num_workers=2,
                                collate_fn=collate, drop_last=(k == "train"))
        for variant in ["image_only", "full"]:
            key = f"{variant}|keyword_masked|{seed}"
            if key in results:
                continue
            set_seed(seed)
            model = FusionModel(variant, backbones=True).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FINETUNE_EPOCHS, eta_min=1e-6)
            scaler = torch.cuda.amp.GradScaler(enabled=DEVICE == "cuda")
            best, bad, state = 1e9, 0, None

            def ev(dl, drop=()):
                model.eval(); Y, P, L = [], [], 0.0
                with torch.no_grad():
                    for im, tx, io, se, yy, ss in dl:
                        im, io, se, yy, ss = [a.to(DEVICE) for a in (im, io, se, yy, ss)]
                        tx = (tx[0].to(DEVICE), tx[1].to(DEVICE))
                        with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                            lo, sv, _ = model(im, tx, io, se, drop=drop)
                        L += loss_fn(variant, lo.float(), sv.float(), yy, ss).item() * len(yy)
                        Y.append(yy.cpu()); P.append(lo.argmax(1).cpu())
                Y, P = torch.cat(Y).numpy(), torch.cat(P).numpy()
                return L / len(Y), Y, P

            for ep in range(FINETUNE_EPOCHS):
                model.train()
                for im, tx, io, se, yy, ss in dls["train"]:
                    im, io, se, yy, ss = [a.to(DEVICE) for a in (im, io, se, yy, ss)]
                    tx = (tx[0].to(DEVICE), tx[1].to(DEVICE))
                    with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                        lo, sv, _ = model(im, tx, io, se)
                        loss = loss_fn(variant, lo.float(), sv.float(), yy, ss)
                    opt.zero_grad(); scaler.scale(loss).backward()
                    scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                sched.step()
                vl, _, _ = ev(dls["val"])
                print(f"  {key} epoch {ep + 1} val_loss={vl:.4f}")
                if vl < best - 1e-4:
                    best, bad = vl, 0
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
            model.load_state_dict(state)
            _, Y, P = ev(dls["test"])
            r = {"test": metrics(Y, P)}
            if variant == "full":
                _, Y2, P2 = ev(dls["test"], drop=("txt", "iot", "seis"))
                r["image_node_only"] = metrics(Y2, P2)
            results[key] = r
            save_json(results, "finetune.json")
            print(f"  {key}: acc={r['test']['acc']:.4f}")
            del model; torch.cuda.empty_cache()
    return results


# =============================================================================
# STEP 6 - EFFICIENCY
# =============================================================================
@torch.no_grad()
def step_efficiency():
    print("\n=== STEP 6: efficiency ===")
    res = {"device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
           "torch": torch.__version__}
    tok, _ = make_bert()
    for variant in ["image_only", "full"]:
        m = FusionModel(variant, backbones=True).to(DEVICE).eval()
        r = {"params": int(sum(p.numel() for p in m.parameters()))}
        for bs in [1, 16]:
            im = torch.randn(bs, 3, 224, 224, device=DEVICE)
            enc = tok(["flood water rising in the city centre"] * bs, padding="max_length",
                      max_length=128, return_tensors="pt").to(DEVICE)
            tx = (enc["input_ids"], enc["attention_mask"])
            io = torch.randn(bs, IOT_WINDOW, 6, device=DEVICE)
            se = torch.randn(bs, 6, device=DEVICE)
            for _ in range(10):
                m(im, tx, io, se)
            if DEVICE == "cuda":
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            ts = []
            for _ in range(50):
                t0 = time.perf_counter(); m(im, tx, io, se)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            r[f"bs{bs}_batch_ms_mean"] = 1000 * float(np.mean(ts))
            r[f"bs{bs}_batch_ms_std"] = 1000 * float(np.std(ts))
            r[f"bs{bs}_per_sample_ms"] = 1000 * float(np.mean(ts)) / bs
            r[f"bs{bs}_throughput_per_s"] = bs / float(np.mean(ts))
            if DEVICE == "cuda":
                r[f"bs{bs}_peak_mem_MB"] = torch.cuda.max_memory_allocated() / 2 ** 20
            if bs == 1:
                try:
                    from fvcore.nn import FlopCountAnalysis
                    fca = FlopCountAnalysis(m, (im, tx, io, se))
                    fca.unsupported_ops_warnings(False); fca.uncalled_modules_warnings(False)
                    r["GFLOPs_per_sample"] = fca.total() / 1e9   # fvcore counts 1 MAC = 1 flop
                except Exception as ex:
                    r["GFLOPs_per_sample"] = f"failed: {ex}"
        res[variant] = r
        print(" ", variant, json.dumps(r, default=str))
        del m; torch.cuda.empty_cache() if DEVICE == "cuda" else None
    save_json(res, "efficiency.json")
    return res


# =============================================================================
# STEP 7 - SUMMARY FOR THE PAPER
# =============================================================================
def ms(vals, pct=True, d=2):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    f = 100 if pct else 1
    m, s = f * np.mean(vals), f * np.std(vals, ddof=1) if len(vals) > 1 else 0.0
    return f"{m:.{d}f} ± {s:.{d}f}"


def summarize():
    L = ["# DAMGF-Net - results for the manuscript", ""]
    au = load_json("audit.json")
    if au:
        L += ["## Image audit", "",
              f"- images before de-duplication: {au['n_images_before']}, unreadable: {au['unreadable']}",
              f"- removed as near-duplicates (pHash Hamming <= {HASH_THRESH}): {au['n_removed_as_near_duplicates']}"
              f" (of which cross-source: {au['cross_source_near_dups_removed']})",
              f"- images after de-duplication: {au['n_images_after_dedup']}; per class: {au['per_class_after']}",
              f"- ORIGINAL random split: {au['orig_split_test_with_near_dup_in_train']} of {au['orig_split_test_size']}"
              f" test images ({100 * au['orig_split_near_dup_fraction']:.2f}%) had a near-duplicate in train"
              f" ({au['orig_split_test_with_exact_dup_in_train']} exact)",
              f"- per class: { {k: (None if v is None else round(100 * v, 2)) for k, v in au['orig_split_near_dup_fraction_per_class'].items()} }",
              "", "Per source x class after de-duplication:", "",
              pd.DataFrame(au["per_source_class_after"]).fillna(0).astype(int).to_markdown(), ""]
    pools = load_json("text_pool_sizes.json")
    if pools:
        L += [f"Text keyword-pool sizes: {pools}", ""]
    g = load_json("grid.json")
    if g:
        agg = defaultdict(list)
        for k, r in g.items():
            v, p, s = k.split("|"); agg[(v, p)].append(r)
        L += ["## Frozen-backbone ablation (mean ± sd over seeds, %)", "",
              "| Variant | Text pairing | n seeds | Accuracy | Macro-F1 | MCC | Smoke recall |",
              "|---|---|---|---|---|---|---|"]
        for v in VARIANTS:
            for p in PAIRINGS:
                rs = agg.get((v, p))
                if not rs:
                    continue
                L.append(f"| {v} | {p} | {len(rs)} | {ms([r['test']['acc'] for r in rs])} | "
                         f"{ms([r['test']['macro_f1'] for r in rs])} | {ms([r['test']['mcc'] for r in rs])} | "
                         f"{ms([r['test']['recall_smoke'] for r in rs])} |")
        L += ["", "## Test-time controls on the full model (accuracy %, mean ± sd)", "",
              "| Trained variant | Pairing | none | IoT+seismic shuffled | drop img | drop txt | drop IoT | drop seismic | drop IoT+seismic | image node only |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for v in ["full", "full_moddrop"]:
            for p in PAIRINGS:
                rs = [r for r in agg.get((v, p), []) if "controls" in r]
                if not rs:
                    continue
                c = lambda n: ms([r["controls"][n]["acc"] for r in rs])
                L.append(f"| {v} | {p} | {ms([r['test']['acc'] for r in rs])} | {c('iot_seis_shuffled')} | "
                         f"{c('drop_img')} | {c('drop_txt')} | {c('drop_iot')} | {c('drop_seis')} | "
                         f"{c('drop_iot_seis')} | {c('image_node_only')} |")
        L += ["", "## Adjacency diagnostics, dynamic graph", "",
              "| Variant | Pairing | row entropy (uniform = 1.386) | rows with max > 0.9 | rows with max > 0.99 | alpha mean (img, txt, iot, seis) |",
              "|---|---|---|---|---|---|"]
        for v in ["full", "full_moddrop", "static_graph"]:
            for p in PAIRINGS:
                rs = [r for r in agg.get((v, p), []) if r.get("adjacency")]
                if not rs:
                    continue
                a = [r["adjacency"] for r in rs]
                al = np.mean([[x["alpha_mean"][m] for m in MOD] for x in a], 0).round(3).tolist()
                L.append(f"| {v} | {p} | {ms([x['row_entropy_mean'] for x in a], pct=False, d=3)} | "
                         f"{ms([x['frac_rows_max_gt_0.9'] for x in a])} % | {ms([x['frac_rows_max_gt_0.99'] for x in a])} % | {al} |")
        fr = [r for k, r in g.items() if k.startswith("full|keyword|")]
        if fr:
            L += ["", "Mean adjacency, full|keyword, first seed (rows = source img, txt, iot, seis):", "",
                  "```", json.dumps(fr[0]["adjacency"]["mean_A"]), "std:", json.dumps(fr[0]["adjacency"]["std_A"]), "```"]
        L += ["", "Kappa vs MCC (5 decimals), full|keyword:"]
        for k, r in g.items():
            if k.startswith("full|keyword|"):
                L.append(f"- seed {k.split('|')[2]}: MCC={r['test']['mcc']:.5f}, kappa={r['test']['kappa']:.5f}")
    lo = load_json("loso.json")
    if lo:
        L += ["", "## Leave-one-source-out", "",
              "| Held-out source | classes | n test | image-only acc | full (keyword) acc | full (masked) acc | full (masked) macro-F1 |",
              "|---|---|---|---|---|---|---|"]
        for src in sorted({k.split("|")[0] for k in lo}):
            if isinstance(lo.get(src), dict) and "skipped" in lo[src]:
                L.append(f"| {src} | {lo[src]['skipped']} | | | | | |"); continue
            def gv(v, p, f="acc"):
                r = lo.get(f"{src}|{v}|{p}")
                return f"{100 * r['test'][f]:.2f}" if r else "-"
            any_r = next((lo[k] for k in lo if k.startswith(src + "|") and "test" in lo[k]), None)
            if any_r:
                L.append(f"| {src} | {','.join(any_r['classes'])} | {any_r['n_test']} | {gv('image_only', 'keyword')} | "
                         f"{gv('full', 'keyword')} | {gv('full', 'keyword_masked')} | {gv('full', 'keyword_masked', 'macro_f1')} |")
    ft = load_json("finetune.json")
    if ft:
        L += ["", "## End-to-end fine-tuning (keyword-masked text)", ""]
        for k, r in ft.items():
            L.append(f"- {k}: acc={100 * r['test']['acc']:.2f}, macro-F1={100 * r['test']['macro_f1']:.2f}, "
                     f"MCC={r['test']['mcc']:.5f}, kappa={r['test']['kappa']:.5f}"
                     + (f"; image node only acc={100 * r['image_node_only']['acc']:.2f}" if "image_node_only" in r else ""))
    ef = load_json("efficiency.json")
    if ef:
        L += ["", f"## Efficiency - device: {ef['device']}", "",
              "| Model | Params (M) | GFLOPs/sample | latency bs=1 (ms) | per-sample bs=16 (ms) | throughput bs=16 (/s) | peak mem bs=16 (MB) |",
              "|---|---|---|---|---|---|---|"]
        for v in ["image_only", "full"]:
            r = ef[v]
            gf = r.get("GFLOPs_per_sample"); gf = f"{gf:.2f}" if isinstance(gf, float) else str(gf)
            L.append(f"| {v} | {r['params'] / 1e6:.2f} | {gf} | {r['bs1_batch_ms_mean']:.2f} ± {r['bs1_batch_ms_std']:.2f} | "
                     f"{r['bs16_per_sample_ms']:.2f} | {r['bs16_throughput_per_s']:.1f} | {r.get('bs16_peak_mem_MB', 0):.0f} |")
    open(os.path.join(OUT_DIR, "RESULTS_FOR_PAPER.md"), "w").write("\n".join(L))
    print("\n".join(L))


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("device:", DEVICE, "| smoke test:", SMOKE_TEST)
    img, txt, iot, seis = load_raw()
    print(f"images {len(img)}, posts {len(txt)}, IoT rows {len(iot)}, seismic events {len(seis)}")
    ded_p = os.path.join(OUT_DIR, "image_manifest_dedup.csv")
    if RUN_STEP["audit"] and not os.path.exists(ded_p):
        ded = step_audit(img)
    elif os.path.exists(ded_p):
        ded = pd.read_csv(ded_p)
    else:
        ded = img.reset_index(drop=True)

    iot_np = iot[IOT_COLS].values.astype(np.float32)
    seis_np = seis[SEIS_COLS].values.astype(np.float32)
    if NORMALIZE_AUX:   # original notebook: raw values (False)
        iot_np = (iot_np - iot_np.mean(0)) / (iot_np.std(0) + 1e-8)
        seis_np = (seis_np - seis_np.min(0)) / (seis_np.max(0) - seis_np.min(0) + 1e-8)

    if RUN_STEP["export"]:
        step_export(ded, txt, iot_np, seis_np)
    if RUN_STEP["cache"] or RUN_STEP["grid"] or RUN_STEP["loso"]:
        F_img, F_to, F_tm = step_cache(ded, txt)
        assert len(F_img) == len(ded), "cached image features do not match manifest - delete cache/"
    if RUN_STEP["grid"]:
        step_grid(ded, txt, iot_np, seis_np, F_img, F_to, F_tm)
    if RUN_STEP["loso"]:
        step_loso(ded, txt, iot_np, seis_np, F_img, F_to, F_tm)
    if RUN_STEP["finetune"]:
        step_finetune(ded, txt, iot_np, seis_np)
    if RUN_STEP["efficiency"]:
        step_efficiency()
    summarize()


if __name__ == "__main__":
    main()
