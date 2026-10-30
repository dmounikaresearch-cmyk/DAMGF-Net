# DAMGF-Net: Dynamic Adaptive Multimodal Graph Fusion for Multi-Hazard Recognition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmounikaresearch-cmyk/DAMGF-Net/blob/main/notebooks/run_in_colab.ipynb)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23725882.svg)](https://doi.org/10.5281/zenodo.23725882)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code and data for the paper:

> D. Mounika, J. Kolluri, S. Srinivas. **DAMGF-Net: Dynamic Adaptive Multimodal Graph Fusion for Multi-Hazard Recognition from Disaster Imagery, Social Media, IoT and Seismic Data — A Controlled Evaluation under Weak Cross-Modal Pairing.** Manuscript submitted for publication, 2026.

Everything needed to repeat the experiments is in this one repository:

- **Code:** data download, perceptual-hash de-duplication, cross-modal pairing, the DAMGF-Net model, baselines, ablations, test-time controls, leave-one-source-out testing and cost profiling.
- **Data:** for every seed, the data splits, the harmonised class mapping, the de-duplication lists and the cross-modal pairing indices (`data/`).
- **Results:** the result files of the run reported in the paper (`results/paper_run/`).

Archived version: Zenodo, DOI [10.5281/zenodo.23725882](https://doi.org/10.5281/zenodo.23725882).

---

## Quick start (3 steps, about 1–2 hours on a free Colab T4 GPU)

1. Click **Open in Colab** above. In Colab choose *Runtime → Change runtime type → T4 GPU*.
2. Have your Kaggle API token ready (`kaggle.json`: Kaggle → *Settings* → *API* → *Create New Token*). The notebook asks for it once.
3. Choose *Runtime → Run all*. The notebook downloads the data, runs every experiment and shows all tables of the paper. The results are saved in your Google Drive (`MyDrive/DAMGF-Net_run/`), so a disconnect loses nothing: run all cells again, and finished steps are skipped.

To check the code without any data or GPU (about 2 minutes, synthetic data):

```bash
pip install -r requirements.txt
DAMGF_SMOKE=1 python src/damgf_experiments.py
```

To check the reported numbers without retraining, open `results/paper_run/RESULTS_FOR_PAPER.md` (Section 6).

---

## 1. Repository layout

```
DAMGF-Net/
├── README.md                       this file
├── LICENSE                         MIT licence
├── requirements.txt                Python packages
├── src/
│   └── damgf_experiments.py        full pipeline (model, training, evaluation, all controls)
├── notebooks/
│   └── run_in_colab.ipynb          one-click reproduction in Google Colab
├── scripts/
│   ├── download_data.py            downloads the four public Kaggle datasets
│   └── check_comments_english.py   helper that lists non-ASCII lines in the code
├── data/
│   ├── class_mappings/             source folder -> harmonised class; text-pairing keywords
│   ├── dedup/                      pHash of every image; list of removed near-duplicates
│   ├── splits/seed{42,43,44}/      train / val / test image identifiers
│   ├── splits/loso/<source>/       leave-one-source-out splits
│   └── pairing/seed{42,43,44}/     post, sensor and seismic record assigned to every image
└── results/
    ├── paper_run/                  result files behind Tables 10-14 of the paper
    └── run/                        output folder of a new run
```

## 2. Where each part of the paper is in the code

All components are in `src/damgf_experiments.py`.

| Paper | Code |
|---|---|
| Image encoder (Swin-Tiny + projection), text encoder (BERT-base + projection) | `make_swin`, `make_bert`, `proj_head` |
| Sensor encoder (Conv1d × 2 + BiLSTM), seismic encoder (MLP), Sect. 4.3.1 | `IoTEncoder`, `SeisEncoder` |
| Adaptive cross-modal attention fusion (ACMAF), Eqs. 5–7 | `ACMAF` |
| Dynamic graph constructor and graph transformer (DGT; named STGT in the code), Eqs. 8–17 | `GraphModule` |
| Prediction head with auxiliary severity term, Eqs. 18–19 | `FusionModel.shared/cls/sev`, `loss_fn` |
| Baselines and ablations (image-only, text-only, IoT-only, seismic-only, concatenation, ACMAF-only, static graph, full, λ = 0, modality dropout) | `FusionModel(variant=...)`, `VARIANTS` |
| Cross-modal pairing (keyword, keyword-masked, random), Sect. 4.2 | `text_pools`, `make_pairs`, `mask_text` |
| pHash de-duplication (Hamming distance ≤ 4), Eq. 20 | `phash_all`, `step_audit` |
| Test-time shuffling and modality removal, Eq. 21 | `step_grid` |
| Adjacency entropy and saturation, Eqs. 22–23 | `adjacency_diag` |
| Metrics (accuracy, macro-F1, MCC, kappa, AUC), Eqs. 24–26 | `metrics` |
| Leave-one-source-out, Eq. 27 | `step_loso` |
| Parameters, GFLOPs, latency, memory | `step_efficiency` |

## 3. Environment

The experiments were run on Google Colab with one NVIDIA Tesla T4 GPU (16 GB) and Python 3.12. Any CUDA GPU with at least 8 GB of memory works; a CPU-only run is possible but slow.

```bash
git clone https://github.com/dmounikaresearch-cmyk/DAMGF-Net.git
cd DAMGF-Net
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
```

The exact package versions of a run are written to `results/run/environment.txt` (Colab route).

## 4. Data

### 4.1 Source datasets (all public)

| Modality | Kaggle dataset | Records used | Licence | In this repository |
|---|---|---|---|---|
| Images | [`georgemystriotis/disasters-dataset`](https://www.kaggle.com/datasets/georgemystriotis/disasters-dataset): compilation of seven collections (Table 3 of the paper) | 68,308 → 58,939 after de-duplication | per collection: CC0, CC BY 4.0, CC BY-NC-SA 4.0, GPL-2.0, not stated | identifiers only |
| Social-media posts | [`nihajasminesk/disaster-or-not-dataset`](https://www.kaggle.com/datasets/nihajasminesk/disaster-or-not-dataset) (`Tweets.csv`) | 45,384 | CC BY 4.0 | row indices |
| Accelerometer and gyroscope | [`aaryapandya/earthquake-detection`](https://www.kaggle.com/datasets/aaryapandya/earthquake-detection) (`Data.csv`) | 59,276 | LGPL-3.0 | row indices |
| Seismic catalogue | [`ahmeduzaki/earthquake-alert-prediction-dataset`](https://www.kaggle.com/datasets/ahmeduzaki/earthquake-alert-prediction-dataset) | 1,300 | CC BY 4.0 | row indices |

The images are not redistributed here, because two collections carry no licence statement and one is CC BY-NC-SA 4.0. The download script fetches the original files; every released file refers to them by identifier, so the exact corpus is rebuilt.

### 4.2 Download (command-line route)

1. Create a Kaggle API token (Kaggle → *Settings* → *API* → *Create New Token*) and save it as `~/.kaggle/kaggle.json` (on Linux/macOS also run `chmod 600 ~/.kaggle/kaggle.json`).
2. Run:

```bash
python scripts/download_data.py --out data/MultiHazard
```

This downloads about 1.8 GB and checks the record counts (68,308 images, 45,384 posts, 59,276 sensor readings, 1,300 seismic events). Resulting layout:

```
data/MultiHazard/RemoteSensing/<source>/.../<class folder>/<image>
data/MultiHazard/SocialMedia/Tweets.csv
data/MultiHazard/IoT/Data.csv
data/MultiHazard/Seismic/earthquake_alert_balanced_dataset.csv
```

### 4.3 Released files in `data/`

| File | Content |
|---|---|
| `class_mappings/class_mapping.csv` | source folder name → harmonised class (earthquake, flood, fire, smoke, normal) |
| `class_mappings/text_pairing_keywords.csv` | keywords that define each class's post pool, pool sizes, severity targets |
| `dedup/phash.csv` | `image_id, source, label, phash, kept` for all 68,308 images (in processing order) |
| `dedup/removed.csv` | each removed image, the earlier image it duplicates, its source and the Hamming distance (9,369 rows) |
| `splits/seed{42,43,44}/{train,val,test}.csv` | `image_id, source, label` (70/15/15, stratified, de-duplicated corpus) |
| `splits/loso/<source>/{train,val,test}.csv` | leave-one-source-out splits (seed 42) |
| `pairing/seed{s}/{keyword,random_text}_{train,val,test}.csv` | `image_id, label, post_row, sensor_row, seismic_row, severity_target` |

- `image_id` is the path of the image relative to `data/MultiHazard/RemoteSensing/`.
- `post_row`, `sensor_row` and `seismic_row` are 0-based row indices in `Tweets.csv`, `Data.csv` and `earthquake_alert_balanced_dataset.csv`.
- The keyword-masked condition uses the same indices as `keyword_*`; every pairing keyword is replaced by `[MASK]` before encoding (`mask_text`).
- The pipeline regenerates all of these files. The `EXPORT` step writes them with the same functions and seeds used for training, so a new run can be compared with the released files line by line.

## 5. Reproducing the experiments

### Option A: Google Colab (recommended, no installation)

Click **Open in Colab** at the top of this page and follow the Quick start. The notebook has six cells:

| Cell | What it does |
|---|---|
| 1 | mounts Google Drive, where all results are kept, and checks that a GPU is available |
| 2 | clones this repository and installs the packages |
| 3 | asks for `kaggle.json` and downloads the four datasets |
| 4 | runs all experiments (about 1–2 h) and records the package versions |
| 5 | shows all result tables |
| 6 | downloads a zip with the regenerated `data/` and `results/run/` folders |

### Option B: command line (one GPU)

```bash
python scripts/download_data.py --out data/MultiHazard
python src/damgf_experiments.py
```

The script runs the steps below in order and prints its progress. Every step is resumable: finished results are stored as JSON and skipped on the next run.

| Pipeline step | Paper output | Written to |
|---|---|---|
| STEP 1: image audit, pHash de-duplication, near-duplicate leakage | Table 3 (after de-duplication), Sect. 5.7, Fig. 2 | `results/run/audit.json`, `fig_examples_per_class.png`, `data/dedup/` |
| EXPORT: splits and pairing indices | Sect. 4.2, 4.7 | `data/splits/`, `data/pairing/`, `data/class_mappings/` |
| STEP 2: frozen Swin-Tiny / BERT features | — | `results/run/cache/` |
| STEP 3: baselines, ablations, three pairings, three seeds; test-time interventions; adjacency diagnostics | Tables 10, 11, Sect. 5.8 | `results/run/grid.json` |
| STEP 4: leave-one-source-out | Table 12 | `results/run/loso.json` |
| STEP 5 (optional): end-to-end fine-tuning | Sect. 5.1–5.5 | `results/run/` |
| STEP 6: parameters, GFLOPs, latency, memory | Table 14 | `results/run/efficiency.json` |
| STEP 7: summary of all tables | all of the above | `results/run/RESULTS_FOR_PAPER.md` |

Running time on a Colab T4: about 1–2 hours for steps 1–4, 6 and 7, most of it feature caching.

Optional settings (environment variables):

| Variable | Meaning | Default |
|---|---|---|
| `DAMGF_DATA` | folder of the downloaded data | `data/MultiHazard` |
| `DAMGF_OUT` | output folder for results | `results/run` |
| `DAMGF_EXPORT` | output folder for split and pairing files | `data` |
| `DAMGF_FINETUNE` | `1` = also run the end-to-end fine-tuning of step 5 (several hours) | `0` |
| `DAMGF_SMOKE` | `1` = 2-minute test on synthetic data | `0` |

Example, end-to-end fine-tuning of the image-only and full models (keyword-masked text):

```bash
DAMGF_FINETUNE=1 python src/damgf_experiments.py
```

## 6. Expected results

Values reported in the paper, from the run stored in `results/paper_run/`. Controlled protocol, de-duplicated corpus (58,939 images), mean ± standard deviation over seeds 42–44 (Table 10):

| Variant | Text pairing | Accuracy (%) |
|---|---|---|
| Image only | — | 96.08 ± 0.03 |
| Text only | keyword | 67.85 ± 4.84 |
| Text only | keyword-masked | 70.44 ± 2.87 |
| Text only / IoT only / seismic only | random / — / — | 30.23 (majority class) |
| Full DAMGF-Net | keyword | 95.69 ± 0.24 |
| Full DAMGF-Net | keyword-masked | 96.38 ± 0.44 |
| Full DAMGF-Net | random | 95.67 ± 0.15 |

Other checks:

- De-duplication removes 9,369 near-duplicates (68,308 → 58,939 images).
- In a random split of the original corpus, 1,860 of 10,247 test images (18.15%) have a near-duplicate in training.
- Leave-one-source-out mean accuracy is 80.45% (image only) to 83.71% (full, keyword-masked), Table 12.
- The full model needs 15.42 GFLOPs per sample (Table 14).

Files in `results/paper_run/`:

| File | Content |
|---|---|
| `RESULTS_FOR_PAPER.md` | all tables in readable form |
| `grid.json` | every variant × pairing × seed: test metrics, confusion matrix, training history, test-time controls, adjacency diagnostics |
| `loso.json` | leave-one-source-out results |
| `efficiency.json` | parameters, GFLOPs, latency and memory |
| `text_pool_sizes.json` | size of each keyword post pool |
| `run_log.txt` | console log of the run |
| `fig_examples_per_class.png` | example images per class and source (Fig. 2) |

A new run with the same seeds gives the same splits and pairings as the released `data/` files. Accuracies can differ in the second decimal place across GPU types and library versions, because some CUDA kernels are non-deterministic.

## 7. Training settings

- **Controlled protocol** (all variants, same procedure): frozen Swin-Tiny and BERT-base features; AdamW (learning rate 1e-3, weight decay 1e-4), cosine schedule, batch size 256, at most 30 epochs, early stopping with patience 5 on validation loss, label smoothing 0.1, λ = 0.30, gradient clipping 1.0; seeds 42, 43 and 44.
- **End-to-end protocol** (optional step 5): AdamW (learning rate 1e-4, weight decay 1e-4), batch size 16, at most 20 epochs, mixed precision, other settings as above.
- **Model**: 512-d embeddings; graph transformer with 2 layers, 8 heads, FFN 2048; dropout 0.30.
- **Inputs**: images resized to 256 and centre-cropped to 224, ImageNet normalisation; BERT tokenizer, maximum 128 tokens, raw posts; sensor and seismic values unscaled.

## 8. Troubleshooting

| Problem | Solution |
|---|---|
| `401 Unauthorized` or `Could not find kaggle.json` | Create a new Kaggle API token and place it at `~/.kaggle/kaggle.json` (Colab: choose it when cell 3 asks). |
| `403 Forbidden` for a Kaggle dataset | Open the dataset page on Kaggle once while signed in and accept its terms, then run the download again. |
| `GPU: NONE` in Colab | *Runtime → Change runtime type → T4 GPU*, then run all cells again. |
| Colab disconnected | Run all cells again; finished steps are read from Google Drive and skipped. |
| `CUDA out of memory` | Use a GPU with at least 8 GB, or close other sessions using the GPU. |
| Pretrained weights fail to download | The first run needs internet access to Hugging Face (BERT) and timm (Swin-Tiny). |

## 9. Citation

```bibtex
@article{mounika2026damgf,
  title   = {DAMGF-Net: Dynamic Adaptive Multimodal Graph Fusion for Multi-Hazard Recognition from Disaster Imagery, Social Media, IoT and Seismic Data---A Controlled Evaluation under Weak Cross-Modal Pairing},
  author  = {Mounika, Dannuri and Kolluri, Johnson and Srinivas, Sunil},
  journal = {Manuscript submitted for publication},
  year    = {2026},
  doi     = {10.5281/zenodo.23725882}
}
```

## 10. Licence and contact

The code is released under the MIT licence (`LICENSE`). The source datasets keep their original licences (Section 4.1).

Corresponding author: Johnson Kolluri, johnson.kolluri@gmail.com. First author: Dannuri Mounika, dmounikaresearch@gmail.com.
