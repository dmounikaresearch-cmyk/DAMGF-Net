"""Download the four public Kaggle datasets used by DAMGF-Net.

Usage:
    python scripts/download_data.py --out data/MultiHazard

Requirements:
    pip install kaggle
    A Kaggle API token in ~/.kaggle/kaggle.json (Kaggle -> Settings -> API ->
    Create New Token), or the environment variables KAGGLE_USERNAME and KAGGLE_KEY.

Resulting layout (the layout expected by src/damgf_experiments.py):
    <out>/RemoteSensing/<source>/.../<class folder>/<image>   seven image collections
    <out>/SocialMedia/Tweets.csv                              45,384 posts
    <out>/IoT/Data.csv                                        59,276 sensor readings
    <out>/Seismic/earthquake_alert_balanced_dataset.csv       1,300 seismic events

The image collections are downloaded as the single Kaggle compilation
georgemystriotis/disasters-dataset, which contains the seven collections listed
in Table 3 of the paper (about 1.8 GB).
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

IMAGES = "georgemystriotis/disasters-dataset"
TEXT = "nihajasminesk/disaster-or-not-dataset"
IOT = "aaryapandya/earthquake-detection"
SEISMIC = "ahmeduzaki/earthquake-alert-prediction-dataset"

EXPECTED = {"images": 68308, "posts": 45384, "iot": 59276, "seismic": 1300}


def kaggle_download(ref, dest):
    """Download and unzip one Kaggle dataset into dest."""
    os.makedirs(dest, exist_ok=True)
    print(f"[download] {ref} -> {dest}")
    subprocess.run(["kaggle", "datasets", "download", "-d", ref, "-p", dest, "--unzip", "-q"],
                   check=True)


def copy_first_csv(src_dir, target, prefer=None):
    """Copy the CSV called `prefer` (or the first CSV found) from src_dir to target."""
    csvs = glob.glob(os.path.join(src_dir, "**", "*.csv"), recursive=True)
    if not csvs:
        sys.exit(f"no CSV file found in {src_dir}")
    chosen = [c for c in csvs if prefer and os.path.basename(c).lower() == prefer.lower()] or csvs
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copy(chosen[0], target)


def count_images(root):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sum(1 for _, _, fs in os.walk(root) for f in fs if os.path.splitext(f)[1].lower() in exts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/MultiHazard", help="output root")
    args = ap.parse_args()
    root = args.out
    if shutil.which("kaggle") is None:
        sys.exit("The Kaggle command-line tool is missing. Install it with: pip install kaggle")

    img_dir = os.path.join(root, "RemoteSensing")
    if not os.path.isdir(os.path.join(img_dir, "disasters-dataset-final")):
        kaggle_download(IMAGES, img_dir)
    else:
        print("[skip] images already present")

    targets = [(TEXT, os.path.join(root, "SocialMedia", "Tweets.csv"), "Tweets.csv"),
               (IOT, os.path.join(root, "IoT", "Data.csv"), "Data.csv"),
               (SEISMIC, os.path.join(root, "Seismic", "earthquake_alert_balanced_dataset.csv"),
                "earthquake_alert_balanced_dataset.csv")]
    for ref, target, prefer in targets:
        if os.path.exists(target):
            print(f"[skip] {target} already present")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            kaggle_download(ref, tmp)
            copy_first_csv(tmp, target, prefer)

    # Sanity check against the counts reported in the paper
    import pandas as pd
    found = {"images": count_images(img_dir),
             "posts": len(pd.read_csv(targets[0][1])),
             "iot": len(pd.read_csv(targets[1][1])),
             "seismic": len(pd.read_csv(targets[2][1]))}
    print("\nRecords found (paper value in brackets):")
    for k, v in found.items():
        flag = "OK" if k != "images" or v >= EXPECTED[k] else "CHECK"
        print(f"  {k:8s} {v:>7d}  ({EXPECTED[k]})  {flag}")
    print("Note: the image compilation also contains folders that are not used; "
          "src/damgf_experiments.py keeps only the class folders listed in "
          "data/class_mappings/class_mapping.csv (68,308 images).")


if __name__ == "__main__":
    main()
