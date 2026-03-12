"""
scripts/download_data.py
Dataset download utilities for ASVspoof 2019 LA (Kaggle) and
In-the-Wild (HuggingFace).

Usage:
    python main.py --mode download --dataset all
    python main.py --mode download --dataset asvspoof
    python main.py --mode download --dataset in_the_wild
    python scripts/download_data.py          (standalone)
"""

import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Target directories
DATA_ROOT = Path("data")
ASVSPOOF_DIR = DATA_ROOT / "ASVspoof2019_LA"
ITW_CACHE_DIR = DATA_ROOT / "in_the_wild"


# ============================================================
# ASVspoof 2019 LA — Kaggle
# ============================================================

def download_asvspoof(target_dir: Path | None = None) -> None:
    """Download and extract ASVspoof 2019 LA dataset from Kaggle.

    Requires:
        - kaggle.json in ~/.kaggle/ or KAGGLE_USERNAME + KAGGLE_KEY env vars
        - `kaggle` Python package installed

    The Kaggle dataset slug is: awsaf49/asvpoof-2019-dataset
    (common community upload of ASVspoof 2019 LA)

    Args:
        target_dir: Target directory. Defaults to data/ASVspoof2019_LA.
    """
    if target_dir is None:
        target_dir = ASVSPOOF_DIR

    target_dir = Path(target_dir)

    # Check if already downloaded
    train_flac = target_dir / "ASVspoof2019_LA_train" / "flac"
    if train_flac.exists() and any(train_flac.iterdir()):
        logger.info("ASVspoof 2019 LA already exists at %s", target_dir)
        print(f"✅ ASVspoof 2019 LA already downloaded at: {target_dir}")
        return

    print("=" * 60)
    print("  Downloading ASVspoof 2019 LA from Kaggle")
    print("=" * 60)

    # Verify Kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env_creds = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")

    if not kaggle_json.exists() and not has_env_creds:
        print("\n⚠️  Kaggle credentials not found!")
        print("  Option 1: Place kaggle.json in ~/.kaggle/")
        print("  Option 2: Set KAGGLE_USERNAME and KAGGLE_KEY env vars")
        print("  Get your API token from: https://www.kaggle.com/settings")
        sys.exit(1)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        print("✅ Kaggle API authenticated")
    except Exception as e:
        print(f"❌ Kaggle authentication failed: {e}")
        print("  Ensure kaggle package is installed: uv pip install kaggle")
        sys.exit(1)

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset
    dataset_slug = "awsaf49/asvpoof-2019-dataset"
    print(f"\nDownloading: {dataset_slug}")
    print(f"Target:      {target_dir}")
    print("This may take a while (~6 GB)...\n")

    try:
        api.dataset_download_files(
            dataset_slug,
            path=str(target_dir),
            unzip=True,
        )
        print(f"\n✅ ASVspoof 2019 LA downloaded to: {target_dir}")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("  Try manually: kaggle datasets download -d awsaf49/asvpoof-2019-dataset")
        sys.exit(1)

    # Verify expected structure
    expected_dirs = [
        "ASVspoof2019_LA_train/flac",
        "ASVspoof2019_LA_dev/flac",
        "ASVspoof2019_LA_eval/flac",
        "ASVspoof2019_LA_cm_protocols",
        "ASVspoof2019_LA_asv_scores",
    ]

    all_found = True
    for d in expected_dirs:
        full_path = target_dir / d
        if full_path.exists():
            if full_path.is_dir():
                n_files = len(list(full_path.iterdir()))
                print(f"  ✅ {d} ({n_files} items)")
            else:
                print(f"  ✅ {d}")
        else:
            print(f"  ❌ {d} — NOT FOUND")
            all_found = False

    if all_found:
        print("\n✅ All expected directories verified!")
    else:
        print("\n⚠️  Some directories are missing. The dataset may have a nested structure.")
        print("  Check the download directory and reorganize if needed.")


# ============================================================
# In-the-Wild — HuggingFace
# ============================================================

def download_in_the_wild(cache_dir: Path | None = None) -> None:
    """Download In-the-Wild dataset from HuggingFace.

    Uses: mueller91/In-The-Wild
    The dataset will be cached locally by the `datasets` library.

    Args:
        cache_dir: Local cache directory. Defaults to data/in_the_wild.
    """
    if cache_dir is None:
        cache_dir = ITW_CACHE_DIR

    cache_dir = Path(cache_dir)

    print("=" * 60)
    print("  Downloading In-the-Wild from HuggingFace")
    print("=" * 60)

    try:
        from datasets import load_dataset

        print("\nLoading mueller91/In-The-Wild...")
        print(f"Cache dir: {cache_dir}")
        print("This may take a while on first download...\n")

        cache_dir.mkdir(parents=True, exist_ok=True)

        ds = load_dataset(
            "mueller91/In-The-Wild",
            cache_dir=str(cache_dir),
            trust_remote_code=True,
        )

        # Print info
        for split_name in ds:
            split = ds[split_name]
            print(f"  Split '{split_name}': {len(split)} samples")
            if len(split) > 0:
                print(f"    Features: {list(split.features.keys())}")
                # Count labels
                if "label" in split.features:
                    labels = split["label"]
                    n_real = sum(1 for l in labels if l == 1)
                    n_fake = sum(1 for l in labels if l == 0)
                    print(f"    Real: {n_real}, Fake: {n_fake}")

        print(f"\n✅ In-the-Wild dataset cached at: {cache_dir}")

    except ImportError:
        print("❌ 'datasets' package not installed.")
        print("  Install: uv pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)


# ============================================================
# Standalone Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["asvspoof", "in_the_wild", "all"],
        help="Which dataset to download.",
    )
    args = parser.parse_args()

    if args.dataset in ("asvspoof", "all"):
        download_asvspoof()

    if args.dataset in ("in_the_wild", "all"):
        download_in_the_wild()

    print("\n✅ All downloads complete!")
