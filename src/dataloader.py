"""
src/dataloader.py
Dataset classes and K-Fold manager for audio anti-spoofing.

Includes:
    - ASVspoof2019Dataset: ASVspoof 2019 LA dataset
    - InTheWildDataset: In-the-Wild evaluation dataset
    - KFoldManager: Merged train+dev K-Fold cross validation
    - get_dataloader: DataLoader factory
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from sklearn.model_selection import StratifiedKFold
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.utils import RawBoost

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

SAMPLE_RATE = 16000
TARGET_SAMPLES = 64000  # 4 seconds at 16 kHz
LABEL_MAP = {"bonafide": 1, "spoof": 0}


# ============================================================
# 4a. ASVspoof 2019 Dataset
# ============================================================

class ASVspoof2019Dataset(Dataset):
    """ASVspoof 2019 LA dataset for training, validation, or evaluation.

    Protocol format (space-separated):
        speaker_id  filename  -  attack_type  label
        label values: "bonafide" or "spoof"
        Label mapping: bonafide=1, spoof=0

    Args:
        samples: List of dicts with keys 'filename', 'label', 'speaker_id'.
        flac_dir: Path to directory containing .flac files.
        target_len: Target waveform length in samples.
        augment: Whether to apply RawBoost augmentation.
        rawboost_params: RawBoost hyperparameters dict.
        rawboost_algo: RawBoost algorithm number (1–4).
        is_eval: If True, use deterministic padding (no random crop).
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        flac_dir: str | Path,
        target_len: int = TARGET_SAMPLES,
        augment: bool = False,
        rawboost_params: dict[str, Any] | None = None,
        rawboost_algo: int = 4,
        is_eval: bool = False,
    ) -> None:
        self.samples = samples
        self.flac_dir = Path(flac_dir)
        self.target_len = target_len
        self.augment = augment
        self.is_eval = is_eval
        self.rawboost_algo = rawboost_algo

        if augment:
            self.rawboost = RawBoost(params=rawboost_params, sr=SAMPLE_RATE)
        else:
            self.rawboost = None

        logger.info(
            "ASVspoof2019Dataset: %d samples from %s (augment=%s, eval=%s)",
            len(self.samples), self.flac_dir, self.augment, self.is_eval,
        )

        self._audio_cache: list[np.ndarray] = []
        logger.info("Pre-loading %d audio files to RAM...", len(self.samples))
        for s in self.samples:
            fpath = self.flac_dir / f"{s['filename']}.flac"
            wav, _ = sf.read(str(fpath))
            self._audio_cache.append(wav.astype(np.float64))
        logger.info("Audio pre-load complete. RAM usage approx %.1f MB",
                    sum(a.nbytes for a in self._audio_cache) / 1e6)

    @staticmethod
    def parse_protocol(protocol_path: str | Path) -> list[dict[str, Any]]:
        """Parse an ASVspoof 2019 protocol file.

        Args:
            protocol_path: Path to protocol .txt file.

        Returns:
            List of dicts with keys: 'speaker_id', 'filename',
            'attack_type', 'label' (int: bonafide=1, spoof=0).
        """
        samples = []
        protocol_path = Path(protocol_path)
        if not protocol_path.exists():
            raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                speaker_id, filename, _, attack_type, label_str = parts[:5]
                label = LABEL_MAP.get(label_str)
                if label is None:
                    logger.warning("Unknown label '%s' for %s, skipping.", label_str, filename)
                    continue
                samples.append({
                    "speaker_id": speaker_id,
                    "filename": filename,
                    "attack_type": attack_type,
                    "label": label,
                })

        logger.info("Parsed %d samples from %s", len(samples), protocol_path.name)
        return samples

    def crop_or_pad(self, waveform: np.ndarray) -> np.ndarray:
        """Crop or pad waveform to target length.

        For training: random crop if longer, tile-pad if shorter.
        For eval: deterministic center crop / zero-pad.

        Args:
            waveform: 1D numpy array.

        Returns:
            Waveform of shape (target_len,).
        """
        x_len = waveform.shape[0]

        if x_len >= self.target_len:
            if self.is_eval:
                # Deterministic: take from beginning
                return waveform[: self.target_len]
            else:
                # Random crop for training
                start = np.random.randint(0, x_len - self.target_len + 1)
                return waveform[start : start + self.target_len]
        else:
            # Pad by tiling
            num_repeats = (self.target_len // x_len) + 1
            padded = np.tile(waveform, num_repeats)[: self.target_len]
            return padded

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str]:
        """Load and return a single sample.

        Returns:
            Tuple of (waveform_tensor[target_len], label_int, filename_str).
        """
        sample = self.samples[index]
        filename = sample["filename"]
        label = sample["label"]

        # Load audio
        waveform = self._audio_cache[index].copy()

        # Crop or pad to target length
        waveform = self.crop_or_pad(waveform)

        # Apply RawBoost augmentation (training only)
        if self.augment and self.rawboost is not None:
            waveform = self.rawboost(waveform, algo=self.rawboost_algo)

        # Convert to tensor
        x = Tensor(waveform.astype(np.float32))

        return x, label, filename


# ============================================================
# 4b. In-the-Wild Dataset
# ============================================================

class InTheWildDataset(Dataset):
    """In-the-Wild evaluation dataset loaded from local files.

    Reads audio files and labels directly from the locally downloaded dataset.
    Expected structure:
        data_dir/release_in_the_wild/
            meta.csv      ← columns: file, speaker, label
            *.wav         ← audio files

    Used for cross-domain evaluation only (no augmentation, deterministic).

    Label convention (same as ASVspoof, no inversion):
        "bonafide" → 1
        "spoof"    → 0

    Args:
        data_dir: Root directory of the In-the-Wild dataset.
        target_len: Target waveform length in samples.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/in_the_wild",
        target_len: int = TARGET_SAMPLES,
    ) -> None:
        import pandas as pd

        self.data_dir = Path(data_dir)
        self.audio_dir = self.data_dir / "release_in_the_wild"
        self.target_len = target_len

        # Validate directory
        if not self.audio_dir.exists():
            raise FileNotFoundError(
                f"In-the-Wild audio directory not found: {self.audio_dir}\n"
                "Expected structure: data_dir/release_in_the_wild/*.wav + meta.csv"
            )

        # Read meta.csv
        meta_path = self.audio_dir / "meta.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.csv not found at: {meta_path}")

        df = pd.read_csv(meta_path)

        # Validate required columns
        required_cols = {"file", "label"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"meta.csv is missing required columns: {missing}. "
                f"Found: {list(df.columns)}"
            )

        # Map label strings to ints: bonafide=1, spoof=0
        # Normalize label: strip whitespace, lowercase, remove hyphens
        # so "bona-fide" and "bonafide" both map to 1
        _itw_label_map = {**LABEL_MAP, "bona-fide": 1}
        df["label_int"] = (
            df["label"].str.strip().str.lower().map(_itw_label_map)
        )
        unknown_mask = df["label_int"].isna()
        if unknown_mask.any():
            unknown_vals = df.loc[unknown_mask, "label"].unique().tolist()
            logger.warning(
                "[InTheWild] Unknown labels found and dropped: %s", unknown_vals
            )
            df = df[~unknown_mask].reset_index(drop=True)

        df["label_int"] = df["label_int"].astype(int)

        # Build a filename→Path lookup in one O(n) directory scan
        # (avoids ~31K individual Path.exists() syscalls)
        existing_files: dict[str, Path] = {
            p.name: p for p in self.audio_dir.iterdir() if p.is_file()
        }

        def _resolve_path(fname: str) -> Path:
            base = Path(fname).name
            if base in existing_files:
                return existing_files[base]
            # Try adding .wav if the meta.csv entry has no extension
            wav_base = base + ".wav"
            if wav_base in existing_files:
                return existing_files[wav_base]
            # Fallback: will raise a clear FileNotFoundError in __getitem__
            return self.audio_dir / base

        df["audio_path"] = df["file"].astype(str).map(_resolve_path)

        self.df = df
        self.samples = df[["audio_path", "label_int", "file"]].to_dict("records")

        # Logging: total count and label distribution
        n_bonafide = int((df["label_int"] == 1).sum())
        n_spoof = int((df["label_int"] == 0).sum())
        total = len(df)
        print(
            f"[InTheWild] Loaded {total} samples — "
            f"bonafide={n_bonafide}, spoof={n_spoof} | "
            f"audio_dir={self.audio_dir}"
        )
        logger.info(
            "[InTheWild] %d samples (bonafide=%d, spoof=%d) from %s",
            total, n_bonafide, n_spoof, self.audio_dir,
        )

    def crop_or_pad(self, waveform: np.ndarray) -> np.ndarray:
        """Deterministic crop or pad for evaluation (always from start).

        Args:
            waveform: 1-D numpy array.

        Returns:
            Waveform of shape (target_len,).
        """
        x_len = waveform.shape[0]
        if x_len >= self.target_len:
            # Always take from the beginning — deterministic for eval
            return waveform[: self.target_len]
        else:
            # Tile-pad to fill target length
            num_repeats = (self.target_len // x_len) + 1
            return np.tile(waveform, num_repeats)[: self.target_len]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str]:
        """Load and return a single sample.

        Returns:
            Tuple of (waveform_tensor[target_len], label_int, filename_stem).
        """
        record = self.samples[index]
        audio_path: Path = record["audio_path"]
        label: int = record["label_int"]
        # Return the stem (filename without extension) as identifier
        fname_stem: str = Path(str(record["file"])).stem

        # Load audio via soundfile
        waveform, sr = sf.read(str(audio_path), dtype="float32")

        # Handle stereo → mono by averaging channels
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)

        # Ensure 1-D float array
        waveform = waveform.astype(np.float32)

        # Resample only if needed
        if sr != SAMPLE_RATE:
            import librosa
            waveform = librosa.resample(
                waveform.astype(np.float64),
                orig_sr=sr,
                target_sr=SAMPLE_RATE,
            ).astype(np.float32)

        # Deterministic crop / tile-pad to target length
        waveform = self.crop_or_pad(waveform)

        x = torch.from_numpy(waveform)
        return x, label, fname_stem


# ============================================================
# 4c. KFoldManager
# ============================================================

class KFoldManager:
    """Manages merged K-Fold cross validation for ASVspoof 2019 LA.

    Merges the official train and dev splits into a single pool,
    then applies StratifiedKFold to create K train/val splits.

    Args:
        data_root: Root directory of ASVspoof 2019 LA data.
        n_splits: Number of folds (default 5).
        seed: Random seed for shuffling (default 42).
    """

    def __init__(
        self,
        data_root: str | Path,
        n_splits: int = 5,
        seed: int = 42,
    ) -> None:
        self.data_root = Path(data_root)
        self.n_splits = n_splits
        self.seed = seed

        # Parse train + dev protocols
        train_protocol = (
            self.data_root
            / "ASVspoof2019_LA_cm_protocols"
            / "ASVspoof2019.LA.cm.train.trn.txt"
        )
        dev_protocol = (
            self.data_root
            / "ASVspoof2019_LA_cm_protocols"
            / "ASVspoof2019.LA.cm.dev.trl.txt"
        )

        train_samples = ASVspoof2019Dataset.parse_protocol(train_protocol)
        dev_samples = ASVspoof2019Dataset.parse_protocol(dev_protocol)

        # Tag samples with their original split (for flac_dir resolution)
        for s in train_samples:
            s["split"] = "train"
        for s in dev_samples:
            s["split"] = "dev"

        # Merge
        self.all_samples = train_samples + dev_samples
        self.all_labels = np.array([s["label"] for s in self.all_samples])

        # Shuffle and create folds
        self.skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        self.folds = list(self.skf.split(self.all_samples, self.all_labels))

        # Log info
        n_bonafide = int(np.sum(self.all_labels == 1))
        n_spoof = int(np.sum(self.all_labels == 0))
        print(
            f"KFoldManager: {len(self.all_samples)} total samples "
            f"(bonafide={n_bonafide}, spoof={n_spoof}), "
            f"{n_splits} folds, seed={seed}"
        )
        logger.info(
            "KFoldManager: %d samples (bonafide=%d, spoof=%d), %d folds",
            len(self.all_samples), n_bonafide, n_spoof, n_splits,
        )

    def _resolve_flac_dir(self, sample: dict[str, Any]) -> Path:
        """Resolve the flac directory based on original split."""
        if sample["split"] == "train":
            return self.data_root / "ASVspoof2019_LA_train" / "flac"
        else:
            return self.data_root / "ASVspoof2019_LA_dev" / "flac"

    def get_fold(
        self,
        fold_idx: int,
        augment: bool,
        config: dict[str, Any],
    ) -> tuple[Dataset, Dataset]:
        """Get train and validation datasets for a specific fold.

        Args:
            fold_idx: Fold index (0 to n_splits-1).
            augment: Whether to apply augmentation to training set.
            config: Configuration dictionary for RawBoost params.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        if fold_idx < 0 or fold_idx >= self.n_splits:
            raise ValueError(
                f"fold_idx must be 0–{self.n_splits - 1}, got {fold_idx}"
            )

        train_indices, val_indices = self.folds[fold_idx]

        # Build per-sample datasets
        # We need to handle mixed flac_dirs (train vs dev samples)
        train_samples = [self.all_samples[i] for i in train_indices]
        val_samples = [self.all_samples[i] for i in val_indices]

        # RawBoost config
        rb_config = config.get("augmentation", {}).get("rawboost", {})
        rb_algo = rb_config.get("algo", 4)

        # We need a MultiDirDataset that resolves per-sample flac dirs
        train_dataset = _MultiDirASVspoofDataset(
            samples=train_samples,
            data_root=self.data_root,
            target_len=config.get("data", {}).get("target_samples", TARGET_SAMPLES),
            augment=augment,
            rawboost_params=rb_config,
            rawboost_algo=rb_algo,
            is_eval=False,
        )

        val_dataset = _MultiDirASVspoofDataset(
            samples=val_samples,
            data_root=self.data_root,
            target_len=config.get("data", {}).get("target_samples", TARGET_SAMPLES),
            augment=False,  # Never augment validation
            rawboost_params=None,
            rawboost_algo=rb_algo,
            is_eval=True,
        )

        logger.info(
            "Fold %d: train=%d, val=%d (augment=%s)",
            fold_idx, len(train_dataset), len(val_dataset), augment,
        )

        return train_dataset, val_dataset

    def get_label_distribution(self, fold_idx: int) -> dict[str, dict[str, int]]:
        """Get label distribution for a specific fold.

        Args:
            fold_idx: Fold index.

        Returns:
            Dict with 'train' and 'val' sub-dicts containing
            bonafide/spoof counts.
        """
        train_indices, val_indices = self.folds[fold_idx]

        train_labels = self.all_labels[train_indices]
        val_labels = self.all_labels[val_indices]

        return {
            "train": {
                "bonafide": int(np.sum(train_labels == 1)),
                "spoof": int(np.sum(train_labels == 0)),
                "total": len(train_labels),
            },
            "val": {
                "bonafide": int(np.sum(val_labels == 1)),
                "spoof": int(np.sum(val_labels == 0)),
                "total": len(val_labels),
            },
        }


# ============================================================
# Internal: Multi-directory ASVspoof Dataset
# ============================================================

class _MultiDirASVspoofDataset(Dataset):
    """ASVspoof dataset that handles samples from mixed directories.

    Since KFold merges train+dev splits, each sample may come from
    a different flac directory. This dataset resolves paths per-sample.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        data_root: Path,
        target_len: int = TARGET_SAMPLES,
        augment: bool = False,
        rawboost_params: dict[str, Any] | None = None,
        rawboost_algo: int = 4,
        is_eval: bool = False,
    ) -> None:
        self.samples = samples
        self.data_root = data_root
        self.target_len = target_len
        self.augment = augment
        self.is_eval = is_eval
        self.rawboost_algo = rawboost_algo

        self.rawboost = None
        self._audio_cache: dict[int, np.ndarray] = {}

        if augment:
            self.rawboost = RawBoost(params=rawboost_params or None, sr=SAMPLE_RATE)
        
        # Pre-load all pure (unaugmented) audio to RAM to save Disk I/O overhead
        self._preload_audio()

    def _preload_audio(self) -> None:
        """Pre-load ALL raw waveforms to RAM to avoid repeated disk reads.
        
        Augmentation (RawBoost) is NOT applied here to preserve randomness
        in every epoch.
        """
        from tqdm.auto import tqdm

        n = len(self.samples)
        logger.info(
            "[PreLoad] Loading %d raw audio files into RAM to avoid disk I/O bottleneck …", 
            n
        )

        for idx in tqdm(range(n), desc="Pre-loading raw audio to RAM", unit="samp"):
            sample = self.samples[idx]
            flac_path = self._resolve_flac_path(sample)
            waveform, _sr = sf.read(str(flac_path))
            self._audio_cache[idx] = waveform.astype(np.float32)

        # Estimate memory footprint
        mem_mb = sum(w.nbytes for w in self._audio_cache.values()) / (1024 ** 2)
        logger.info("[PreLoad] Done — %d raw audio cached (%.0f MB RAM used)", n, mem_mb)
        print(f"[PreLoad] {n} raw waveforms pre-loaded in RAM (≈{mem_mb:.0f} MB RAM)")

    def _resolve_flac_path(self, sample: dict[str, Any]) -> Path:
        """Get the .flac file path for a sample."""
        if sample.get("split") == "train":
            return self.data_root / "ASVspoof2019_LA_train" / "flac" / f"{sample['filename']}.flac"
        else:
            return self.data_root / "ASVspoof2019_LA_dev" / "flac" / f"{sample['filename']}.flac"

    def crop_or_pad(self, waveform: np.ndarray) -> np.ndarray:
        """Crop or pad waveform to target length."""
        x_len = waveform.shape[0]
        if x_len >= self.target_len:
            if self.is_eval:
                return waveform[: self.target_len]
            else:
                start = np.random.randint(0, x_len - self.target_len + 1)
                return waveform[start : start + self.target_len]
        else:
            num_repeats = (self.target_len // x_len) + 1
            return np.tile(waveform, num_repeats)[: self.target_len]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str]:
        sample = self.samples[index]
        filename = sample["filename"]
        label = sample["label"]

        # 1. Fetch RAW audio (from RAM cache if available, else disk)
        if index in self._audio_cache:
            waveform = self._audio_cache[index]
        else:
            flac_path = self._resolve_flac_path(sample)
            waveform, _sr = sf.read(str(flac_path))

        # 2. Crop or pad
        waveform = self.crop_or_pad(waveform)

        # 3. ON-THE-FLY Augmentation (only bonafide in training splits)
        # Fixes catastrophic overfitting issue by ensuring diverse 
        # augmentations every epoch.
        if self.augment and self.rawboost is not None and label == 1:
            waveform = self.rawboost(waveform, algo=self.rawboost_algo)

        x = torch.from_numpy(waveform.astype(np.float32))
        return x, label, filename


# ============================================================
# 4d. DataLoader Factory
# ============================================================

def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = False,
    seed: int | None = None,
    prefetch_factor: int | None = 4,
) -> DataLoader:
    """Create a PyTorch DataLoader with optimized settings.

    Args:
        dataset: PyTorch Dataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for faster GPU transfer.
        persistent_workers: Keep workers alive between epochs.
        drop_last: Drop last incomplete batch.
        seed: Random seed for reproducible shuffling.
        prefetch_factor: Batches to prefetch per worker (default 4).

    Returns:
        Configured DataLoader instance.
    """
    generator = None
    if seed is not None and shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        """Set unique seed per worker for reproducibility."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        import random
        random.seed(worker_seed)

    use_persistent = persistent_workers and num_workers > 0
    pf = prefetch_factor if num_workers > 0 else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
        prefetch_factor=pf,
    )


# ============================================================
# Smoke Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("src/dataloader.py — Smoke Test")
    print("=" * 60)

    # Test protocol parsing (will fail if data not present, that's OK)
    from pathlib import Path

    data_root = Path("data/ASVspoof2019_LA")
    train_proto = data_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"

    if train_proto.exists():
        samples = ASVspoof2019Dataset.parse_protocol(train_proto)
        print(f"✓ Parsed {len(samples)} training samples")
        print(f"  First sample: {samples[0]}")

        # Test KFoldManager
        kfm = KFoldManager(data_root, n_splits=5, seed=42)
        dist = kfm.get_label_distribution(0)
        print(f"✓ Fold 0 distribution: {dist}")

        # Test get_fold
        from src.utils import load_config
        cfg = load_config("configs/schema_b.yaml")
        train_ds, val_ds = kfm.get_fold(0, augment=True, config=cfg)
        print(f"✓ Fold 0: train={len(train_ds)}, val={len(val_ds)}")

        # Test single sample
        x, label, fname = train_ds[0]
        print(f"✓ Sample shape={x.shape}, label={label}, filename={fname}")
    else:
        print("⚠ Data not found at", data_root)
        print("  Run: python main.py --mode download --dataset asvspoof")

    print("\n✅ Dataloader smoke test complete!")
