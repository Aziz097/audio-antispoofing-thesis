"""
src/utils.py
Utility functions for the audio anti-spoofing thesis project.

Includes:
    - set_seed: Reproducibility
    - vram_status: GPU memory monitoring
    - RawBoost: Data augmentation (Tak et al., ICASSP 2022)
    - EarlyStopping: Training callback
    - count_parameters / get_model_stats: Model profiling
    - compute_eer / compute_min_tdcf: Evaluation metrics
    - AveragedCheckpoint: Model averaging for final evaluation
"""

import copy
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from torchinfo import summary as torchinfo_summary

logger = logging.getLogger(__name__)

# ============================================================
# 3a. Seed Management
# ============================================================

def set_seed(seed: int) -> None:
    """Set random seed for full reproducibility across all libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seed set to %d", seed)


# ============================================================
# 3b. VRAM Monitoring
# ============================================================

def vram_status(label: str = "") -> dict[str, float]:
    """Return current GPU VRAM usage in GB for wandb logging.

    Args:
        label: Optional label prefix for log messages.

    Returns:
        Dictionary with keys: allocated_gb, reserved_gb, total_gb.
        Returns zeros if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return {"vram/allocated_gb": 0.0, "vram/reserved_gb": 0.0, "vram/total_gb": 0.0}

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    if label:
        logger.info(
            "[%s] VRAM: %.2f GB allocated, %.2f GB reserved, %.2f GB total",
            label, allocated, reserved, total,
        )

    return {
        "vram/allocated_gb": round(allocated, 4),
        "vram/reserved_gb": round(reserved, 4),
        "vram/total_gb": round(total, 4),
    }


# ============================================================
# 3c. RawBoost — Data Augmentation
# ============================================================

class RawBoost:
    """RawBoost data augmentation for raw audio waveforms.

    Reference:
        Tak et al., "RawBoost: A Raw Data Boosting and Augmentation Method
        applied to Automatic Speaker Verification Anti-Spoofing",
        ICASSP 2022, pp. 6382–6386.

    All methods operate on numpy arrays and return numpy arrays.
    """

    # ── Default hyperparameters ──────────────────────────────
    DEFAULT_PARAMS: dict[str, Any] = {
        # LnL (algo 1)
        "nBands": 5,
        "minF": 20,
        "maxF": 8000,
        "minBW": 100,
        "maxBW": 1000,
        "minCoeff": 10,
        "maxCoeff": 100,
        "minG": 0,
        "maxG": 0,
        "minBiasLinNonLin": 5,
        "maxBiasLinNonLin": 20,
        "N_f": 5,
        # ISD (algo 2)
        "P": 10,
        "g_sd": 2,
        # SSI (algo 3)
        "SNRmin": 10,
        "SNRmax": 40,
    }

    def __init__(self, params: dict[str, Any] | None = None, sr: int = 16000) -> None:
        """Initialize RawBoost with optional custom parameters.

        Args:
            params: Dictionary of RawBoost hyperparameters.
                    Missing keys fall back to DEFAULT_PARAMS.
            sr: Sample rate in Hz.
        """
        self.sr = sr
        self.p = {**self.DEFAULT_PARAMS, **(params or {})}

    # ── Internal helpers ─────────────────────────────────────

    @staticmethod
    def _rand_range(x1: float, x2: float, integer: bool = False) -> float | int:
        """Draw a single uniform random value in [x1, x2]."""
        y = np.random.uniform(low=x1, high=x2)
        return int(y) if integer else float(y)

    @staticmethod
    def _norm_wav(x: np.ndarray, always: bool = False) -> np.ndarray:
        """Normalize waveform to [-1, 1]."""
        peak = np.amax(np.abs(x))
        if always or peak > 1.0:
            x = x / (peak + 1e-12)
        return x

    def _gen_notch_coeffs(
        self,
        n_bands: int,
        min_f: float, max_f: float,
        min_bw: float, max_bw: float,
        min_coeff: int, max_coeff: int,
        min_g: float, max_g: float,
    ) -> np.ndarray:
        """Generate FIR notch filter coefficients."""
        b = np.array([1.0])
        for _ in range(n_bands):
            fc = self._rand_range(min_f, max_f)
            bw = self._rand_range(min_bw, max_bw)
            c = self._rand_range(min_coeff, max_coeff, integer=True)
            if c % 2 == 0:
                c += 1
            f1 = max(fc - bw / 2, 1.0 / 1000)
            f2 = min(fc + bw / 2, self.sr / 2 - 1.0 / 1000)
            b = np.convolve(
                signal.firwin(c, [float(f1), float(f2)], window="hamming", fs=self.sr),
                b,
            )
        g = self._rand_range(min_g, max_g)
        _, h = signal.freqz(b, 1, fs=self.sr)
        b = (10 ** (g / 20)) * b / (np.amax(np.abs(h)) + 1e-12)
        return b

    @staticmethod
    def _filter_fir(x: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Apply FIR filter with zero-phase-like truncation."""
        n = b.shape[0] + 1
        x_pad = np.pad(x, (0, n), mode="constant")
        y = signal.lfilter(b, 1, x_pad)
        y = y[int(n / 2) : int(y.shape[0] - n / 2)]
        return y

    # ── Public augmentation algorithms ───────────────────────

    def algo1(self, x: np.ndarray) -> np.ndarray:
        """Linear and Non-Linear Convolutive Noise (LnL).

        Args:
            x: Input waveform, shape (T,).

        Returns:
            Augmented waveform, shape (T,).
        """
        p = self.p
        y = np.zeros(x.shape[0], dtype=np.float64)
        for i in range(p["N_f"]):
            min_g = p["minG"] - (p["minBiasLinNonLin"] if i >= 1 else 0)
            max_g = p["maxG"] - (p["maxBiasLinNonLin"] if i >= 1 else 0)
            min_g, max_g = min(min_g, max_g), max(min_g, max_g)
            b = self._gen_notch_coeffs(
                p["nBands"], p["minF"], p["maxF"],
                p["minBW"], p["maxBW"], p["minCoeff"], p["maxCoeff"],
                min_g, max_g,
            )
            y = y + self._filter_fir(np.power(x, i + 1), b)
        y = y - np.mean(y)
        y = self._norm_wav(y, always=False)
        return y

    def algo2(self, x: np.ndarray) -> np.ndarray:
        """Impulsive Signal-Dependent Additive Noise (ISD).

        Args:
            x: Input waveform, shape (T,).

        Returns:
            Augmented waveform, shape (T,).
        """
        p = self.p
        beta = self._rand_range(0, p["P"])
        y = copy.deepcopy(x)
        x_len = x.shape[0]
        n = int(x_len * (beta / 100))
        if n == 0:
            return y
        indices = np.random.permutation(x_len)[:n]
        f_r = np.multiply(
            (2 * np.random.rand(indices.shape[0]) - 1),
            (2 * np.random.rand(indices.shape[0]) - 1),
        )
        r = p["g_sd"] * x[indices] * f_r
        y[indices] = x[indices] + r
        y = self._norm_wav(y, always=False)
        return y

    def algo3(self, x: np.ndarray) -> np.ndarray:
        """Stationary Signal-Independent Additive Noise (SSI).

        Args:
            x: Input waveform, shape (T,).

        Returns:
            Augmented waveform, shape (T,).
        """
        p = self.p
        noise = np.random.normal(0, 1, x.shape[0])
        b = self._gen_notch_coeffs(
            p["nBands"], p["minF"], p["maxF"],
            p["minBW"], p["maxBW"], p["minCoeff"], p["maxCoeff"],
            p["minG"], p["maxG"],
        )
        noise = self._filter_fir(noise, b)
        noise = self._norm_wav(noise, always=True)
        snr = self._rand_range(p["SNRmin"], p["SNRmax"])
        x_norm = np.linalg.norm(x, 2)
        noise_norm = np.linalg.norm(noise, 2) + 1e-12
        noise = noise / noise_norm * x_norm / (10.0 ** (0.05 * snr))
        return x + noise

    def algo4(self, x: np.ndarray) -> np.ndarray:
        """Series combination: LnL → ISD → SSI (Algorithm 4).

        This is the primary augmentation used in Schema B.

        Args:
            x: Input waveform, shape (T,).

        Returns:
            Augmented waveform, shape (T,).
        """
        x = self.algo1(x)
        x = self.algo2(x)
        x = self.algo3(x)
        return x

    def __call__(self, x: np.ndarray, algo: int = 4) -> np.ndarray:
        """Apply specified RawBoost algorithm.

        Args:
            x: Input waveform, shape (T,).
            algo: Algorithm number (1–4).

        Returns:
            Augmented waveform.
        """
        dispatch = {1: self.algo1, 2: self.algo2, 3: self.algo3, 4: self.algo4}
        if algo not in dispatch:
            raise ValueError(f"RawBoost algo must be 1–4, got {algo}")
        return dispatch[algo](x)


# ============================================================
# 3d. Early Stopping
# ============================================================

class EarlyStopping:
    """Early stopping callback that monitors a metric.

    Args:
        patience: Number of epochs to wait after last improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' for metrics where lower is better (e.g., EER),
              'max' for metrics where higher is better.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self._best_value: float = float("inf") if mode == "min" else float("-inf")
        self._best_epoch: int = 0
        self._waited: int = 0
        self._triggered: bool = False
        self._current_epoch: int = 0

    def step(self, metric: float) -> bool:
        """Check if training should stop.

        Args:
            metric: Current metric value.

        Returns:
            True if training should stop (patience exhausted).
        """
        improved = False
        if self.mode == "min":
            improved = metric < (self._best_value - self.min_delta)
        else:
            improved = metric > (self._best_value + self.min_delta)

        if improved:
            self._best_value = metric
            self._best_epoch = self._current_epoch
            self._waited = 0
        else:
            self._waited += 1

        self._current_epoch += 1

        if self._waited >= self.patience:
            self._triggered = True
            return True
        return False

    def reset(self) -> None:
        """Reset all internal state."""
        self._best_value = float("inf") if self.mode == "min" else float("-inf")
        self._best_epoch = 0
        self._waited = 0
        self._triggered = False
        self._current_epoch = 0

    @property
    def best_value(self) -> float:
        """Best metric value seen so far."""
        return self._best_value

    @property
    def best_epoch(self) -> int:
        """Epoch at which best metric was observed."""
        return self._best_epoch

    @property
    def triggered(self) -> bool:
        """Whether early stopping has been triggered."""
        return self._triggered

    @property
    def waited(self) -> int:
        """Number of epochs waited since last improvement."""
        return self._waited


# ============================================================
# 3e. Parameter Counting
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a PyTorch model.

    Args:
        model: PyTorch model.

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# 3f. Model Statistics (Params + MACs)
# ============================================================

def get_model_stats(model: nn.Module, input_shape: tuple[int, ...]) -> dict[str, int]:
    """Get model parameter count and MACs using torchinfo.

    Args:
        model: PyTorch model.
        input_shape: Input tensor shape including batch dimension,
                     e.g. (1, 64000).

    Returns:
        Dictionary with 'params' (int) and 'macs' (int).
    """
    device = next(model.parameters()).device
    info = torchinfo_summary(
        model,
        input_size=input_shape,
        device=device,
        verbose=0,
    )
    return {
        "params": info.trainable_params,
        "macs": info.total_mult_adds,
    }


# ============================================================
# 3g. Equal Error Rate (EER)
# ============================================================

def compute_eer(
    target_scores: np.ndarray,
    nontarget_scores: np.ndarray,
) -> tuple[float, float]:
    """Compute Equal Error Rate and the corresponding threshold.

    Higher scores should indicate stronger support for the
    bonafide (positive) hypothesis.

    Args:
        target_scores: Scores for bonafide (positive) trials.
        nontarget_scores: Scores for spoof (negative) trials.

    Returns:
        Tuple of (eer_percent, threshold).
    """
    # Guard against degenerate cases (e.g. all-equal scores at epoch 0)
    if len(target_scores) < 2 or len(nontarget_scores) < 2:
        return 50.0, 0.0
    if np.std(target_scores) < 1e-6 and np.std(nontarget_scores) < 1e-6:
        return 50.0, 0.0

    # Compute DET curve
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    # False rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )

    # Find EER
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))

    return float(eer * 100), float(thresholds[min_index])


# ============================================================
# 3h. Minimum t-DCF
# ============================================================

def compute_min_tdcf(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
    asv_score_file: str | Path | None = None,
    Pfa_asv: float | None = None,
    Pmiss_asv: float | None = None,
    Pmiss_spoof_asv: float | None = None,
) -> float:
    """Compute minimum Tandem Detection Cost Function (t-DCF).

    Either provide asv_score_file to compute ASV error rates from the
    official score file, or provide Pfa_asv, Pmiss_asv, Pmiss_spoof_asv
    directly.

    Reference:
        ASVspoof 2019 evaluation plan — official scoring script.

    Args:
        bonafide_scores: CM scores for bonafide (positive) trials.
        spoof_scores: CM scores for spoof (negative) trials.
        asv_score_file: Path to ASV score file.
        Pfa_asv: False alarm rate of ASV system.
        Pmiss_asv: Miss rate of ASV system.
        Pmiss_spoof_asv: Spoof miss rate of ASV system.

    Returns:
        Minimum normalized t-DCF value.
    """
    # t-DCF cost model parameters (ASVspoof 2019 defaults)
    P_SPOOF = 0.05
    cost_model = {
        "Pspoof": P_SPOOF,
        "Ptar": (1 - P_SPOOF) * 0.99,
        "Pnon": (1 - P_SPOOF) * 0.01,
        "Cmiss": 1,
        "Cfa": 10,
        "Cmiss_asv": 1,
        "Cfa_asv": 10,
        "Cmiss_cm": 1,
        "Cfa_cm": 10,
    }

    # If ASV score file is provided, compute ASV error rates
    if asv_score_file is not None:
        asv_data = np.genfromtxt(str(asv_score_file), dtype=str)
        asv_keys = asv_data[:, 1]
        asv_scores = asv_data[:, 2].astype(float)

        tar_asv = asv_scores[asv_keys == "target"]
        non_asv = asv_scores[asv_keys == "nontarget"]
        spoof_asv = asv_scores[asv_keys == "spoof"]

        # EER threshold of ASV
        _, asv_threshold = compute_eer(tar_asv, non_asv)
        # ASV threshold is in percentage from compute_eer but we need raw threshold
        # Recompute using raw scores
        n_scores = tar_asv.size + non_asv.size
        all_scores_asv = np.concatenate((tar_asv, non_asv))
        labels_asv = np.concatenate(
            (np.ones(tar_asv.size), np.zeros(non_asv.size))
        )
        indices_asv = np.argsort(all_scores_asv, kind="mergesort")
        labels_asv = labels_asv[indices_asv]
        tar_sums = np.cumsum(labels_asv)
        non_sums = non_asv.size - (np.arange(1, n_scores + 1) - tar_sums)
        frr = np.concatenate((np.atleast_1d(0), tar_sums / tar_asv.size))
        far_curve = np.concatenate(
            (np.atleast_1d(1), non_sums / non_asv.size)
        )
        thresholds_asv = np.concatenate(
            (np.atleast_1d(all_scores_asv[indices_asv[0]] - 0.001), all_scores_asv[indices_asv])
        )
        abs_diffs = np.abs(frr - far_curve)
        min_idx = np.argmin(abs_diffs)
        asv_threshold = thresholds_asv[min_idx]

        Pfa_asv = float(np.sum(non_asv >= asv_threshold) / non_asv.size)
        Pmiss_asv = float(np.sum(tar_asv < asv_threshold) / tar_asv.size)
        Pmiss_spoof_asv = float(np.sum(spoof_asv < asv_threshold) / spoof_asv.size)

    if Pfa_asv is None or Pmiss_asv is None or Pmiss_spoof_asv is None:
        raise ValueError(
            "Must provide either asv_score_file or all three ASV error rates."
        )

    # Compute CM DET curve
    n_cm = bonafide_scores.size + spoof_scores.size
    all_cm = np.concatenate((bonafide_scores, spoof_scores))
    labels_cm = np.concatenate(
        (np.ones(bonafide_scores.size), np.zeros(spoof_scores.size))
    )
    indices_cm = np.argsort(all_cm, kind="mergesort")
    labels_cm = labels_cm[indices_cm]
    tar_cm_sums = np.cumsum(labels_cm)
    non_cm_sums = spoof_scores.size - (np.arange(1, n_cm + 1) - tar_cm_sums)

    Pmiss_cm = np.concatenate(
        (np.atleast_1d(0), tar_cm_sums / bonafide_scores.size)
    )
    Pfa_cm = np.concatenate(
        (np.atleast_1d(1), non_cm_sums / spoof_scores.size)
    )

    # t-DCF constants
    C1 = (
        cost_model["Ptar"] * (cost_model["Cmiss_cm"] - cost_model["Cmiss_asv"] * Pmiss_asv)
        - cost_model["Pnon"] * cost_model["Cfa_asv"] * Pfa_asv
    )
    C2 = cost_model["Cfa_cm"] * cost_model["Pspoof"] * (1 - Pmiss_spoof_asv)

    # t-DCF curve
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm

    # Normalized t-DCF
    tDCF_norm = tDCF / min(C1, C2)

    return float(np.min(tDCF_norm))


# ============================================================
# 3i. Averaged Checkpoint
# ============================================================

class AveragedCheckpoint:
    """Load and average multiple model checkpoints.

    Used to average top-K checkpoints per fold for final evaluation.
    """

    @staticmethod
    def load_and_average(checkpoint_paths: list[Path]) -> dict[str, torch.Tensor]:
        """Load multiple checkpoint state_dicts and return their average.

        Args:
            checkpoint_paths: List of paths to .pth checkpoint files.

        Returns:
            Averaged state_dict.

        Raises:
            ValueError: If no checkpoint paths provided.
            FileNotFoundError: If a checkpoint file doesn't exist.
        """
        if not checkpoint_paths:
            raise ValueError("No checkpoint paths provided for averaging.")

        # Validate all paths exist
        for p in checkpoint_paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"Checkpoint not found: {p}")

        # Load first checkpoint as base
        avg_state = torch.load(
            str(checkpoint_paths[0]), map_location="cpu", weights_only=True
        )

        # Accumulate remaining checkpoints
        for ckpt_path in checkpoint_paths[1:]:
            state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
            for key in avg_state:
                avg_state[key] = avg_state[key] + state[key]

        # Average — only divide floating-point / complex tensors; integer
        # tensors such as BatchNorm's num_batches_tracked must not be cast
        # to float, so we use integer floor-division for those.
        n = len(checkpoint_paths)
        for key in avg_state:
            t = avg_state[key]
            if t.is_floating_point() or t.is_complex():
                avg_state[key] = t / n
            else:
                avg_state[key] = torch.div(t, n, rounding_mode="trunc")

        logger.info("Averaged %d checkpoints.", n)
        return avg_state


# ============================================================
# Config Loader Helper
# ============================================================

def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a nested dictionary.

    Args:
        config_path: Path to .yaml config file.

    Returns:
        Configuration dictionary.
    """
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


class DotDict(dict):
    """Dot-notation access for nested dictionaries.

    Example:
        cfg = DotDict({"train": {"lr": 0.001}})
        cfg.train.lr  # 0.001
    """

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict):
                return DotDict(val)
            return val
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


# ============================================================
# Smoke Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("src/utils.py — Smoke Test")
    print("=" * 60)

    # Test set_seed
    set_seed(42)
    print("✓ set_seed(42)")

    # Test vram_status
    status = vram_status(label="smoke_test")
    print(f"✓ vram_status: {status}")

    # Test RawBoost
    rb = RawBoost()
    dummy_wav = np.random.randn(64000).astype(np.float64)
    aug1 = rb.algo1(dummy_wav)
    aug2 = rb.algo2(dummy_wav)
    aug3 = rb.algo3(dummy_wav)
    aug4 = rb.algo4(dummy_wav)
    print(f"✓ RawBoost algo1 shape: {aug1.shape}")
    print(f"✓ RawBoost algo4 shape: {aug4.shape}")

    # Test EarlyStopping
    es = EarlyStopping(patience=3, min_delta=0.01, mode="min")
    for val in [5.0, 4.0, 3.5, 3.5, 3.5, 3.5]:
        should_stop = es.step(val)
        if should_stop:
            print(f"✓ EarlyStopping triggered at epoch {es.best_epoch}, best={es.best_value}")
            break
    assert es.triggered, "EarlyStopping should have triggered"
    print(f"✓ EarlyStopping: best_value={es.best_value}, best_epoch={es.best_epoch}")

    # Test compute_eer
    np.random.seed(42)
    bonafide = np.random.randn(1000) + 1
    spoof = np.random.randn(1000) - 1
    eer, thr = compute_eer(bonafide, spoof)
    print(f"✓ EER: {eer:.2f}%, threshold: {thr:.4f}")

    # Test count_parameters
    simple_model = nn.Linear(64, 2)
    n_params = count_parameters(simple_model)
    print(f"✓ count_parameters(Linear(64,2)): {n_params}")

    # Test load_config
    cfg = load_config(Path(__file__).parent.parent / "configs" / "schema_b.yaml")
    dot_cfg = DotDict(cfg)
    print(f"✓ Config loaded: augmentation.enabled={dot_cfg.augmentation.enabled}")
    print(f"✓ Config loaded: train.batch_size={dot_cfg.train.batch_size}")

    print("\n✅ All smoke tests passed!")
