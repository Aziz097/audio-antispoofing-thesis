"""
src/train.py
Training loop with K-Fold cross validation, AMP bf16, gradient
accumulation, wandb logging, early stopping, and checkpoint management.

Key features:
    - train_one_epoch: Single epoch loop with AMP + grad accumulation
    - validate: Validation with EER computation
    - run_fold: Full training for a single fold
    - run_kfold_experiment: Orchestrator for all K folds
"""

import logging
import time
from pathlib import Path

from tqdm.auto import tqdm
from typing import Any

import os

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.dataloader import KFoldManager, get_dataloader
from src.models import get_model
from src.utils import (
    AveragedCheckpoint,
    EarlyStopping,
    compute_eer,
    count_parameters,
    set_seed,
    vram_status,
)

logger = logging.getLogger(__name__)


# ============================================================
# Checkpoint Manager
# ============================================================

class CheckpointManager:
    """Manage top-K model checkpoints per fold.

    Keeps only the best `top_k` checkpoints sorted by monitored metric
    and provides averaged checkpoint generation.

    Args:
        save_dir: Directory to save checkpoints.
        top_k: Number of best checkpoints to keep.
        mode: 'min' to keep lowest metric, 'max' for highest.
    """

    def __init__(
        self, save_dir: Path, top_k: int = 5, mode: str = "min"
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.mode = mode
        # List of (metric_value, epoch, filepath)
        self._checkpoints: list[tuple[float, int, Path]] = []

    def _is_better(self, new_val: float, worst_val: float) -> bool:
        if self.mode == "min":
            return new_val < worst_val
        return new_val > worst_val

    @property
    def best_metric(self) -> float | None:
        if not self._checkpoints:
            return None
        if self.mode == "min":
            return min(c[0] for c in self._checkpoints)
        return max(c[0] for c in self._checkpoints)

    @property
    def checkpoint_paths(self) -> list[Path]:
        """Return paths of all stored checkpoints sorted by metric (best first)."""
        reverse = self.mode == "max"
        sorted_ckpts = sorted(self._checkpoints, key=lambda c: c[0], reverse=reverse)
        return [c[2] for c in sorted_ckpts]

    def save(
        self,
        state_dict: dict[str, Any],
        metric: float,
        epoch: int,
        model_name: str,
        fold: int,
    ) -> Path | None:
        """Save checkpoint if it qualifies for top-K.

        Args:
            state_dict: Model state_dict to save.
            metric: Metric value for this checkpoint.
            epoch: Current epoch number.
            model_name: Name of the model.
            fold: Fold index.

        Returns:
            Path to saved checkpoint, or None if not saved.
        """
        filename = f"{model_name}_fold{fold}_epoch{epoch:03d}_eer{metric:.4f}.pth"
        filepath = self.save_dir / filename

        if len(self._checkpoints) < self.top_k:
            torch.save(state_dict, str(filepath))
            self._checkpoints.append((metric, epoch, filepath))
            logger.info(
                "Checkpoint saved: %s (metric=%.4f, %d/%d slots)",
                filename, metric, len(self._checkpoints), self.top_k,
            )
            return filepath

        # Find worst checkpoint
        if self.mode == "min":
            worst_idx = max(range(len(self._checkpoints)), key=lambda i: self._checkpoints[i][0])
        else:
            worst_idx = min(range(len(self._checkpoints)), key=lambda i: self._checkpoints[i][0])

        worst_val, _, worst_path = self._checkpoints[worst_idx]

        if self._is_better(metric, worst_val):
            # Remove worst checkpoint file
            if worst_path.exists():
                worst_path.unlink()
            # Save new
            torch.save(state_dict, str(filepath))
            self._checkpoints[worst_idx] = (metric, epoch, filepath)
            logger.info(
                "Checkpoint replaced: %s (metric=%.4f, replaced %.4f)",
                filename, metric, worst_val,
            )
            return filepath

        return None


# ============================================================
# Training Loop
# ============================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp_dtype: torch.dtype,
    grad_accum_steps: int = 2,
    clip_grad_norm: float = 1.0,
    set_to_none: bool = True,
    epoch: int = 0,
    fold_idx: int = 0,
    max_epochs: int = 100,
    last_eer: float | None = None,
) -> dict[str, float]:
    """Train model for one epoch with AMP and gradient accumulation.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        scaler: GradScaler for AMP.
        device: Target device.
        amp_dtype: AMP dtype (torch.bfloat16).
        grad_accum_steps: Gradient accumulation steps.
        clip_grad_norm: Max gradient norm for clipping.
        set_to_none: Use set_to_none in zero_grad.
        epoch: Current epoch (for logging).
        fold_idx: Current fold index (for progress bar).
        max_epochs: Total epochs (for progress bar).
        last_eer: Latest validation EER (for progress bar display).

    Returns:
        Dictionary with training metrics:
            train/loss, train/accuracy, train/steps_per_sec.
    """
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    correct = 0
    total = 0
    step_count = 0
    epoch_start = time.time()

    optimizer.zero_grad(set_to_none=set_to_none)

    # Build tqdm description
    eer_str = f"{last_eer:.2f}%" if last_eer is not None else "N/A"
    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Fold {fold_idx} | Epoch {epoch+1}/{max_epochs}",
        unit="batch",
        leave=False,
    )

    for batch_idx, (waveforms, labels, _filenames) in pbar:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass with AMP
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(waveforms)
            loss = criterion(logits, labels)
            loss = loss / grad_accum_steps  # Scale for accumulation

        # Backward
        scaler.scale(loss).backward()

        # Accumulate metrics (use unscaled loss)
        batch_loss = loss.item() * grad_accum_steps
        total_loss += batch_loss
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        step_count += 1

        # Update progress bar every batch
        running_loss = total_loss / step_count
        pbar.set_postfix({
            "loss": f"{running_loss:.4f}",
            "EER": eer_str,
        })

        # Optimizer step every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            # Unscale for gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad_norm
            ).item()
            total_grad_norm += grad_norm

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=set_to_none)

    pbar.close()

    elapsed = time.time() - epoch_start
    avg_loss = total_loss / max(step_count, 1)
    accuracy = correct / max(total, 1)
    steps_per_sec = step_count / max(elapsed, 1e-6)

    return {
        "train/loss": avg_loss,
        "train/grad_norm": total_grad_norm / max(step_count // grad_accum_steps, 1),
        "train/accuracy": accuracy,
        "train/steps_per_sec": steps_per_sec,
        "train/epoch_time_sec": elapsed,
    }


# ============================================================
# Validation Loop
# ============================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    """Validate model and compute EER.

    Args:
        model: PyTorch model.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        device: Target device.
        amp_dtype: AMP dtype.

    Returns:
        Dictionary with validation metrics:
            val/loss, val/accuracy, val/eer.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    step_count = 0

    all_scores: list[float] = []
    all_labels: list[int] = []

    for waveforms, labels, _filenames in val_loader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(waveforms)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        step_count += 1

        # Scores for EER: use logit difference (bonafide_logit - spoof_logit)
        # Higher score → more likely bonafide
        scores = logits[:, 1] - logits[:, 0]
        all_scores.extend(scores.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Compute EER
    all_scores_np = np.array(all_scores)
    all_labels_np = np.array(all_labels)
    bonafide_scores = all_scores_np[all_labels_np == 1]
    spoof_scores = all_scores_np[all_labels_np == 0]

    if len(bonafide_scores) > 0 and len(spoof_scores) > 0:
        eer, _ = compute_eer(bonafide_scores, spoof_scores)
    else:
        eer = 50.0  # Fallback if no samples

    avg_loss = total_loss / max(step_count, 1)
    accuracy = correct / max(total, 1)

    return {
        "val/loss": avg_loss,
        "val/accuracy": accuracy,
        "val/eer": eer,
    }


# ============================================================
# Single Fold Training
# ============================================================

def run_fold(
    model_name: str,
    fold_idx: int,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    config: dict[str, Any],
    device: torch.device,
    wandb_run: Any | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Train and validate a model for a single fold.

    Args:
        model_name: Name of the model ('aasist', 'se_rawformer', etc.).
        fold_idx: Fold index.
        train_dataset: Training dataset for this fold.
        val_dataset: Validation dataset for this fold.
        config: Full config dictionary.
        device: Target device.
        wandb_run: Optional wandb run object.

    Returns:
        Dictionary with fold results:
            best_eer, best_epoch, checkpoint_paths, all_metrics.
    """
    train_cfg = config.get("train", {})
    es_cfg = config.get("early_stopping", {})
    ckpt_cfg = config.get("checkpoint", {})
    vram_cfg = config.get("vram", {})
    data_cfg = config.get("data", {})

    # ── DataLoaders ──────────────────────────────────────────
    _n_workers = data_cfg.get("num_workers", 2)
    logger.info("DataLoader num_workers=%d", _n_workers)

    train_loader = get_dataloader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 24),
        shuffle=True,
        num_workers=_n_workers,
        pin_memory=data_cfg.get("pin_memory", True),
        persistent_workers=data_cfg.get("persistent_workers", True),
        drop_last=True,
        seed=config.get("experiment", {}).get("seed", 42) + fold_idx,
    )
    val_loader = get_dataloader(
        val_dataset,
        batch_size=train_cfg.get("batch_size", 24),
        shuffle=False,
        num_workers=_n_workers,
        pin_memory=data_cfg.get("pin_memory", True),
        persistent_workers=False,
        drop_last=False,
    )

    # ── Model ────────────────────────────────────────────────
    model = get_model(model_name).to(device)
    if wandb_run is not None and fold_idx == 0:
        import wandb
        wandb.watch(model, log="gradients", log_freq=100)
    n_params = count_parameters(model)
    logger.info(
        "Model '%s' created: %s trainable params", model_name, f"{n_params:,}",
    )

    # Calculate path early to check for resume capability
    experiment_name = config.get("experiment", {}).get("name", "experiment")
    save_dir = Path(ckpt_cfg.get("save_dir", "experiments")) / experiment_name / model_name / f"fold_{fold_idx}" / "checkpoints"
    latest_ckpt_path = save_dir / f"{model_name}_fold{fold_idx}_latest.pth"

    # ── Resume: load full training state ────────────────────
    start_epoch = 0
    resumed_best_eer: float | None = None
    
    # Also support averaged topk check — if the fold is fully completed it will have this
    avg_path = save_dir / f"{model_name}_fold{fold_idx}_averaged_top{ckpt_cfg.get('top_k', 5)}.pth"
    fold_already_completed = False
    
    if resume:
        if avg_path.exists():
            logger.info("Fold %d is already fully completed (averaged checkpoint exists). Skipping fold.", fold_idx)
            fold_already_completed = True
            
            # Load the averaged state to extract dummy metrics 
            try:
                avg_state = torch.load(avg_path, map_location="cpu", weights_only=True)
                resumed_best_eer = avg_state.get("best_metric", 100.0) # default bad eer
            except Exception:
                pass
                
        elif latest_ckpt_path.exists():
            logger.info("Smart Resume triggered for fold %d. Loading %s", fold_idx, latest_ckpt_path.name)
            ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                # Full training state checkpoint
                model.load_state_dict(ckpt["model_state_dict"])
                start_epoch = ckpt.get("epoch", 0) + 1
                resumed_best_eer = ckpt.get("best_eer", None)
                logger.info(
                    "Resumed full state from checkpoint: %s (epoch=%d/%d, best_eer=%s)",
                    latest_ckpt_path.name, start_epoch, train_cfg.get("max_epochs", 100),
                    f"{resumed_best_eer:.4f}" if resumed_best_eer is not None else "N/A",
                )
            else:
                # Legacy: plain model state_dict
                model.load_state_dict(ckpt)
                logger.warning("Resumed model weights from legacy state dict, epoch status lost.")
        else:
            logger.info("Resume requested, but no latest.pth found for fold %d. Starting from scratch.", fold_idx)

    # ── Loss Function ────────────────────────────────────────
    # Compute inverse-frequency weights dynamically from the training fold so
    # the loss accounts for the ~10% bonafide / ~90% spoof imbalance.  Config
    # overrides are still honoured when both keys are explicitly set.
    loss_cfg = train_cfg.get("loss", {})
    override_bonafide = loss_cfg.get("weight_bonafide", None)
    override_spoof = loss_cfg.get("weight_spoof", None)
    if override_bonafide is not None and override_spoof is not None:
        # Explicit config override — use as-is
        w_bonafide = float(override_bonafide)
        w_spoof = float(override_spoof)
    else:
        # Inverse-frequency: w_c = N / (2 * N_c)
        train_labels = [s["label"] for s in train_dataset.samples]
        n_total = len(train_labels)
        n_bonafide = sum(train_labels)          # label=1
        n_spoof = n_total - n_bonafide          # label=0
        w_bonafide = n_total / (2.0 * max(n_bonafide, 1))
        w_spoof = n_total / (2.0 * max(n_spoof, 1))
        logger.info(
            "Dynamic loss weights — bonafide: %d samples (w=%.3f), "
            "spoof: %d samples (w=%.3f)",
            n_bonafide, w_bonafide, n_spoof, w_spoof,
        )
    loss_weights = torch.tensor([w_spoof, w_bonafide], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    # ── Optimizer ────────────────────────────────────────────
    opt_cfg = train_cfg.get("optimizer", {})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt_cfg.get("lr", 1e-4),
        weight_decay=opt_cfg.get("weight_decay", 1e-4),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
    )

    # ── AMP & GradScaler ─────────────────────────────────────
    amp_dtype_str = vram_cfg.get("amp_dtype", "bfloat16")
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16
    scaler = GradScaler("cuda", enabled=vram_cfg.get("grad_scaler", True))

    # Re-load optimizer + scaler state if resuming from a full checkpoint
    if resume and not fold_already_completed and latest_ckpt_path.exists():
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                logger.info("Optimizer state restored from checkpoint.")
            if "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
                logger.info("GradScaler state restored from checkpoint.")

    # ── Early Stopping ───────────────────────────────────────
    early_stopper = EarlyStopping(
        patience=es_cfg.get("patience", 10),
        min_delta=es_cfg.get("min_delta", 0.001),
        mode=es_cfg.get("mode", "min"),
    )

    # ── Checkpoint Manager ───────────────────────────────────
    ckpt_manager = CheckpointManager(
        save_dir=save_dir,
        top_k=ckpt_cfg.get("top_k", 5),
        mode=es_cfg.get("mode", "min"),
    )

    # ── Training Configuration ───────────────────────────────
    max_epochs = train_cfg.get("max_epochs", 100)
    grad_accum_steps = train_cfg.get("grad_accumulation_steps", 2)
    clip_grad_norm = train_cfg.get("clip_grad_norm", 1.0)
    set_to_none = vram_cfg.get("set_to_none", True)

    logger.info(
        "Starting fold %d: max_epochs=%d, batch_size=%d, grad_accum=%d, lr=%.1e%s",
        fold_idx, max_epochs, train_cfg.get("batch_size", 24),
        grad_accum_steps, opt_cfg.get("lr", 1e-4),
        f" [RESUMED from epoch {start_epoch}]" if start_epoch > 0 else "",
    )

    # ── Epoch Loop ───────────────────────────────────────────
    all_metrics: list[dict[str, float]] = []
    
    if fold_already_completed:
        logger.info("Fold %d complete: best_eer=%.2f%% (from previous run)", fold_idx, resumed_best_eer if resumed_best_eer else 0.0)
        return {
            "best_eer": resumed_best_eer if resumed_best_eer is not None else 100.0,
            "best_epoch": start_epoch,
            "checkpoint_paths": [str(avg_path)],
            "metrics": [],
        }

    # Fold metadata — log via summary (not a timestep)
    if wandb_run is not None:
        wandb_run.summary.update({
            f"fold{fold_idx}/n_train": len(train_dataset),
            f"fold{fold_idx}/n_val": len(val_dataset),
            f"fold{fold_idx}/batch_size": train_cfg.get("batch_size", 32),
            f"fold{fold_idx}/num_workers": _n_workers,
        })

    # ── Epoch Loop ───────────────────────────────────────────
    last_eer: float | None = None         # Track for tqdm display
    for epoch in range(start_epoch, max_epochs):
        torch.cuda.reset_peak_memory_stats()
        
        # Train
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_dtype=amp_dtype,
            grad_accum_steps=grad_accum_steps,
            clip_grad_norm=clip_grad_norm,
            set_to_none=set_to_none,
            epoch=epoch,
            fold_idx=fold_idx,
            max_epochs=max_epochs,
            last_eer=last_eer,
        )

        # Validate
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            amp_dtype=amp_dtype,
        )
        last_eer = val_metrics["val/eer"]

        # Merge metrics
        global_step = fold_idx * max_epochs + epoch
        epoch_metrics = {
            "global_step": global_step,
            "epoch": epoch,
            "fold": fold_idx,
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/scaler_scale": scaler.get_scale(),
            **train_metrics,
            **val_metrics,
            **vram_status(label=f"fold{fold_idx}_ep{epoch}", reset_peak=False),
        }
        all_metrics.append(epoch_metrics)

        # Log to wandb — use a fold-offset global step so fold N's epoch 0
        # does not overwrite fold (N-1)'s epoch 0 in the wandb timeline.
        if wandb_run is not None:
            wandb_run.log(epoch_metrics, step=global_step)

        # Console log
        logger.info(
            "[Fold %d | Epoch %d/%d] "
            "train_loss=%.4f, train_acc=%.4f | "
            "val_loss=%.4f, val_acc=%.4f, val_eer=%.2f%% | "
            "%.1f steps/s",
            fold_idx, epoch + 1, max_epochs,
            train_metrics["train/loss"], train_metrics["train/accuracy"],
            val_metrics["val/loss"], val_metrics["val/accuracy"],
            val_metrics["val/eer"],
            train_metrics["train/steps_per_sec"],
        )

        # Checkpoint (top-K best by EER)
        ckpt_manager.save(
            state_dict=model.state_dict(),
            metric=val_metrics["val/eer"],
            epoch=epoch,
            model_name=model_name,
            fold=fold_idx,
        )

        # Save rolling full-state checkpoint for crash recovery / resume.
        # Overwrites every epoch — always reflects the latest completed epoch.
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_eer": early_stopper.best_value,
            "val_eer": val_metrics["val/eer"],
            "config": config,
        }, str(latest_ckpt_path))

        # Early stopping
        if early_stopper.step(val_metrics["val/eer"]):
            logger.info(
                "Early stopping triggered at epoch %d. "
                "Best EER=%.2f%% at epoch %d.",
                epoch + 1, early_stopper.best_value, early_stopper.best_epoch,
            )
            break

    # ── Generate averaged checkpoint ─────────────────────────
    avg_state = None
    if ckpt_cfg.get("average_top_k", True) and len(ckpt_manager.checkpoint_paths) > 0:
        avg_state = AveragedCheckpoint.load_and_average(ckpt_manager.checkpoint_paths)
        avg_path = save_dir / f"{model_name}_fold{fold_idx}_averaged_top{ckpt_cfg.get('top_k', 5)}.pth"
        torch.save(avg_state, str(avg_path))
        logger.info("Averaged checkpoint saved: %s", avg_path.name)

    if wandb_run is not None:
        import wandb
        wandb_run.log({
            f"fold_summary/fold{fold_idx}_best_eer": early_stopper.best_value,
            f"fold_summary/fold{fold_idx}_best_epoch": early_stopper.best_epoch,
            f"fold_summary/fold{fold_idx}_n_epochs": epoch + 1,
        })

    return {
        "fold": fold_idx,
        "best_eer": early_stopper.best_value,
        "best_epoch": early_stopper.best_epoch,
        "checkpoint_paths": ckpt_manager.checkpoint_paths,
        "averaged_checkpoint": avg_state,
        "all_metrics": all_metrics,
    }


# ============================================================
# K-Fold Experiment Orchestrator
# ============================================================

def run_kfold_experiment(
    model_name: str,
    config: dict[str, Any],
    device: torch.device,
    wandb_run: Any | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run full K-Fold cross-validation experiment for one model.

    This is the top-level training function called from main.py.

    Args:
        model_name: Name of the model to train.
        config: Full config dictionary.
        device: Target device.
        wandb_run: Optional wandb run object.

    Returns:
        Dictionary with experiment results:
            model_name, fold_results, mean_eer, std_eer.
    """
    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    kfold_cfg = config.get("kfold", {})
    n_splits = kfold_cfg.get("n_splits", 5)
    augment = config.get("augmentation", {}).get("enabled", False)

    # Data root
    data_root = Path(config.get("data", {}).get("root", "data/ASVspoof2019_LA"))

    # Initialize KFoldManager
    kfold_manager = KFoldManager(
        data_root=data_root,
        n_splits=n_splits,
        seed=seed,
    )

    logger.info(
        "Starting K-Fold experiment: model=%s, schema=%s, n_splits=%d, augment=%s",
        model_name,
        config.get("experiment", {}).get("name", "unknown"),
        n_splits,
        augment,
    )

    fold_results: list[dict[str, Any]] = []

    for fold_idx in range(n_splits):
        logger.info("=" * 60)
        logger.info("FOLD %d / %d", fold_idx + 1, n_splits)
        logger.info("=" * 60)

        # Re-seed for each fold (deterministic initialization)
        set_seed(seed + fold_idx)

        # Get fold datasets
        train_dataset, val_dataset = kfold_manager.get_fold(
            fold_idx=fold_idx,
            augment=augment,
            config=config,
        )

        # Log label distribution
        dist = kfold_manager.get_label_distribution(fold_idx)
        logger.info("Label distribution: %s", dist)

        # Run fold
        result = run_fold(
            model_name=model_name,
            fold_idx=fold_idx,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            device=device,
            wandb_run=wandb_run,
            resume=resume,
        )
        fold_results.append(result)

        # Log fold summary
        logger.info(
            "Fold %d complete: best_eer=%.2f%% at epoch %d",
            fold_idx, result["best_eer"], result["best_epoch"],
        )

        # Explicit cleanup — del datasets so their reference counts hit zero
        # before empty_cache; model/optimizer/loaders are local to run_fold
        # and are already released when it returned.
        del train_dataset, val_dataset
        torch.cuda.empty_cache()

    # ── Aggregate results ────────────────────────────────────
    fold_eers = [r["best_eer"] for r in fold_results]
    mean_eer = float(np.mean(fold_eers))
    std_eer = float(np.std(fold_eers))

    logger.info("=" * 60)
    logger.info("K-FOLD EXPERIMENT COMPLETE")
    logger.info("Model: %s", model_name)
    logger.info("Per-fold EERs: %s", [f"{e:.2f}%" for e in fold_eers])
    logger.info("Mean EER: %.2f%% ± %.2f%%", mean_eer, std_eer)
    logger.info("=" * 60)

    # Log summary to wandb
    if wandb_run is not None:
        wandb_run.summary[f"{model_name}/mean_eer"] = mean_eer
        wandb_run.summary[f"{model_name}/std_eer"] = std_eer
        for i, eer in enumerate(fold_eers):
            wandb_run.summary[f"{model_name}/fold_{i}_eer"] = eer

    return {
        "model_name": model_name,
        "fold_results": fold_results,
        "mean_eer": mean_eer,
        "std_eer": std_eer,
        "fold_eers": fold_eers,
    }


# ============================================================
# Smoke Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("src/train.py — Smoke Test (structure only)")
    print("=" * 60)

    # Test CheckpointManager
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(Path(tmpdir), top_k=3, mode="min")
        dummy_state = {"w": torch.randn(10)}

        # Save 5 checkpoints, only 3 should survive
        mgr.save(dummy_state, metric=5.0, epoch=0, model_name="test", fold=0)
        mgr.save(dummy_state, metric=3.0, epoch=1, model_name="test", fold=0)
        mgr.save(dummy_state, metric=4.0, epoch=2, model_name="test", fold=0)
        mgr.save(dummy_state, metric=2.0, epoch=3, model_name="test", fold=0)
        mgr.save(dummy_state, metric=6.0, epoch=4, model_name="test", fold=0)

        assert len(mgr.checkpoint_paths) == 3
        assert mgr.best_metric == 2.0
        print(f"✓ CheckpointManager: {len(mgr.checkpoint_paths)} checkpoints, best={mgr.best_metric}")

    print("\n✅ train.py structure smoke test passed!")
