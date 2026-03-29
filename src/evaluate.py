"""
src/evaluate.py
Evaluation pipeline: load averaged checkpoints, evaluate on
ASVspoof 2019 LA eval set + In-the-Wild dataset, compute EER
and min t-DCF, run inference benchmark.

Key functions:
    - evaluate_model: Evaluate a single model on a dataset
    - evaluate_all_folds: Evaluate all folds and aggregate
    - run_inference_benchmark: Measure latency and throughput
    - run_full_evaluation: Top-level evaluation orchestrator
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataloader import ASVspoof2019Dataset, InTheWildDataset, get_dataloader
from src.models import get_model
from src.utils import (
    AveragedCheckpoint,
    compute_eer,
    compute_min_tdcf,
    count_parameters,
    get_model_stats,
    set_seed,
    vram_status,
)

logger = logging.getLogger(__name__)


# ============================================================
# Single-Model Evaluation
# ============================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
    fold_idx: int = 0,
    dataset_name: str = "",
) -> dict[str, Any]:
    """Evaluate a model and collect per-sample scores.

    Args:
        model: Model in eval mode.
        dataloader: Evaluation DataLoader.
        device: Target device.
        amp_dtype: AMP dtype.

    Returns:
        Dictionary with:
            scores: np.ndarray of per-sample scores (bonafide - spoof logits).
            labels: np.ndarray of per-sample labels.
            filenames: list of per-sample filenames.
            predictions: np.ndarray of predicted labels.
    """
    model.eval()
    all_scores: list[float] = []
    all_labels: list[int] = []
    all_filenames: list[str] = []
    all_preds: list[int] = []

    pbar = tqdm(
        dataloader, 
        desc=f"Eval Fold {fold_idx} ({dataset_name})", 
        unit="batch", 
        leave=False,
    )
    
    for waveforms, labels, filenames in pbar:
        waveforms = waveforms.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(waveforms)

        # Score = bonafide_logit - spoof_logit (higher → more bonafide)
        scores = (logits[:, 1] - logits[:, 0]).cpu().tolist()
        preds = logits.argmax(dim=1).cpu().tolist()

        all_scores.extend(scores)
        all_labels.extend(labels.tolist() if isinstance(labels, torch.Tensor) else labels)
        all_filenames.extend(filenames)
        all_preds.extend(preds)
        
        pbar.set_postfix({"samples": len(all_scores)})

    pbar.close()

    return {
        "scores": np.array(all_scores),
        "labels": np.array(all_labels),
        "filenames": all_filenames,
        "predictions": np.array(all_preds),
    }


# ============================================================
# Metrics Computation
# ============================================================

def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    asv_score_file: str | Path | None = None,
) -> dict[str, float]:
    """Compute EER and optionally min t-DCF from scores and labels.

    Args:
        scores: Per-sample scores.
        labels: Per-sample labels (1=bonafide, 0=spoof).
        asv_score_file: Path to ASV score file for t-DCF (optional).

    Returns:
        Dictionary with: eer (%), eer_threshold, accuracy (%).
        Optionally: min_tdcf.
    """
    bonafide_scores = scores[labels == 1]
    spoof_scores = scores[labels == 0]

    eer, threshold = compute_eer(bonafide_scores, spoof_scores)

    # Accuracy at EER threshold
    preds = (scores >= threshold).astype(int)
    accuracy = float(np.mean(preds == labels) * 100)

    metrics = {
        "eer": eer,
        "eer_threshold": threshold,
        "accuracy": accuracy,
        "n_bonafide": int(len(bonafide_scores)),
        "n_spoof": int(len(spoof_scores)),
    }

    # min t-DCF (only if ASV score file is provided)
    if asv_score_file is not None and Path(asv_score_file).exists():
        try:
            min_tdcf = compute_min_tdcf(
                bonafide_scores=bonafide_scores,
                spoof_scores=spoof_scores,
                asv_score_file=asv_score_file,
            )
            metrics["min_tdcf"] = min_tdcf
        except Exception as e:
            logger.warning("Could not compute min t-DCF: %s", e)

    return metrics


# ============================================================
# Inference Benchmark
# ============================================================

def run_inference_benchmark(
    model: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
    n_samples: int = 100,
    warmup_samples: int = 10,
    input_length: int = 64000,
) -> dict[str, float]:
    """Benchmark model inference latency and throughput.

    Args:
        model: Model in eval mode.
        device: Target device.
        amp_dtype: AMP dtype.
        n_samples: Number of samples to benchmark.
        warmup_samples: Number of warmup passes (not timed).
        input_length: Input waveform length.

    Returns:
        Dictionary with:
            latency_ms_mean, latency_ms_std, latency_ms_p95,
            throughput_samples_per_sec, rtf (real-time factor).
    """
    model.eval()
    dummy_input = torch.randn(1, input_length, device=device)

    # Warmup — no_grad prevents gradient graph build-up that would inflate latency
    with torch.no_grad():
        for _ in range(warmup_samples):
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _ = model(dummy_input)
    torch.cuda.synchronize()

    # Benchmark — no_grad is required for accurate latency; gradient tracking
    # both allocates extra memory and adds overhead to every kernel launch.
    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(n_samples):
            dummy_input = torch.randn(1, input_length, device=device)
            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _ = model(dummy_input)

            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

    latencies_np = np.array(latencies)
    audio_duration_sec = input_length / 16000  # Assuming 16kHz

    return {
        "latency_ms_mean": float(np.mean(latencies_np)),
        "latency_ms_std": float(np.std(latencies_np)),
        "latency_ms_p95": float(np.percentile(latencies_np, 95)),
        "latency_ms_min": float(np.min(latencies_np)),
        "latency_ms_max": float(np.max(latencies_np)),
        "throughput_samples_per_sec": float(1000.0 / np.mean(latencies_np)),
        "rtf": float(np.mean(latencies_np) / 1000.0 / audio_duration_sec),
    }


# ============================================================
# Evaluate All Folds (Averaged Checkpoints)
# ============================================================

def evaluate_all_folds(
    model_name: str,
    config: dict[str, Any],
    device: torch.device,
    dataset_name: str = "asvspoof_eval",
) -> dict[str, Any]:
    """Load averaged checkpoints from all folds and evaluate.

    Args:
        model_name: Model name.
        config: Full config dict.
        device: Target device.
        dataset_name: 'asvspoof_eval' or 'in_the_wild'.

    Returns:
        Per-fold and aggregated results.
    """
    experiment_name = config.get("experiment", {}).get("name", "experiment")
    ckpt_cfg = config.get("checkpoint", {})
    eval_cfg = config.get("eval", {})
    vram_cfg = config.get("vram", {})
    data_cfg = config.get("data", {})
    n_splits = config.get("kfold", {}).get("n_splits", 5)
    top_k = ckpt_cfg.get("top_k", 5)

    amp_dtype_str = vram_cfg.get("amp_dtype", "bfloat16")
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16

    # ── Prepare Dataset ──────────────────────────────────────
    if dataset_name == "asvspoof_eval":
        protocol_path = Path(eval_cfg.get(
            "asvspoof_eval_protocol",
            "data/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
        ))
        flac_dir = Path(eval_cfg.get(
            "asvspoof_eval_flac_dir",
            "data/ASVspoof2019_LA/ASVspoof2019_LA_eval/flac",
        ))
        samples = ASVspoof2019Dataset.parse_protocol(protocol_path)
        dataset = ASVspoof2019Dataset(
            samples=samples,
            flac_dir=flac_dir,
            target_len=data_cfg.get("target_samples", 64000),
            augment=False,
            is_eval=True,
        )
        asv_score_file = eval_cfg.get("asv_score_file")
    elif dataset_name == "in_the_wild":
        cache_dir = eval_cfg.get("in_the_wild_cache", "data/in_the_wild")
        dataset = InTheWildDataset(
            data_dir=cache_dir,
            target_len=data_cfg.get("target_samples", 64000),
        )
        asv_score_file = None  # No ASV scores for In-the-Wild
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    eval_batch_size = config.get("train", {}).get("batch_size", 24)
    dataloader = get_dataloader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        persistent_workers=False,
        drop_last=False,
    )

    logger.info(
        "Evaluating %s on %s (%d samples)",
        model_name, dataset_name, len(dataset),
    )

    # ── Evaluate each fold ───────────────────────────────────
    fold_results: list[dict[str, Any]] = []
    all_fold_scores: list[np.ndarray] = []
    all_fold_labels: np.ndarray | None = None

    for fold_idx in range(n_splits):
        ckpt_dir = (
            Path(ckpt_cfg.get("save_dir", "experiments"))
            / experiment_name / model_name / f"fold_{fold_idx}" / "checkpoints"
        )

        # Try averaged checkpoint first
        avg_pattern = f"{model_name}_fold{fold_idx}_averaged_top{top_k}.pth"
        avg_path = ckpt_dir / avg_pattern

        if avg_path.exists():
            logger.info("Loading averaged checkpoint: %s", avg_path.name)
            state_dict = torch.load(str(avg_path), map_location="cpu", weights_only=True)
        else:
            # Fall back to best single checkpoint
            ckpt_files = sorted(ckpt_dir.glob(f"{model_name}_fold{fold_idx}_epoch*.pth"))
            if not ckpt_files:
                logger.warning("No checkpoints found for fold %d, skipping.", fold_idx)
                continue
            # Load and average available checkpoints
            state_dict = AveragedCheckpoint.load_and_average(ckpt_files[:top_k])

        # Load model
        model = get_model(model_name)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        # Evaluate
        eval_result = evaluate_model(model, dataloader, device, amp_dtype, fold_idx=fold_idx, dataset_name=dataset_name)
        metrics = compute_metrics(
            eval_result["scores"],
            eval_result["labels"],
            asv_score_file=asv_score_file,
        )

        fold_results.append({
            "fold": fold_idx,
            **metrics,
        })
        all_fold_scores.append(eval_result["scores"])
        if all_fold_labels is None:
            all_fold_labels = eval_result["labels"]

        logger.info(
            "Fold %d — %s: EER=%.2f%%, Acc=%.2f%%%s",
            fold_idx, dataset_name, metrics["eer"], metrics["accuracy"],
            f", min-tDCF={metrics['min_tdcf']:.4f}" if "min_tdcf" in metrics else "",
        )

        # Clean up
        del model
        torch.cuda.empty_cache()

    if not fold_results:
        logger.error("No fold results computed!")
        return {"error": "No checkpoints found"}

    # ── Score Fusion (average logit scores across folds) ─────
    fused_scores = np.mean(np.stack(all_fold_scores), axis=0)
    # Labels are identical across folds (same eval dataset), so take from
    # the first fold's evaluation result stored in all_fold_labels.
    fused_labels = all_fold_labels
    fused_metrics = compute_metrics(fused_scores, fused_labels, asv_score_file)

    # ── Per-fold EER stats ───────────────────────────────────
    fold_eers = [r["eer"] for r in fold_results]
    mean_eer = float(np.mean(fold_eers))
    std_eer = float(np.std(fold_eers))

    logger.info(
        "%s on %s — Per-fold EERs: %s",
        model_name, dataset_name, [f"{e:.2f}%" for e in fold_eers],
    )
    logger.info("Mean EER: %.2f%% ± %.2f%%", mean_eer, std_eer)
    logger.info(
        "Fused EER: %.2f%%%s",
        fused_metrics["eer"],
        f", min-tDCF={fused_metrics['min_tdcf']:.4f}" if "min_tdcf" in fused_metrics else "",
    )

    return {
        "model_name": model_name,
        "dataset": dataset_name,
        "fold_results": fold_results,
        "fold_eers": fold_eers,
        "mean_eer": mean_eer,
        "std_eer": std_eer,
        "fused_metrics": fused_metrics,
    }


# ============================================================
# Full Evaluation Pipeline
# ============================================================

def run_full_evaluation(
    model_name: str,
    config: dict[str, Any],
    device: torch.device,
    wandb_run: Any | None = None,
) -> dict[str, Any]:
    """Run complete evaluation: ASVspoof eval + In-the-Wild + benchmark.

    This is the top-level evaluation function called from main.py.

    Args:
        model_name: Model name.
        config: Full config dict.
        device: Target device.
        wandb_run: Optional wandb run.

    Returns:
        Complete evaluation results dictionary.
    """
    eval_cfg = config.get("eval", {})
    vram_cfg = config.get("vram", {})
    bench_cfg = eval_cfg.get("inference_benchmark", {})

    amp_dtype_str = vram_cfg.get("amp_dtype", "bfloat16")
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16

    results = {"model_name": model_name}

    # ── 1. ASVspoof 2019 LA Eval ─────────────────────────────
    logger.info("=" * 60)
    logger.info("Evaluating on ASVspoof 2019 LA Eval")
    logger.info("=" * 60)

    asvspoof_results = evaluate_all_folds(
        model_name=model_name,
        config=config,
        device=device,
        dataset_name="asvspoof_eval",
    )
    results["asvspoof_eval"] = asvspoof_results

    # ── 2. In-the-Wild ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Evaluating on In-the-Wild")
    logger.info("=" * 60)

    try:
        itw_results = evaluate_all_folds(
            model_name=model_name,
            config=config,
            device=device,
            dataset_name="in_the_wild",
        )
        results["in_the_wild"] = itw_results
    except Exception as e:
        logger.warning("In-the-Wild evaluation failed: %s", e)
        results["in_the_wild"] = {"error": str(e)}

    # ── 3. Inference Benchmark ───────────────────────────────
    logger.info("=" * 60)
    logger.info("Running inference benchmark")
    logger.info("=" * 60)

    model = get_model(model_name).to(device)
    model.eval()

    # Model stats
    n_params = count_parameters(model)
    try:
        model_stats = get_model_stats(model, input_shape=(1, 64000))
    except Exception:
        model_stats = {"params": n_params, "macs": -1}

    benchmark = run_inference_benchmark(
        model=model,
        device=device,
        amp_dtype=amp_dtype,
        n_samples=bench_cfg.get("n_samples", 100),
        warmup_samples=bench_cfg.get("warmup_samples", 10),
    )

    results["model_stats"] = {
        "trainable_params": n_params,
        **model_stats,
    }
    results["inference_benchmark"] = benchmark

    logger.info(
        "Model stats: %s params, latency=%.2f±%.2f ms, RTF=%.4f",
        f"{n_params:,}", benchmark["latency_ms_mean"],
        benchmark["latency_ms_std"], benchmark["rtf"],
    )

    del model
    torch.cuda.empty_cache()

    # ── 4. Log to wandb ──────────────────────────────────────
    if wandb_run is not None:
        prefix = f"eval/{model_name}"
        if "asvspoof_eval" in results and "fused_metrics" in results["asvspoof_eval"]:
            fm = results["asvspoof_eval"]["fused_metrics"]
            wandb_run.summary[f"{prefix}/asvspoof_fused_eer"] = fm["eer"]
            if "min_tdcf" in fm:
                wandb_run.summary[f"{prefix}/asvspoof_fused_tdcf"] = fm["min_tdcf"]
            wandb_run.summary[f"{prefix}/asvspoof_mean_eer"] = results["asvspoof_eval"]["mean_eer"]

        if "in_the_wild" in results and "fused_metrics" in results.get("in_the_wild", {}):
            wandb_run.summary[f"{prefix}/itw_fused_eer"] = results["in_the_wild"]["fused_metrics"]["eer"]

        wandb_run.summary[f"{prefix}/params"] = n_params
        wandb_run.summary[f"{prefix}/latency_ms"] = benchmark["latency_ms_mean"]
        wandb_run.summary[f"{prefix}/rtf"] = benchmark["rtf"]

    # ── 5. Summary Table & Save to Disk ──────────────────────
    _print_summary_table(results)
    
    # Save to local JSON file
    import json
    experiment_name = config.get("experiment", {}).get("name", "experiment")
    save_dir = Path(config.get("checkpoint", {}).get("save_dir", "experiments")) / experiment_name / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "evaluation_results.json"
    
    # helper to handle numpy types in json
    def default_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4, default=default_numpy)
        
    logger.info("Evaluation results saved locally to: %s", json_path)

    return results


def _print_summary_table(results: dict[str, Any]) -> None:
    """Print a formatted summary table of evaluation results."""
    model_name = results.get("model_name", "unknown")
    stats = results.get("model_stats", {})
    bench = results.get("inference_benchmark", {})

    print("\n" + "=" * 70)
    print(f"  EVALUATION SUMMARY: {model_name}")
    print("=" * 70)

    print(f"  Parameters:     {stats.get('trainable_params', 'N/A'):>12,}")
    if stats.get("macs", -1) > 0:
        print(f"  MACs:           {stats['macs']:>12,}")
    print(f"  Latency (mean): {bench.get('latency_ms_mean', 0):>10.2f} ms")
    print(f"  Latency (p95):  {bench.get('latency_ms_p95', 0):>10.2f} ms")
    print(f"  RTF:            {bench.get('rtf', 0):>10.4f}")
    print(f"  Throughput:     {bench.get('throughput_samples_per_sec', 0):>10.1f} samples/s")

    # ASVspoof results
    asv = results.get("asvspoof_eval", {})
    if "fused_metrics" in asv:
        fm = asv["fused_metrics"]
        print(f"\n  ASVspoof 2019 LA Eval:")
        print(f"    Fused EER:    {fm['eer']:>10.2f}%")
        if "min_tdcf" in fm:
            print(f"    min t-DCF:    {fm['min_tdcf']:>10.4f}")
        print(f"    Mean EER:     {asv['mean_eer']:>10.2f}% ± {asv['std_eer']:.2f}%")

    # In-the-Wild results
    itw = results.get("in_the_wild", {})
    if "fused_metrics" in itw:
        fm = itw["fused_metrics"]
        print(f"\n  In-the-Wild:")
        print(f"    Fused EER:    {fm['eer']:>10.2f}%")
        print(f"    Mean EER:     {itw['mean_eer']:>10.2f}% ± {itw['std_eer']:.2f}%")

    print("=" * 70 + "\n")


# ============================================================
# Smoke Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("src/evaluate.py — Smoke Test (structure only)")
    print("=" * 60)

    # Test compute_metrics
    np.random.seed(42)
    scores = np.concatenate([np.random.randn(500) + 1, np.random.randn(500) - 1])
    labels = np.concatenate([np.ones(500), np.zeros(500)])
    metrics = compute_metrics(scores, labels)
    print(f"✓ compute_metrics: EER={metrics['eer']:.2f}%, Acc={metrics['accuracy']:.2f}%")

    # Test benchmark (CPU only for smoke test)
    import torch.nn as nn

    class DummyModel(nn.Module):
        def forward(self, x):
            return torch.randn(x.size(0), 2)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = DummyModel().to(device)
        bench = run_inference_benchmark(model, device, n_samples=10, warmup_samples=2)
        print(f"✓ Benchmark: latency={bench['latency_ms_mean']:.2f}ms, RTF={bench['rtf']:.4f}")
    else:
        print("⚠ CUDA not available, skipping benchmark test")

    print("\n✅ evaluate.py structure smoke test passed!")
