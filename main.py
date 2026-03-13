"""
main.py
CLI entry point for audio anti-spoofing comparative study.

Modes:
    train    — Run K-Fold training for a model under a schema
    eval     — Evaluate trained model on ASVspoof eval + In-the-Wild
    download — Download datasets (ASVspoof from Kaggle, In-the-Wild from HF)
    verify   — Verify GPU, bf16 support, and model forward pass

Usage:
    python main.py --mode train --model aasist --config configs/schema_b.yaml
    python main.py --mode eval  --model se_rawformer --config configs/schema_b.yaml
    python main.py --mode download --dataset asvspoof
    python main.py --mode verify
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

from src.utils import load_config, set_seed


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logger with consistent formatting."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Audio Anti-Spoofing Comparative Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models:
  aasist          AASIST (Graph Attention)
  se_rawformer    SE-Rawformer (CNN-Transformer Hybrid)
  rawtfnet_32     RawTFNet tau=32
  rawtfnet_16     RawTFNet tau=16

Examples:
  python main.py --mode train --model aasist --config configs/schema_b.yaml
  python main.py --mode eval  --model se_rawformer --config configs/schema_a.yaml
  python main.py --mode download --dataset all
  python main.py --mode verify
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "eval", "download", "verify"],
        help="Operation mode.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["aasist", "se_rawformer", "rawtfnet_32", "rawtfnet_16"],
        help="Model to train or evaluate.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/schema_b.yaml",
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["asvspoof", "in_the_wild", "all"],
        help="Dataset to download (for --mode download).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device index.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pth file to resume training from.",
    )

    return parser.parse_args()


# ============================================================
# Mode Handlers
# ============================================================

def mode_train(args: argparse.Namespace) -> None:
    """Run K-Fold training for a specified model."""
    logger = logging.getLogger(__name__)

    if args.model is None:
        logger.error("--model is required for train mode.")
        sys.exit(1)

    config = load_config(args.config)
    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(args.gpu))
        logger.info("VRAM: %.2f GB", torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3)

    # ── wandb ────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb

            wandb_cfg = config.get("wandb", {})
            run_name = f"{config.get('experiment', {}).get('name', 'exp')}_{args.model}"
            wandb_run = wandb.init(
                project=wandb_cfg.get("project", "audio-antispoofing-thesis"),
                name=run_name,
                config={
                    "model": args.model,
                    "schema": config.get("experiment", {}).get("name"),
                    **config,
                },
                tags=wandb_cfg.get("tags", []),
                mode=wandb_cfg.get("mode", "online"),
            )
            logger.info("wandb initialized: %s", run_name)
        except Exception as e:
            logger.warning("wandb init failed: %s. Continuing without wandb.", e)

    # ── Train ────────────────────────────────────────────────
    from src.train import run_kfold_experiment

    results = run_kfold_experiment(
        model_name=args.model,
        config=config,
        device=device,
        wandb_run=wandb_run,
        resume=args.resume,
    )

    logger.info(
        "Training complete: %s — Mean EER: %.2f%% ± %.2f%%",
        args.model, results["mean_eer"], results["std_eer"],
    )

    if wandb_run is not None:
        wandb_run.finish()


def mode_eval(args: argparse.Namespace) -> None:
    """Evaluate trained model on eval datasets."""
    logger = logging.getLogger(__name__)

    if args.model is None:
        logger.error("--model is required for eval mode.")
        sys.exit(1)

    config = load_config(args.config)
    set_seed(config.get("experiment", {}).get("seed", 42))

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── wandb ────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb

            wandb_cfg = config.get("wandb", {})
            run_name = f"eval_{config.get('experiment', {}).get('name', 'exp')}_{args.model}"
            wandb_run = wandb.init(
                project=wandb_cfg.get("project", "audio-antispoofing-thesis"),
                name=run_name,
                config={"model": args.model, "mode": "eval", **config},
                tags=[*wandb_cfg.get("tags", []), "eval"],
                mode=wandb_cfg.get("mode", "online"),
            )
        except Exception as e:
            logger.warning("wandb init failed: %s", e)

    # ── Evaluate ─────────────────────────────────────────────
    from src.evaluate import run_full_evaluation

    results = run_full_evaluation(
        model_name=args.model,
        config=config,
        device=device,
        wandb_run=wandb_run,
    )

    if wandb_run is not None:
        wandb_run.finish()


def mode_download(args: argparse.Namespace) -> None:
    """Download datasets."""
    logger = logging.getLogger(__name__)

    if args.dataset in ("asvspoof", "all"):
        logger.info("Downloading ASVspoof 2019 LA from Kaggle...")
        try:
            from scripts.download_data import download_asvspoof
            download_asvspoof()
        except Exception as e:
            logger.error("ASVspoof download failed: %s", e)

    if args.dataset in ("in_the_wild", "all"):
        logger.info("Downloading In-the-Wild from HuggingFace...")
        try:
            from scripts.download_data import download_in_the_wild
            download_in_the_wild()
        except Exception as e:
            logger.error("In-the-Wild download failed: %s", e)


def mode_verify(args: argparse.Namespace) -> None:
    """Run GPU verification and model smoke tests."""
    logger = logging.getLogger(__name__)

    try:
        from scripts.verify_gpu import run_verification
        run_verification(gpu_idx=args.gpu)
    except Exception as e:
        logger.error("Verification failed: %s", e)
        sys.exit(1)


# ============================================================
# Entry Point
# ============================================================

def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Audio Anti-Spoofing Comparative Study")
    logger.info("Mode: %s", args.mode)

    mode_dispatch = {
        "train": mode_train,
        "eval": mode_eval,
        "download": mode_download,
        "verify": mode_verify,
    }

    mode_dispatch[args.mode](args)


if __name__ == "__main__":
    main()
