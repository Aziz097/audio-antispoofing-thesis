# Audio Anti-Spoofing Comparative Study

Undergraduate thesis project comparing lightweight audio anti-spoofing models on the ASVspoof 2019 LA benchmark and the In-the-Wild evaluation dataset.

## Models

| Model | Params | Description |
|-------|--------|-------------|
| **AASIST** | ~297K | Graph Attention Network with heterogeneous spectral-temporal processing |
| **SE-Rawformer** | ~370K | CNN-Transformer hybrid with SE-Res2Net frontend and 1D positional encoding |
| **RawTFNet τ=32** | ~410K | Depthwise-separable SE-Res2Net frontend + Time-Frequency Separable convolution classifier |
| **RawTFNet τ=16** | ~80K | Compact variant with halved channels and deeper classifier |

All models operate on **raw 16 kHz waveforms** (4 seconds, 64,000 samples) — no handcrafted features.

## Experimental Schemas

- **Schema B** (primary): Training **with** RawBoost Algorithm 4 (LnL → ISD → SSI) augmentation
- **Schema A** (ablation): Training **without** augmentation — isolates augmentation contribution

Both schemas use **5-fold stratified cross-validation** (train+dev merged), **early stopping** (patience=10), and **top-5 checkpoint averaging**.

## Hardware Requirements

- **GPU**: NVIDIA RTX 5070 (12 GB VRAM) or equivalent
- **AMP**: bf16 mixed precision (Blackwell/Ampere+ architecture)
- **Python**: 3.12+
- **CUDA**: 12.8

## Quick Start

### 1. Environment Setup (using uv)

```bash
# Create and activate virtual environment
uv venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"
```

### 2. Download Data

```bash
# Download ASVspoof 2019 LA from Kaggle (requires kaggle.json credentials)
python main.py --mode download --dataset asvspoof

# Download In-the-Wild from HuggingFace
python main.py --mode download --dataset in_the_wild

# Download both
python main.py --mode download --dataset all
```

### 3. Verify GPU Setup

```bash
python main.py --mode verify
```

### 4. Train Models

```bash
# Schema B (with RawBoost augmentation) — train each model
python main.py --mode train --model aasist --config configs/schema_b.yaml
python main.py --mode train --model se_rawformer --config configs/schema_b.yaml
python main.py --mode train --model rawtfnet_32 --config configs/schema_b.yaml
python main.py --mode train --model rawtfnet_16 --config configs/schema_b.yaml

# Schema A (without augmentation) — ablation study
python main.py --mode train --model aasist --config configs/schema_a.yaml
python main.py --mode train --model se_rawformer --config configs/schema_a.yaml
python main.py --mode train --model rawtfnet_32 --config configs/schema_a.yaml
python main.py --mode train --model rawtfnet_16 --config configs/schema_a.yaml
```

### 5. Evaluate Models

```bash
python main.py --mode eval --model aasist --config configs/schema_b.yaml
python main.py --mode eval --model se_rawformer --config configs/schema_b.yaml
python main.py --mode eval --model rawtfnet_32 --config configs/schema_b.yaml
python main.py --mode eval --model rawtfnet_16 --config configs/schema_b.yaml
```

## Project Structure

```
audio-antispoofing-thesis/
├── main.py                          # CLI entry point
├── pyproject.toml                   # Project config & dependencies
├── configs/
│   ├── schema_a.yaml                # Schema A (no augmentation)
│   └── schema_b.yaml                # Schema B (with RawBoost)
├── src/
│   ├── __init__.py
│   ├── utils.py                     # Seeds, VRAM, RawBoost, EER, t-DCF, etc.
│   ├── dataloader.py                # Datasets, K-Fold manager, DataLoader factory
│   ├── train.py                     # Training loop, checkpoint management
│   ├── evaluate.py                  # Evaluation pipeline, inference benchmark
│   └── models/
│       ├── __init__.py              # Model registry
│       ├── aasist.py                # AASIST implementation
│       ├── se_rawformer.py          # SE-Rawformer implementation
│       └── rawtfnet.py              # RawTFNet (τ=16 and τ=32)
├── scripts/
│   ├── verify_gpu.py                # GPU/bf16/model verification
│   └── download_data.py             # Dataset download utilities
├── data/                            # (gitignored) Downloaded datasets
│   ├── ASVspoof2019_LA/
│   └── in_the_wild/
└── experiments/                     # (gitignored) Training outputs
    └── <schema>/<model>/fold_*/checkpoints/
```

## Training Details

| Setting | Value |
|---------|-------|
| Input | Raw waveform, 4s @ 16 kHz (64,000 samples) |
| Cross-validation | 5-fold stratified (train+dev merged) |
| Batch size | 24 (effective 48 with grad accumulation) |
| Optimizer | Adam (lr=1e-4, weight_decay=1e-4) |
| Loss | Weighted Cross-Entropy |
| AMP | bf16 with GradScaler |
| Max epochs | 100 |
| Early stopping | Patience=10, monitor val EER |
| Checkpointing | Top-5 by val EER, averaged for eval |

## Evaluation Metrics

- **EER** (Equal Error Rate) — primary metric
- **min t-DCF** (minimum tandem Detection Cost Function) — ASVspoof 2019 standard
- **Inference latency** (ms), **RTF** (Real-Time Factor), **throughput** (samples/s)
- **Parameter count** and **MACs**

## Experiment Tracking

Training metrics are logged to **Weights & Biases** (wandb).

```bash
# Set your API key
export WANDB_API_KEY="your_key_here"

# Or disable wandb
python main.py --mode train --model aasist --config configs/schema_b.yaml --no-wandb
```

## References

- **AASIST**: Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks", ICASSP 2022
- **Rawformer/SE-Rawformer**: "Leveraging Positional-Related Local-Global Dependency for Synthetic Speech Detection"
- **RawTFNet**: Xiao et al., "RawTFNet: Utilising Raw Time-Frequency Representations for Spoofed Speech Detection"
- **RawBoost**: Tak et al., "RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing", ICASSP 2022
- **ASVspoof 2019**: Wang et al., "ASVspoof 2019: A Large-Scale Public Database of Synthesized, Converted and Replayed Speech", Computer Speech & Language, 2020

## License

This project is for academic research purposes (undergraduate thesis).
