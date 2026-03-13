"""
scripts/verify_gpu.py
GPU verification: CUDA availability, bf16 support, model forward pass test.

Usage:
    python main.py --mode verify
    python scripts/verify_gpu.py          (standalone)
"""

import sys
import time

import torch


def run_verification(gpu_idx: int = 0) -> None:
    """Run comprehensive GPU and model verification.

    Tests:
        1. CUDA availability and device info
        2. bf16 / AMP support
        3. Forward pass for all models
        4. VRAM usage summary

    Args:
        gpu_idx: GPU device index to test.
    """
    print("=" * 70)
    print("  GPU & MODEL VERIFICATION")
    print("=" * 70)

    # ── 1. CUDA Check ────────────────────────────────────────
    print("\n[1/4] CUDA Availability")
    print("-" * 40)

    if not torch.cuda.is_available():
        print("  ❌ CUDA is NOT available!")
        print("  Install PyTorch with CUDA support:")
        print("  uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu128")
        sys.exit(1)

    device = torch.device(f"cuda:{gpu_idx}")
    props = torch.cuda.get_device_properties(gpu_idx)

    print(f"  ✅ CUDA available")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA version:    {torch.version.cuda}")
    print(f"  cuDNN version:   {torch.backends.cudnn.version()}")
    print(f"  Device [{gpu_idx}]:    {props.name}")
    print(f"  Compute cap:     {props.major}.{props.minor}")
    print(f"  Total VRAM:      {props.total_memory / 1024**3:.2f} GB")
    print(f"  Multi-processor: {props.multi_processor_count}")

    # ── 2. bf16 / AMP Support ────────────────────────────────
    print("\n[2/4] bf16 / AMP Support")
    print("-" * 40)

    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"  bf16 hardware support: {'✅ Yes' if bf16_supported else '❌ No'}")

    # Test AMP context
    try:
        x = torch.randn(2, 64000, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = x * 2
        print(f"  bf16 autocast:         ✅ Working")
        print(f"  Output dtype:          {y.dtype}")
    except Exception as e:
        print(f"  bf16 autocast:         ❌ Failed: {e}")
        if not bf16_supported:
            print("  Falling back to fp16 is recommended.")

    # Test GradScaler
    try:
        scaler = torch.cuda.amp.GradScaler()
        print(f"  GradScaler:            ✅ Available")
    except Exception as e:
        print(f"  GradScaler:            ❌ Failed: {e}")

    # Flash Attention check
    try:
        sdpa_available = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        print(f"  SDPA (Flash Attn):     {'✅ Available' if sdpa_available else '❌ Not available'}")
    except Exception:
        pass

    # ── 3. Model Forward Pass Tests ──────────────────────────
    print("\n[3/4] Model Forward Pass Tests")
    print("-" * 40)

    from src.models import get_model, list_models
    from src.utils import count_parameters

    all_models = list_models()
    print(f"  Registered models: {all_models}")

    results = []
    x = torch.randn(2, 64000, device=device)

    for model_name in all_models:
        print(f"\n  Testing '{model_name}'...")
        try:
            model = get_model(model_name).to(device)
            model.eval()
            n_params = count_parameters(model)

            # Forward pass with bf16
            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)

            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert out.shape == (2, 2), f"Expected (2,2), got {out.shape}"
            assert not torch.isnan(out).any(), "Output contains NaN!"

            vram_mb = torch.cuda.memory_allocated(gpu_idx) / 1024**2

            print(f"    ✅ Passed")
            print(f"    Parameters:  {n_params:>10,}")
            print(f"    Output:      {out.shape}")
            print(f"    Latency:     {elapsed_ms:>8.2f} ms")
            print(f"    VRAM used:   {vram_mb:>8.1f} MB")

            results.append({
                "model": model_name,
                "status": "✅",
                "params": n_params,
                "latency_ms": elapsed_ms,
                "vram_mb": vram_mb,
            })

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    ❌ Failed: {e}")
            results.append({
                "model": model_name,
                "status": "❌",
                "error": str(e),
            })
            torch.cuda.empty_cache()

    # ── 4. Summary ───────────────────────────────────────────
    print("\n[4/4] Summary")
    print("-" * 40)

    total_vram_gb = props.total_memory / 1024**3
    print(f"\n  {'Model':<20} {'Status':<6} {'Params':>12} {'Latency':>10} {'VRAM':>10}")
    print(f"  {'─'*20} {'─'*6} {'─'*12} {'─'*10} {'─'*10}")

    all_passed = True
    for r in results:
        if r["status"] == "✅":
            print(
                f"  {r['model']:<20} {r['status']:<6} "
                f"{r['params']:>12,} "
                f"{r['latency_ms']:>8.2f}ms "
                f"{r['vram_mb']:>8.1f}MB"
            )
        else:
            print(f"  {r['model']:<20} {r['status']:<6} {r.get('error', 'Unknown error')}")
            all_passed = False

    print(f"\n  Total GPU VRAM:  {total_vram_gb:.2f} GB")
    print(f"  bf16 supported:  {'Yes' if bf16_supported else 'No'}")

    if all_passed:
        print("\n  ✅ All verification checks passed!")
    else:
        print("\n  ⚠️  Some checks failed. Review errors above.")

    print("=" * 70)


if __name__ == "__main__":
    run_verification()
