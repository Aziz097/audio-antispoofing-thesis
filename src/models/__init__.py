"""
src/models/__init__.py
Model registry for audio anti-spoofing models.

Provides a unified interface to instantiate any model by name.
"""

from typing import Any

import torch.nn as nn

# ============================================================
# Model Registry
# ============================================================

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "aasist": {
        "module": "src.models.aasist",
        "class": "AASIST",
        "kwargs": {},
        "description": "AASIST — Graph attention baseline (~0.30M params)",
    },
    "se_rawformer": {
        "module": "src.models.se_rawformer",
        "class": "SERAWFormer",
        "kwargs": {},
        "description": "SE-Rawformer with 1D-PE (~0.37M params)",
    },
    "rawtfnet_16": {
        "module": "src.models.rawtfnet",
        "class": "RawTFNet",
        "kwargs": {"tau": 16},
        "description": "RawTFNet tau=16 (~0.07M params)",
    },
    "rawtfnet_32": {
        "module": "src.models.rawtfnet",
        "class": "RawTFNet",
        "kwargs": {"tau": 32},
        "description": "RawTFNet tau=32 (~0.17M params)",
    },
}


def get_model(name: str, **override_kwargs: Any) -> nn.Module:
    """Instantiate a model by registry name.

    Args:
        name: Model name (one of MODEL_REGISTRY keys).
        **override_kwargs: Additional keyword arguments to pass
                           to the model constructor.

    Returns:
        Instantiated nn.Module.

    Raises:
        ValueError: If model name is not in the registry.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{name}'. Available: {available}"
        )

    entry = MODEL_REGISTRY[name]

    # Dynamic import
    import importlib
    module = importlib.import_module(entry["module"])
    model_class = getattr(module, entry["class"])

    # Merge default kwargs with overrides
    kwargs = {**entry["kwargs"], **override_kwargs}

    return model_class(**kwargs)


def list_models() -> list[str]:
    """List all available model names."""
    return list(MODEL_REGISTRY.keys())


# ============================================================
# Smoke Test
# ============================================================

if __name__ == "__main__":
    import torch
    from src.utils import count_parameters

    print("=" * 60)
    print("Model Registry — Smoke Test")
    print("=" * 60)

    device = "cpu"
    dummy_input = torch.randn(2, 64000, device=device)

    for name in list_models():
        print(f"\n--- {name} ---")
        model = get_model(name).to(device)
        model.eval()

        with torch.no_grad():
            output = model(dummy_input)

        n_params = count_parameters(model)
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters:   {n_params:,}")
        print(f"  Description:  {MODEL_REGISTRY[name]['description']}")

    print("\n✅ All models instantiated successfully!")
