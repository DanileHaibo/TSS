"""Patch Spec-Bench helpers for newer transformers (4.57+) on Blackwell GPUs."""
from __future__ import annotations

from typing import Any


def _crop_past_key_values(model: Any, past_key_values: Any, new_cache_size: int) -> Any:
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(new_cache_size)
        return past_key_values
    cropped = []
    for layer_past in past_key_values:
        cropped.append(tuple(past[:, :, :new_cache_size, :] for past in layer_past))
    return tuple(cropped)


def install_transformers_compat() -> None:
    import transformers.generation.candidate_generator as cg
    import transformers.generation.utils as gu

    if not hasattr(cg, "_crop_past_key_values"):
        cg._crop_past_key_values = _crop_past_key_values
    if not hasattr(gu, "_crop_past_key_values"):
        gu._crop_past_key_values = _crop_past_key_values
