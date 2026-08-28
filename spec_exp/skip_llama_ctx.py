"""Context manager to skip Llama decoder layers (transformers 4.37.x and 4.57+)."""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Any, Callable

import transformers


def _layer_returns_tensor() -> bool:
    ver = transformers.__version__.split(".")[:2]
    try:
        return int(ver[0]) > 4 or (int(ver[0]) == 4 and int(ver[1]) >= 57)
    except ValueError:
        return False


_LAYER_RETURNS_TENSOR = _layer_returns_tensor()


def _resolve_layers(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    return model.layers


def _make_skip_forward(original_forward: Callable, layer_idx: int, skip_set: set[int]) -> Callable:
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        if layer_idx not in skip_set:
            return original_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        pkv = past_key_values if past_key_values is not None else past_key_value
        if use_cache:
            normed = self.input_layernorm(hidden_states)
            attn_kwargs = {
                "hidden_states": normed,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "output_attentions": output_attentions,
                "use_cache": True,
            }
            if _LAYER_RETURNS_TENSOR:
                attn_kwargs["past_key_values"] = pkv
                if cache_position is not None:
                    attn_kwargs["cache_position"] = cache_position
                if position_embeddings is not None:
                    attn_kwargs["position_embeddings"] = position_embeddings
            else:
                attn_kwargs["past_key_value"] = pkv
            attn_out = self.self_attn(**attn_kwargs)
            if isinstance(attn_out, tuple):
                if len(attn_out) == 3:
                    _, attn_weights, present_key_value = attn_out
                else:
                    _, present_key_value = attn_out
                    attn_weights = None
            else:
                attn_weights = None
                present_key_value = pkv

            if _LAYER_RETURNS_TENSOR:
                return hidden_states

            outputs: tuple[Any, ...] = (hidden_states,)
            if output_attentions:
                outputs += (attn_weights,)
            outputs += (present_key_value,)
            return outputs

        if _LAYER_RETURNS_TENSOR:
            return hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)
        return outputs

    return forward


@contextmanager
def skip_llama_ctx(model: Any, skip_layers: set[int]):
    if not skip_layers:
        yield
        return

    layers = _resolve_layers(model)
    saved: list[tuple[int, Callable]] = []
    for idx, layer in enumerate(layers):
        if idx not in skip_layers:
            continue
        orig = layer.forward
        layer.forward = MethodType(_make_skip_forward(orig, idx, skip_layers), layer)
        saved.append((idx, orig))
    try:
        yield
    finally:
        for idx, orig in saved:
            layers[idx].forward = orig
