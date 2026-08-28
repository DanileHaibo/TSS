"""Generic skip-layer wrapper for Qwen/Llama-style decoder stacks."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class SkipLayerTargetModel(nn.Module):
    def __init__(self, base_model: nn.Module, skip_layers: Iterable[int] | None = None):
        super().__init__()
        self.base = base_model
        self.skip_layers = frozenset(int(i) for i in (skip_layers or []))

    @property
    def config(self):
        return self.base.config

    def _layer_mask(self, attention_mask, layer):
        if not isinstance(attention_mask, dict):
            return attention_mask
        key = getattr(layer, "attention_type", "full_attention")
        return attention_mask.get(key, attention_mask.get("full_attention"))

    def forward(self, input_ids: torch.Tensor | None = None, **kwargs):
        if input_ids is None and kwargs.get("inputs_embeds") is None:
            return self.base(input_ids=input_ids, **kwargs)

        model = self.base.model
        use_cache = kwargs.get("use_cache", False)
        attention_mask = kwargs.get("attention_mask")
        position_ids = kwargs.get("position_ids")
        past_key_values = kwargs.get("past_key_values")
        cache_position = kwargs.get("cache_position")

        if kwargs.get("inputs_embeds") is not None:
            inputs_embeds = kwargs["inputs_embeds"]
        else:
            inputs_embeds = model.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            from transformers.cache_utils import DynamicCache

            past_key_values = DynamicCache(config=model.config)

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if attention_mask is not None and not isinstance(attention_mask, dict):
            from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

            mask_kwargs = {
                "config": model.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            attention_mask = {"full_attention": create_causal_mask(**mask_kwargs)}
            if getattr(model, "has_sliding_layers", False):
                attention_mask["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = model.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(model.layers):
            if layer_idx in self.skip_layers:
                continue
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=self._layer_mask(attention_mask, decoder_layer),
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = model.norm(hidden_states)
        logits = self.base.lm_head(hidden_states)
        return type("LMOutput", (), {"logits": logits, "past_key_values": past_key_values})()
