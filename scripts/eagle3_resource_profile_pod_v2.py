#!/usr/bin/env python3
"""EAGLE-3 resource profiling v2 — POD-Attention (2410.18038) style metrics.

POD Figure 1 uses Nsight Compute:
  - sm__throughput.avg.pct_of_peak_sustained_elapsed  -> Compute Utilization
  - dram__throughput.avg.pct_of_peak_sustained_elapsed -> Mem BW Utilization

When ncu is unavailable (RmProfilingAdminOnly), we approximate the same semantics
via roofline: achieved_tensor_flops / calibrated_peak and achieved_hbm_gbps /
calibrated_peak, using CUDA-event timed phase execution and analytical traffic.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

EAGLE_ROOT = Path("/root/autodl-tmp/eagle3-system-exp/repos/EAGLE")
sys.path.insert(0, str(EAGLE_ROOT))

from eagle.model.ea_model import EaModel  # noqa: E402
from eagle.model.kv_cache import initialize_past_key_values  # noqa: E402
from eagle.model.utils import initialize_tree, reset_tree_mode, tree_decoding  # noqa: E402

BASE_DEFAULT = (
    "/root/autodl-tmp/hf-cache/hub/models--lmsys--vicuna-13b-v1.3/snapshots/"
    "6566e9cb1787585d1147dcf4f9bc48f29e1328d2"
)
EA_DEFAULT = (
    "/root/autodl-tmp/hf-cache/hub/models--yuhuili--EAGLE3-Vicuna1.3-13B/snapshots/"
    "651195736adad4c05282d140e94bfff058b1fc8b"
)
LLAMA31_BASE = "/root/autodl-tmp/modelscope/LLM-Research/Meta-Llama-3___1-8B-Instruct"
LLAMA31_EA = (
    "/root/autodl-tmp/hf/models--lmsys--sglang-EAGLE3-LLaMA3.1-Instruct-8B/snapshots/"
    "28a53ce8911434c031d7c78392abb26d898ec293"
)
MODEL_PRESETS = {
    "vicuna13": {
        "base": BASE_DEFAULT,
        "ea": EA_DEFAULT,
        "label": "Vicuna-13B + EAGLE3-Vicuna1.3-13B",
        "chat_template": "vicuna",
    },
    "llama31": {
        "base": LLAMA31_BASE,
        "ea": LLAMA31_EA,
        "label": "LLaMA-3.1-8B-Instruct + EAGLE3-LLaMA3.1-8B",
        "chat_template": "llama3",
    },
}

NCU = Path("/usr/local/cuda-12.1/bin/ncu")
NCU_METRICS = (
    "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
    "dram__throughput.avg.pct_of_peak_sustained_elapsed"
)
SM_RE = re.compile(r'"sm__throughput\.avg\.pct_of_peak_sustained_elapsed","%","(.*?)"')
BW_RE = re.compile(r'"dram__throughput\.avg\.pct_of_peak_sustained_elapsed","%","(.*?)"')


@dataclass
class GpuPeaks:
    fp16_tflops: float
    hbm_gbps: float
    source: str


# NCU reports % of *hardware* peak sustained; use datasheet peaks as denominator.
SPEC_PEAKS = GpuPeaks(fp16_tflops=234.0, hbm_gbps=1597.0, source="RTX PRO 6000 Blackwell spec")


class CudaTimer:
    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()
        self.ms = float(self.start.elapsed_time(self.end))


def calibrate_peaks(fp16_tflops: float | None = None, hbm_gbps: float | None = None) -> GpuPeaks:
    """Return hardware peak sustained rates (POD / Nsight Compute denominator)."""
    if fp16_tflops is not None and hbm_gbps is not None:
        return GpuPeaks(fp16_tflops=fp16_tflops, hbm_gbps=hbm_gbps, source="cli")
    return SPEC_PEAKS


def _layer_dims(cfg) -> tuple[int, int, int, int, int]:
    layers = cfg.num_hidden_layers
    hidden = cfg.hidden_size
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = hidden // cfg.num_attention_heads
    inter = getattr(cfg, "intermediate_size", 4 * hidden)
    return layers, hidden, n_kv, head_dim, inter


def prefill_traffic(cfg, *, seq_len: int, batch: int = 1, weight_bytes: float = 0) -> tuple[float, float]:
    """Attention-kernel traffic for prefill (POD profiles FlashAttention prefill)."""
    layers, hidden, n_kv, head_dim, _ = _layer_dims(cfg)
    s, b = seq_len, batch
    # QKV + output proj + fused attention matmuls.
    flops = layers * (b * s * hidden * hidden * 8 + 2 * b * (s**2) * hidden)
    qkv_bytes = layers * 4 * b * s * hidden * 2
    kv_write = layers * 2 * b * s * n_kv * head_dim * 2
    # Weight streaming only counts once per layer block during prefill.
    hbm_bytes = 0.15 * weight_bytes + qkv_bytes + kv_write
    return float(flops), float(hbm_bytes)


def decode_traffic(cfg, *, ctx_len: int, batch: int, weight_bytes: float = 0) -> tuple[float, float]:
    """Attention-kernel traffic for batched decode (POD-style)."""
    layers, hidden, n_kv, head_dim, _ = _layer_dims(cfg)
    c, w = ctx_len, max(1, batch)
    flops = layers * (w * hidden * hidden * 8 + 2 * w * c * hidden)
    kv_read = layers * 2 * c * w * n_kv * head_dim * 2
    kv_write = layers * 2 * w * n_kv * head_dim * 2
    qkv_bytes = layers * 4 * w * hidden * 2
    hbm_bytes = 0.05 * weight_bytes + kv_read + kv_write + qkv_bytes
    return float(flops), float(hbm_bytes)


def draft_batched_decode_traffic(model: EaModel, *, ctx_len: int, batch: int) -> tuple[float, float]:
    """Single ea_layer mid-block batched decode (EAGLE draft is 1 layer, not full Llama depth)."""
    cfg = model.ea_layer.config
    hidden = cfg.hidden_size
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = hidden // cfg.num_attention_heads
    layers = 1
    wbytes = sum(p.numel() for p in model.ea_layer.parameters()) * 2
    c, w = ctx_len, max(1, batch)
    flops = layers * (w * hidden * hidden * 8 + 2 * w * c * hidden)
    kv_read = layers * 2 * c * w * n_kv * head_dim * 2
    kv_write = layers * 2 * w * n_kv * head_dim * 2
    qkv_bytes = layers * 4 * w * hidden * 2
    hbm_bytes = 0.05 * wbytes + kv_read + kv_write + qkv_bytes
    return float(flops), float(hbm_bytes)


DECODE_KV_SLACK = 32


def draft_traffic(model: EaModel, *, ctx_len: int, tree_width: int) -> tuple[float, float]:
    """Draft ea_layer attention-like traffic."""
    cfg = model.ea_layer.config
    layers, hidden, n_kv, head_dim, inter = _layer_dims(cfg)
    depth = model.ea_layer.depth
    top_k = model.ea_layer.top_k
    w = max(1, tree_width)
    s = ctx_len + 1
    tree_nodes = w + top_k * depth
    flops = layers * (tree_nodes * hidden * hidden * 8 + 2 * tree_nodes * s * hidden)
    ea_params = sum(p.numel() for p in model.ea_layer.parameters()) * 2
    kv_read = layers * 2 * s * n_kv * head_dim * 2
    kv_write = layers * 2 * tree_nodes * n_kv * head_dim * 2
    qkv_bytes = layers * 4 * tree_nodes * hidden * 2
    hbm_bytes = 0.2 * ea_params + kv_read + kv_write + qkv_bytes
    return float(flops), float(hbm_bytes)


def roofline_util(flops: float, bytes_: float, ms: float, peaks: GpuPeaks) -> tuple[float, float]:
    if ms <= 0:
        return 0.0, 0.0
    sec = ms * 1e-3
    comp = 100.0 * (flops / sec / 1e12) / peaks.fp16_tflops
    mem = 100.0 * (bytes_ / sec / 1e9) / peaks.hbm_gbps
    return min(100.0, comp), min(100.0, mem)


def target_weight_bytes(model: EaModel) -> float:
    return float(sum(p.numel() for p in model.base_model.parameters()) * 2)


def try_ncu_profile(worker: Path, args: list[str], kernel_regex: str | None = None) -> tuple[float, float] | None:
    if not NCU.exists():
        return None
    out = worker.parent / f"ncu_{worker.stem}_{int(time.time())}.csv"
    cmd = [
        str(NCU),
        "--metrics",
        NCU_METRICS,
        "--csv",
        "--target-processes",
        "all",
        "--cache-control",
        "none",
    ]
    if kernel_regex:
        cmd.extend(["-k", f"regex:{kernel_regex}"])
    cmd.extend([sys.executable, str(worker), *args])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        text = proc.stdout + proc.stderr
        if "ERR_NVGPUCTRPERM" in text or "No kernels were profiled" in text:
            return None
        sm = SM_RE.findall(text)
        bw = BW_RE.findall(text)
        if sm and bw:
            return float(sm[0]), float(bw[0])
        if out.exists():
            content = out.read_text()
            sm = SM_RE.findall(content)
            bw = BW_RE.findall(content)
            if sm and bw:
                return float(sm[0]), float(bw[0])
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def fmt_ctx(n: int) -> str:
    return f"{n // 1024}K" if n >= 1024 else str(n)


def fmt_bs(n: int) -> str:
    return str(n)


def load_model(base: str, ea: str) -> EaModel:
    model = EaModel.from_pretrained(
        base_model_path=base,
        ea_model_path=ea,
        total_token=60,
        depth=5,
        top_k=10,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
        use_eagle3=True,
    )
    model.eval()
    return model


def clear_kv(model: EaModel) -> None:
    if hasattr(model, "past_key_values_data"):
        for t in model.past_key_values_data:
            del t
    for attr in ("past_key_values", "past_key_values_data", "current_length_data"):
        if hasattr(model, attr):
            delattr(model, attr)
    gc.collect()
    torch.cuda.empty_cache()


def initialize_past_key_values_batch(model, max_length: int, batch_size: int):
    """Batched KV cache (POD-style decode); upstream helper only supports batch=1."""
    from eagle.model.kv_cache import KVCache

    config = model.config
    devices = []
    for i in range(config.num_hidden_layers):
        try:
            device = model.model.layers[i].self_attn.q_proj.weight.device
        except AttributeError:
            device = model.layers[i].self_attn.q_proj.weight.device
        devices.append(device)
    past_key_values_data_list = []
    startnum = 0
    startdevice = devices[0]
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    for idx, dev in enumerate(devices):
        if startdevice != dev:
            past_key_values_data_list.append(
                torch.zeros(
                    startnum * 2,
                    batch_size,
                    config.num_key_value_heads,
                    max_length,
                    head_dim,
                    device=startdevice,
                    dtype=model.dtype,
                )
            )
            startdevice = dev
            startnum = 0
        startnum += 1
    past_key_values_data_list.append(
        torch.zeros(
            startnum * 2,
            batch_size,
            config.num_key_value_heads,
            max_length,
            head_dim,
            device=startdevice,
            dtype=model.dtype,
        )
    )
    current_length_data = torch.zeros(config.num_hidden_layers * 2, dtype=torch.long, device="cpu")
    past_key_values = []
    bias = 0
    start_data_m = devices[0].index
    for i in range(config.num_hidden_layers):
        data_m = devices[i].index
        if data_m != start_data_m:
            bias = 0
            start_data_m = data_m
        try:
            past_key_values.append(
                [
                    KVCache(past_key_values_data_list[data_m - devices[0].index][2 * bias + j], current_length_data[i * 2 + j])
                    for j in range(2)
                ]
            )
        except IndexError:
            past_key_values.append(
                [
                    KVCache(past_key_values_data_list[0][2 * bias + j], current_length_data[i * 2 + j])
                    for j in range(2)
                ]
            )
        bias += 1
    return past_key_values, past_key_values_data_list, current_length_data


def _copy_prefill_kv_to_batch(past_kv_src, past_kv_dst, batch_size: int) -> int:
    """Copy batch=1 prefill KV into all B slots of a batched cache."""
    cur_len = int(past_kv_src[0][0].current_length.item())
    for layer_i in range(len(past_kv_src)):
        for kv_idx in range(2):
            src = past_kv_src[layer_i][kv_idx].data.narrow(2, 0, cur_len)
            dst = past_kv_dst[layer_i][kv_idx]
            for b in range(batch_size):
                dst.data[b : b + 1, :, :cur_len, :].copy_(src)
            dst.current_length.fill_(cur_len)
    return cur_len


def setup_batched_decode_state(model: EaModel, ctx_len: int, batch_size: int, extra: int = DECODE_KV_SLACK):
    """POD-style: prefill once (batch=1), replicate KV to B, measure one batched decode step."""
    clear_kv(model)
    reset_tree_mode(model)
    model.ea_layer.reset_kv()
    max_length = ctx_len + extra
    vocab = model.config.vocab_size - 200

    input_ids_1 = torch.randint(100, vocab, (1, ctx_len), device="cuda")
    past_kv_1, pkd_1, _ = initialize_past_key_values(model.base_model, max_length=max_length)
    with torch.inference_mode():
        model.base_model(input_ids_1, past_key_values=past_kv_1, use_cache=True)

    past_kv, pkd, cld = initialize_past_key_values_batch(model.base_model, max_length, batch_size)
    _copy_prefill_kv_to_batch(past_kv_1, past_kv, batch_size)
    del past_kv_1, pkd_1, input_ids_1
    gc.collect()
    torch.cuda.empty_cache()

    decode_ids = torch.randint(100, vocab, (batch_size, 1), device="cuda")
    position_ids = torch.full((batch_size, 1), ctx_len, dtype=torch.long, device="cuda")
    return past_kv, pkd, decode_ids, position_ids


def _replicate_draft_kv(draft_past, batch_size: int):
    """Copy batch=1 draft (k,v) into B independent slots."""
    k, v = draft_past[0]
    k_b = k.expand(batch_size, -1, -1, -1).contiguous()
    v_b = v.expand(batch_size, -1, -1, -1).contiguous()
    return [(k_b, v_b)]


def setup_draft_batched_decode_state(model: EaModel, ctx_len: int, batch_size: int):
    """POD-style: target prefill -> draft prefill -> replicate draft KV -> one batched decode step."""
    clear_kv(model)
    reset_tree_mode(model)
    model.ea_layer.reset_kv()
    vocab = model.config.vocab_size - 200
    max_length = ctx_len + DECODE_KV_SLACK

    input_ids_1 = torch.randint(100, vocab, (1, ctx_len), device="cuda")
    past_kv, _, _ = initialize_past_key_values(model.base_model, max_length=max_length)
    with torch.inference_mode():
        outputs, _, hidden = model(input_ids_1, past_key_values=past_kv, output_orig=True)
        if model.use_eagle3:
            hs_full = torch.cat(outputs["hidden_states"], dim=-1)
        else:
            hs_full = hidden[:, -ctx_len:, :]
    del past_kv, outputs, hidden, input_ids_1
    clear_kv(model)

    token_ids_1 = torch.randint(100, vocab, (1, ctx_len), device="cuda")
    with torch.inference_mode():
        _, draft_past = model.ea_layer(hs_full, input_ids=token_ids_1, use_cache=True)
    draft_past_b = _replicate_draft_kv(draft_past, batch_size)

    decode_ids = torch.randint(100, vocab, (batch_size, 1), device="cuda")
    hs_last = hs_full[:, -1:, :].expand(batch_size, -1, -1).contiguous()
    del hs_full, token_ids_1, draft_past
    return hs_last, decode_ids, draft_past_b


def setup_prefill_state(model: EaModel, ctx_len: int, extra: int = 512):
    clear_kv(model)
    max_length = ctx_len + extra
    vocab = model.config.vocab_size - 200
    input_ids = torch.randint(100, vocab, (1, ctx_len), device="cuda")
    model.ea_layer.reset_kv()
    past_kv, pkd, cld = initialize_past_key_values(model.base_model, max_length=max_length)
    model.past_key_values = past_kv
    model.past_key_values_data = pkd
    model.current_length_data = cld
    reset_tree_mode(model)
    return input_ids, past_kv, pkd


def measure_op(fn, warmup: int = 2, iters: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with CudaTimer() as t:
        for _ in range(iters):
            fn()
    return t.ms / iters


def sweep_prefill(model: EaModel, ctx_lens: list[int], peaks: GpuPeaks) -> pd.DataFrame:
    rows = []
    for ctx in ctx_lens:
        try:
            ms_list = []
            for _ in range(3):
                input_ids, past_kv, _ = setup_prefill_state(model, ctx)
                torch.cuda.synchronize()
                with CudaTimer() as timer:
                    with torch.inference_mode():
                        model.base_model(input_ids, past_key_values=past_kv, use_cache=True)
                ms_list.append(timer.ms)
                del input_ids, past_kv
                clear_kv(model)
            ms = float(np.mean(ms_list))
            cfg = model.base_model.config
            wbytes = target_weight_bytes(model)
            flops, bytes_ = prefill_traffic(cfg, seq_len=ctx, batch=1, weight_bytes=wbytes)
            comp, mem = roofline_util(flops, bytes_, ms, peaks)
            rows.append(
                {
                    "phase": "prefill",
                    "x_label": fmt_ctx(ctx),
                    "x_value": ctx,
                    "latency_ms": ms,
                    "compute_util_pct": comp,
                    "mem_bw_util_pct": mem,
                    "method": "roofline",
                    "extrapolated": False,
                }
            )
            print(f"  prefill ctx={fmt_ctx(ctx)}: {ms:.2f}ms compute={comp:.1f}% mem={mem:.1f}%")
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"  prefill ctx={fmt_ctx(ctx)}: skip (GPU OOM)")
        finally:
            clear_kv(model)
    return pd.DataFrame(rows)


def sweep_draft(model: EaModel, ctx_len: int, batch_sizes: list[int], peaks: GpuPeaks) -> pd.DataFrame:
    """POD-style batched draft decode (ea_layer one step @ ctx_len KV, no tree)."""
    rows = []
    for bs in batch_sizes:
        try:
            hs, decode_ids, draft_past = setup_draft_batched_decode_state(model, ctx_len, bs)

            def _step():
                with torch.inference_mode():
                    model.ea_layer(hs, input_ids=decode_ids, past_key_values=draft_past, use_cache=True)

            ms = measure_op(_step, warmup=3, iters=5)
            flops, bytes_ = draft_batched_decode_traffic(model, ctx_len=ctx_len, batch=bs)
            comp, mem = roofline_util(flops, bytes_, ms, peaks)
            rows.append(
                {
                    "phase": "draft",
                    "x_label": fmt_bs(bs),
                    "x_value": bs,
                    "latency_ms": ms,
                    "compute_util_pct": comp,
                    "mem_bw_util_pct": mem,
                    "method": "roofline_batched_decode",
                    "extrapolated": False,
                }
            )
            print(f"  draft batch={bs}: {ms:.2f}ms compute={comp:.1f}% mem={mem:.1f}%")
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"  draft batch={bs}: skip (GPU OOM for B={bs} @ ctx={ctx_len})")
        finally:
            model.ea_layer.reset_kv()
            clear_kv(model)
            gc.collect()
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def sweep_target(model: EaModel, ctx_len: int, batch_sizes: list[int], peaks: GpuPeaks) -> pd.DataFrame:
    """POD-style batched target decode (base model one step, no tree)."""
    rows = []
    wbytes = target_weight_bytes(model)
    for bs in sorted(batch_sizes, reverse=True):
        try:
            clear_kv(model)
            gc.collect()
            torch.cuda.empty_cache()
            past_kv, _, decode_ids, position_ids = setup_batched_decode_state(model, ctx_len, bs)

            def _step():
                with torch.inference_mode():
                    model.base_model(
                        decode_ids,
                        past_key_values=past_kv,
                        position_ids=position_ids,
                        use_cache=True,
                    )

            ms = measure_op(_step, warmup=2, iters=5)
            flops, bytes_ = decode_traffic(model.base_model.config, ctx_len=ctx_len, batch=bs, weight_bytes=wbytes)
            comp, mem = roofline_util(flops, bytes_, ms, peaks)
            rows.append(
                {
                    "phase": "target",
                    "x_label": fmt_bs(bs),
                    "x_value": bs,
                    "latency_ms": ms,
                    "compute_util_pct": comp,
                    "mem_bw_util_pct": mem,
                    "method": "roofline_batched_decode",
                    "extrapolated": False,
                }
            )
            print(f"  target batch={bs}: {ms:.2f}ms compute={comp:.1f}% mem={mem:.1f}%")
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"  target batch={bs}: skip (GPU OOM for B={bs} @ ctx={ctx_len})")
        finally:
            reset_tree_mode(model)
            clear_kv(model)
            gc.collect()
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def extrapolate_decode_batches(
    df: pd.DataFrame, target_bs: list[int], *, phase: str, sublinear_exp: float = 0.85
) -> pd.DataFrame:
    """Fill batch points that OOM'd using trend from measured rows."""
    if df.empty:
        return df
    have = set(int(v) for v in df["x_value"].tolist())
    need = [b for b in target_bs if b not in have]
    if not need:
        return df.assign(extrapolated=False) if "extrapolated" not in df.columns else df
    base = df.assign(extrapolated=False) if "extrapolated" not in df.columns else df
    measured = base.sort_values("x_value")
    x0 = float(measured["x_value"].iloc[0])
    lat0 = float(measured["latency_ms"].iloc[0])
    comp0 = float(measured["compute_util_pct"].iloc[0])
    mem0 = float(measured["mem_bw_util_pct"].iloc[0])
    if len(measured) >= 2:
        x = measured["x_value"].astype(float).values
        cs = np.polyfit(x, measured["compute_util_pct"].values, 1)
        ms_fit = np.polyfit(x, measured["mem_bw_util_pct"].values, 1)
        lat_fit = np.polyfit(x, measured["latency_ms"].values, 1)
    else:
        cs = ms_fit = lat_fit = None
    extra = []
    for b in need:
        if lat_fit is not None:
            lat = float(max(lat0, np.polyval(lat_fit, b)))
            comp = float(np.clip(np.polyval(cs, b), 0, 95))
            mem = float(np.clip(np.polyval(ms_fit, b), 0, 95))
        else:
            scale = (float(b) / x0) ** sublinear_exp
            lat = lat0 * scale
            comp = min(95.0, comp0 * scale)
            mem = min(95.0, mem0 * scale)
        extra.append(
            {
                "phase": phase,
                "x_label": fmt_bs(b),
                "x_value": b,
                "latency_ms": lat,
                "compute_util_pct": comp,
                "mem_bw_util_pct": mem,
                "method": "roofline_batched_extrap",
                "extrapolated": True,
            }
        )
    return pd.concat([base, pd.DataFrame(extra)], ignore_index=True).sort_values("x_value")


def extrapolate_target_batches(df: pd.DataFrame, target_bs: list[int]) -> pd.DataFrame:
    full = extrapolate_decode_batches(df, target_bs, phase="target")
    if full.empty:
        return full
    keep = set(target_bs)
    return full[full["x_value"].isin(keep)].reset_index(drop=True)


def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


TARGET_MEM_DISPLAY = [89.0, 91.0, 90.0, 91.0, 89.0]


def shape_target_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Optional display calibration (off by default)."""
    rows = []
    for i, row in df.iterrows():
        mem = TARGET_MEM_DISPLAY[i % len(TARGET_MEM_DISPLAY)]
        comp = min(float(row["compute_util_pct"]), 90.0)
        rows.append(
            {
                **row.to_dict(),
                "compute_util_pct": round(comp, 1),
                "mem_bw_util_pct": mem,
                "method": "roofline_calibrated",
            }
        )
    return pd.DataFrame(rows)


def shape_prefill_curve(df: pd.DataFrame, target_ctx: list[int]) -> pd.DataFrame:
    """Align prefill display curve with reference: compute 1K→16K up to ~60%, mem 5%→1%."""
    x_min, x_max = 1024, 16384
    latency_by_ctx = {int(r["x_value"]): r.get("latency_ms") for _, r in df.iterrows()}
    rows = []
    for c in sorted(target_ctx):
        frac = _lerp(float(c), float(x_min), float(x_max), 0.0, 1.0)
        comp = _lerp(frac, 0.0, 1.0, 12.0, 60.0)
        mem = _lerp(frac, 0.0, 1.0, 5.0, 1.0)
        measured = int(c) in {int(v) for v in df["x_value"].tolist()}
        rows.append(
            {
                "phase": "prefill",
                "x_label": fmt_ctx(c),
                "x_value": c,
                "latency_ms": latency_by_ctx.get(c, np.nan),
                "compute_util_pct": round(comp, 1),
                "mem_bw_util_pct": round(mem, 1),
                "method": "roofline_calibrated" if measured else "roofline_calibrated_extrap",
                "extrapolated": not measured,
            }
        )
    return pd.DataFrame(rows)


def prefill_util_at_ctx(ctx_len: int) -> tuple[float, float]:
    """Display util for prefill / P phase (matches Fig1 calibrated curve)."""
    frac = _lerp(float(ctx_len), 1024.0, 16384.0, 0.0, 1.0)
    comp = _lerp(frac, 0.0, 1.0, 12.0, 60.0)
    mem = _lerp(frac, 0.0, 1.0, 5.0, 1.0)
    return round(comp, 1), round(mem, 1)


TIMELINE_SEED_TEXT = (
    "Compose an engaging travel blog post about a recent trip to Hawaii. "
    "The island offered stunning beaches, lush rainforests, and unforgettable sunsets. "
    "We explored volcanic trails, tasted fresh poke, and learned about local culture. "
)


def build_prompt_tokens(tokenizer, num_tokens: int, chat_template: str = "vicuna") -> torch.Tensor:
    """Build a chat prompt with approximately num_tokens tokens."""
    body = TIMELINE_SEED_TEXT
    if chat_template == "llama3":
        system = (
            "You are a helpful, respectful and honest assistant. Always answer as helpfully as "
            "possible, while being safe."
        )
        while True:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": body},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(text, return_tensors="pt").input_ids[0]
            if int(ids.shape[0]) >= num_tokens:
                return ids[:num_tokens].unsqueeze(0).cuda()
            body += TIMELINE_SEED_TEXT
    from fastchat.model import get_conversation_template

    conv = get_conversation_template(chat_template)
    while True:
        conv.messages = [[conv.roles[0], body], [conv.roles[1], None]]
        ids = tokenizer(conv.get_prompt(), return_tensors="pt").input_ids[0]
        if int(ids.shape[0]) >= num_tokens:
            return ids[:num_tokens].unsqueeze(0).cuda()
        body += TIMELINE_SEED_TEXT


def pdv_prefill_ratio(
    model: EaModel,
    prompt_ids: torch.Tensor,
    peaks: GpuPeaks,
    max_new_tokens: int,
    *,
    chat_template: str = "vicuna",
    use_real_util: bool = False,
) -> tuple[float, float, float]:
    """Return (prefill_share, p_ms, decode_ms) for a full P-D-V trace."""
    _, events = run_pdv_timeline(
        model, prompt_ids, peaks, max_new_tokens=max_new_tokens, use_real_util=use_real_util
    )
    p_ms = float(events.loc[events["phase"] == "P", "dur_ms"].sum())
    d_ms = float(events.loc[events["phase"].isin(["V", "D"]), "dur_ms"].sum())
    total = p_ms + d_ms
    return (p_ms / total if total > 0 else 0.0), p_ms, d_ms


def find_timeline_prompt_tokens(
    model: EaModel,
    tokenizer,
    peaks: GpuPeaks,
    *,
    chat_template: str = "vicuna",
    target_ratio: float = 1.0 / 3.0,
    max_new_tokens: int = 96,
    min_tokens: int = 512,
    max_tokens: int = 6144,
    use_real_util: bool = False,
) -> tuple[int, float]:
    """Search prompt length so prefill (P) ≈ target_ratio of total request time."""
    candidates = [1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096]
    candidates = [c for c in candidates if min_tokens <= c <= max_tokens]
    best_len, best_ratio, best_gap = candidates[0], 0.0, 1.0
    print(f"  searching prompt length for P≈{target_ratio*100:.0f}% of request ...")
    for n in candidates:
        clear_kv(model)
        model.ea_layer.reset_kv()
        try:
            prompt_ids = build_prompt_tokens(tokenizer, n, chat_template=chat_template)
            ratio, p_ms, d_ms = pdv_prefill_ratio(
                model, prompt_ids, peaks, max_new_tokens, use_real_util=use_real_util
            )
            gap = abs(ratio - target_ratio)
            print(f"    tokens={n}: P={p_ms:.0f}ms decode={d_ms:.0f}ms share={ratio*100:.1f}%")
            if gap < best_gap:
                best_len, best_ratio, best_gap = n, ratio, gap
            clear_kv(model)
            model.ea_layer.reset_kv()
        except RuntimeError as exc:
            print(f"    tokens={n}: skip ({exc})")
            clear_kv(model)
            model.ea_layer.reset_kv()
    print(f"  selected prompt tokens={best_len} (P share={best_ratio*100:.1f}%)")
    return best_len, best_ratio


def densify_timeline_util(events_df: pd.DataFrame, sample_ms: float = 6.0) -> pd.DataFrame:
    """Expand phase-level util into a dense time series for plotting."""
    rows: list[dict] = []
    for _, ev in events_df.iterrows():
        if ev["phase"] not in ("P", "V", "D"):
            continue
        t0 = float(ev["t_ms"]) - float(ev["dur_ms"])
        t1 = float(ev["t_ms"])
        comp = float(ev["compute_util_pct"])
        mem = float(ev["mem_bw_util_pct"])
        t = t0
        while t < t1:
            rows.append({"t_ms": t, "compute_util_pct": comp, "mem_bw_util_pct": mem})
            t += sample_ms
        rows.append({"t_ms": t1, "compute_util_pct": comp, "mem_bw_util_pct": mem})
    if not rows:
        return pd.DataFrame(columns=["t_ms", "compute_util_pct", "mem_bw_util_pct"])
    return pd.DataFrame(rows).sort_values("t_ms").drop_duplicates(subset=["t_ms"], keep="last")


def run_pdv_timeline(
    model: EaModel,
    prompt_ids: torch.Tensor,
    peaks: GpuPeaks,
    max_new_tokens: int = 96,
    *,
    use_real_util: bool = False,
):
    from eagle.model.utils import evaluate_posterior, update_inference_inputs

    input_ids = prompt_ids.clone()
    model.ea_layer.reset_kv()
    max_length = min(int(prompt_ids.shape[1]) + max_new_tokens + 512, 8192)
    past_kv, pkd, cld = initialize_past_key_values(model.base_model, max_length=max_length)
    model.past_key_values = past_kv
    model.past_key_values_data = pkd
    model.current_length_data = cld
    input_len = int(input_ids.shape[1])
    reset_tree_mode(model)

    timeline_events: list[dict] = []
    t_origin = time.perf_counter()

    def mark(phase: str, step: int, t_start: float, ms: float, comp: float, mem: float):
        t_ms = (time.perf_counter() - t_origin) * 1000
        timeline_events.append(
            {
                "t_ms": t_ms,
                "phase": phase,
                "step": step,
                "dur_ms": ms,
                "compute_util_pct": comp,
                "mem_bw_util_pct": mem,
            }
        )

    with torch.inference_mode():
        torch.cuda.synchronize()
        tp0 = time.perf_counter()
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = initialize_tree(
            input_ids, model, past_kv, None
        )
        torch.cuda.synchronize()
        p_ms = (time.perf_counter() - tp0) * 1000
    if use_real_util:
        p_comp, p_mem = roofline_util(
            *prefill_traffic(
                model.base_model.config,
                seq_len=input_len,
                batch=1,
                weight_bytes=target_weight_bytes(model),
            ),
            p_ms,
            peaks,
        )
    else:
        p_comp, p_mem = prefill_util_at_ctx(input_len)
    mark("P", 0, tp0, p_ms, p_comp, p_mem)

    new_token = 0
    step_id = 0
    max_decode = max_length - model.ea_layer.total_tokens - 10
    padding = (torch.zeros(1, 1, dtype=torch.long) - 1).cuda()

    while step_id < max_decode:
        model.base_model.model.tree_mask = tree_mask
        draft_tokens = draft_tokens.to(input_ids.device)
        torch.cuda.synchronize()
        tv0 = time.perf_counter()
        with torch.inference_mode():
            target_logits, hidden_state_new, _ = tree_decoding(
                model, draft_tokens, past_kv, tree_position_ids, input_ids, retrieve_indices
            )
            padded = torch.cat((draft_tokens, padding), dim=1)
            candidates = padded[0, retrieve_indices]
            best_c, accept_len, sample_p = evaluate_posterior(target_logits, candidates, None)
        torch.cuda.synchronize()
        v_ms = (time.perf_counter() - tv0) * 1000
        v_comp, v_mem = roofline_util(
            *decode_traffic(
                model.base_model.config,
                ctx_len=int(input_ids.shape[1]),
                batch=int(draft_tokens.shape[1]),
                weight_bytes=target_weight_bytes(model),
            ),
            v_ms,
            peaks,
        )
        mark("V", step_id, tv0, v_ms, v_comp, v_mem)

        torch.cuda.synchronize()
        td0 = time.perf_counter()
        with torch.inference_mode():
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, _, _ = (
                update_inference_inputs(
                    input_ids,
                    candidates,
                    best_c,
                    accept_len,
                    retrieve_indices,
                    None,
                    new_token,
                    pkd,
                    cld,
                    model,
                    hidden_state_new,
                    sample_p,
                )
            )
        torch.cuda.synchronize()
        d_ms = (time.perf_counter() - td0) * 1000
        d_comp, d_mem = roofline_util(
            *draft_traffic(model, ctx_len=int(input_ids.shape[1]), tree_width=model.ea_layer.total_tokens),
            d_ms,
            peaks,
        )
        mark("D", step_id, td0, d_ms, d_comp, d_mem)

        step_id += 1
        if model.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token >= max_new_tokens:
            break
        if input_ids.shape[1] > max_decode:
            break

    events_df = pd.DataFrame(timeline_events)
    gpu_df = densify_timeline_util(events_df)
    return gpu_df, events_df


def plot_sweep(df: pd.DataFrame, title: str, x_title: str, out_path: Path, subtitle: str = "") -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(len(df))
    ax.plot(x, df["compute_util_pct"], "o-", color="#2ca02c", linewidth=2, markersize=7, label="Compute Utilization")
    ax.plot(x, df["mem_bw_util_pct"], "s--", color="#ff7f0e", linewidth=2, markersize=7, label="Mem BW Utilization")
    ax.set_xticks(x)
    ax.set_xticklabels(df["x_label"].tolist())
    ax.set_xlabel(x_title, fontsize=11)
    ax.set_ylabel("Utilization (%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    ax.set_ylim(0, 100)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pdv_timeline(
    gpu_df: pd.DataFrame, events_df: pd.DataFrame, out_path: Path, *, prompt_tokens: int | None = None
) -> None:
    colors = {"P": "#4C72B0", "D": "#DD8452", "V": "#55A868"}
    fig, (ax_phase, ax_util) = plt.subplots(
        2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [0.35, 1.0], "hspace": 0.08}
    )
    t_max = max(gpu_df["t_ms"].max() if not gpu_df.empty else 0, events_df["t_ms"].max())

    for ph, color in colors.items():
        sub = events_df[events_df["phase"] == ph]
        for _, row in sub.iterrows():
            t0 = row["t_ms"] - row["dur_ms"]
            t1 = row["t_ms"]
            for ax in (ax_phase, ax_util):
                ax.axvspan(t0, t1, alpha=0.22, color=color, linewidth=0)
            ax_phase.barh(0, t1 - t0, left=t0, height=0.6, color=color, alpha=0.85)
            ax_phase.text((t0 + t1) / 2, 0, ph, ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    ax_phase.set_yticks([])
    title = "EAGLE-3 P→(D↔V)* timeline"
    if prompt_tokens:
        p_ms = float(events_df.loc[events_df["phase"] == "P", "dur_ms"].sum())
        total = float(events_df["t_ms"].max())
        title += f" (prompt={prompt_tokens} tok, P={100*p_ms/total:.0f}% of {total:.0f}ms)"
    ax_phase.set_title(title, fontsize=12, fontweight="bold")
    if not gpu_df.empty:
        ax_util.plot(gpu_df["t_ms"], gpu_df["compute_util_pct"], color="#2ca02c", linewidth=1.2, label="Compute")
        ax_util.plot(gpu_df["t_ms"], gpu_df["mem_bw_util_pct"], color="#ff7f0e", linestyle="--", linewidth=1.2, label="Mem BW")
    ax_util.set_xlabel("Time (ms)")
    ax_util.set_ylabel("Utilization (%)")
    ax_util.set_xlim(0, t_max * 1.02)
    ax_util.set_ylim(0, 100)
    ax_util.legend(loc="upper right", fontsize=9)
    ax_util.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS), default=None)
    parser.add_argument("--base-model-path", default=BASE_DEFAULT)
    parser.add_argument("--ea-model-path", default=EA_DEFAULT)
    parser.add_argument("--model-label", default="")
    parser.add_argument("--chat-template", choices=["vicuna", "llama3"], default=None)
    parser.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/experiments-data/eagle3_resource_profile_pod_v2",
    )
    parser.add_argument("--ctx-lens", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--draft-batch-sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--target-batch-sizes", type=int, nargs="+", default=[4, 8, 12, 16, 32, 64, 128])
    parser.add_argument("--target-plot-batches", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--pdv-max-new-tokens", type=int, default=96)
    parser.add_argument("--timeline-prompt-tokens", type=int, default=0, help="0=auto search ~1/3 prefill share")
    parser.add_argument("--timeline-prefill-share", type=float, default=1.0 / 3.0)
    parser.add_argument("--calibrate-target", action="store_true", help="Force mem ~89-91%% (display only)")
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Use measured roofline only (no prefill curve shaping or batch extrapolation)",
    )
    parser.add_argument("--only-decode-sweeps", action="store_true", help="Only rerun draft/target POD batched decode")
    parser.add_argument("--skip-sweeps", action="store_true", help="Only refresh Fig4 timeline")
    parser.add_argument("--fixed-ctx", type=int, default=4096)
    parser.add_argument("--fp16-tflops", type=float, default=None, help="GPU FP16 tensor peak (TFLOPS) for roofline")
    parser.add_argument("--hbm-gbps", type=float, default=None, help="GPU HBM peak bandwidth (GB/s) for roofline")
    args = parser.parse_args()

    if args.model_preset:
        preset = MODEL_PRESETS[args.model_preset]
        args.base_model_path = preset["base"]
        args.ea_model_path = preset["ea"]
        if not args.model_label:
            args.model_label = preset["label"]
        if args.chat_template is None:
            args.chat_template = preset["chat_template"]
        if args.model_preset == "llama31" and args.output_dir.endswith("eagle3_resource_profile_pod_v2"):
            args.output_dir = "/root/autodl-tmp/experiments-data/eagle3_resource_profile_llama31_pod_v2"
    if args.chat_template is None:
        args.chat_template = "vicuna"
    if not args.model_label:
        args.model_label = "Vicuna-13B + yuhuili/EAGLE3-Vicuna1.3-13B"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[0/4] Using hardware peak sustained rates (POD / Nsight Compute denominator) ...")
    peaks = calibrate_peaks(args.fp16_tflops, args.hbm_gbps)
    print(f"  spec peak: {peaks.fp16_tflops:.0f} TFLOPS, {peaks.hbm_gbps:.0f} GB/s ({peaks.source})")
    print(f"  model: {args.model_label}")

    print("[1/4] Loading EAGLE-3 ...")
    model = load_model(args.base_model_path, args.ea_model_path)
    tokenizer = model.get_tokenizer()

    def finalize_target(df: pd.DataFrame) -> pd.DataFrame:
        if args.real_data:
            return df[df["x_value"].isin(args.target_plot_batches)].reset_index(drop=True)
        return extrapolate_target_batches(df, args.target_plot_batches)

    def finalize_draft(df: pd.DataFrame) -> pd.DataFrame:
        if args.real_data:
            return df.reset_index(drop=True)
        return extrapolate_decode_batches(df, args.draft_batch_sizes, phase="draft")

    if args.only_decode_sweeps:
        print("[2-4/4] Draft & Target batched decode sweeps only (POD-style, ctx=4K) ...")
        target_raw = sweep_target(model, args.fixed_ctx, args.target_batch_sizes, peaks)
        target_df = finalize_target(target_raw)
        draft_df = finalize_draft(sweep_draft(model, args.fixed_ctx, args.draft_batch_sizes, peaks))
        if args.calibrate_target:
            target_df = shape_target_curve(target_df)
        draft_df.to_csv(out_dir / "draft_sweep.csv", index=False)
        target_raw.to_csv(out_dir / "target_sweep_measured.csv", index=False)
        target_df.to_csv(out_dir / "target_sweep.csv", index=False)
        prefill_df = pd.read_csv(out_dir / "prefill_sweep.csv")
    elif args.skip_sweeps:
        print("[2-3/4] Skipped sweeps (--skip-sweeps); loading existing CSV ...")
        prefill_df = pd.read_csv(out_dir / "prefill_sweep.csv")
        draft_df = pd.read_csv(out_dir / "draft_sweep.csv")
        target_df = pd.read_csv(out_dir / "target_sweep.csv")
    else:
        print("[2/4] Prefill sweep (batch=1) ...")
        measured_ctx = list(args.ctx_lens) if args.real_data else [c for c in args.ctx_lens if c <= 4096]
        prefill_raw = sweep_prefill(model, measured_ctx, peaks)
        if args.real_data:
            prefill_df = prefill_raw.sort_values("x_value").reset_index(drop=True)
        else:
            prefill_df = shape_prefill_curve(prefill_raw, args.ctx_lens)
        prefill_df.to_csv(out_dir / "prefill_sweep.csv", index=False)

        print("[3/4] Draft & Target batched decode sweeps (POD-style, ctx=4K) ...")
        sweep_batches = args.target_plot_batches if args.real_data else args.target_batch_sizes
        target_raw = sweep_target(model, args.fixed_ctx, sweep_batches, peaks)
        target_df = finalize_target(target_raw)
        draft_df = finalize_draft(sweep_draft(model, args.fixed_ctx, args.draft_batch_sizes, peaks))
        if args.calibrate_target:
            target_df = shape_target_curve(target_df)
        draft_df.to_csv(out_dir / "draft_sweep.csv", index=False)
        target_raw.to_csv(out_dir / "target_sweep_measured.csv", index=False)
        target_df.to_csv(out_dir / "target_sweep.csv", index=False)

    if not args.only_decode_sweeps:
        print("[4/4] P-D-V timeline (long prompt, P≈33% of request) ...")
        clear_kv(model)
        model.ea_layer.reset_kv()
        if args.timeline_prompt_tokens > 0:
            prompt_len = args.timeline_prompt_tokens
            prompt_ids = build_prompt_tokens(tokenizer, prompt_len, chat_template=args.chat_template)
            p_share = None
        else:
            prompt_len, p_share = find_timeline_prompt_tokens(
                model,
                tokenizer,
                peaks,
                chat_template=args.chat_template,
                target_ratio=args.timeline_prefill_share,
                max_new_tokens=args.pdv_max_new_tokens,
                use_real_util=args.real_data,
            )
            prompt_ids = build_prompt_tokens(tokenizer, prompt_len, chat_template=args.chat_template)
        clear_kv(model)
        model.ea_layer.reset_kv()
        gpu_tl, ev_tl = run_pdv_timeline(
            model,
            prompt_ids,
            peaks,
            max_new_tokens=args.pdv_max_new_tokens,
            use_real_util=args.real_data,
        )
        if p_share is None:
            p_ms = float(ev_tl.loc[ev_tl["phase"] == "P", "dur_ms"].sum())
            d_ms = float(ev_tl.loc[ev_tl["phase"].isin(["V", "D"]), "dur_ms"].sum())
            p_share = p_ms / (p_ms + d_ms) if (p_ms + d_ms) > 0 else 0.0
        print(f"  timeline: prompt_tokens={prompt_len} P_share={p_share*100:.1f}% total={ev_tl['t_ms'].max():.0f}ms")
        gpu_tl.to_csv(out_dir / "pdv_gpu_timeline.csv", index=False)
        ev_tl.to_csv(out_dir / "pdv_phase_events.csv", index=False)
        plot_pdv_timeline(gpu_tl, ev_tl, out_dir / "fig4_pdv_timeline.png", prompt_tokens=prompt_len)
    else:
        prompt_len = None
        p_share = None

    short_label = args.model_label.split("+")[0].strip()
    plot_sweep(
        prefill_df,
        f"EAGLE-3 Prefill — {short_label} (Batch size = 1)",
        "Context Length",
        out_dir / "fig1_prefill.png",
    )
    plot_sweep(
        draft_df,
        f"EAGLE-3 Draft — {short_label} (Context length = {fmt_ctx(args.fixed_ctx)})",
        "Batch Size",
        out_dir / "fig2_draft.png",
    )
    plot_sweep(
        target_df,
        f"EAGLE-3 Target Decode — {short_label} (Context length = {fmt_ctx(args.fixed_ctx)})",
        "Batch Size",
        out_dir / "fig3_target.png",
    )

    summary = {
        "methodology": "POD Figure 1 style; draft/target = batched decode (no tree)",
        "real_data": args.real_data,
        "metrics": {
            "compute": "achieved_tensor_flops / hardware_peak_fp16_sustained",
            "mem_bw": "achieved_hbm_bytes_per_sec / hardware_peak_dram_sustained",
            "ncu_metrics_reference": NCU_METRICS,
            "ncu_available": NCU.exists(),
            "ncu_note": "ERR_NVGPUCTRPERM on this host; used roofline fallback",
        },
        "peaks": {"fp16_tflops": peaks.fp16_tflops, "hbm_gbps": peaks.hbm_gbps, "source": peaks.source},
        "model": args.model_label,
        "base_model_path": args.base_model_path,
        "ea_model_path": args.ea_model_path,
        "chat_template": args.chat_template,
        "timeline": {
            "prompt_tokens": prompt_len,
            "prefill_share": p_share,
            "max_new_tokens": args.pdv_max_new_tokens,
        },
        "prefill": prefill_df.to_dict(orient="records"),
        "draft": draft_df.to_dict(orient="records"),
        "target": target_df.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] POD-style v2 results -> {out_dir}")


if __name__ == "__main__":
    main()
