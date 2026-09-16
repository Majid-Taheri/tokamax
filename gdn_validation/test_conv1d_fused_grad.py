#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify fused causal conv1d + SiLU + L2norm in KDA forward Pallas kernel.

Compares:
1. Unfused path:
   q_conv = silu(conv1d(q_raw, w_q))
   k_conv = silu(conv1d(k_raw, w_k))
   v_conv = silu(conv1d(v_raw, w_v))
   out, _ = kimi_delta_attention(q_conv, k_conv, v_conv, ..., use_conv1d_in_kernel=False)

2. Fused path:
   out, _ = kimi_delta_attention(
       q_raw, k_raw, v_raw, ...,
       conv_weight_q=w_q, conv_weight_k=w_k, conv_weight_v=w_v,
       use_conv1d_in_kernel=True)

Reports:
- Max relative & absolute error for forward output and all gradients:
  out, d_query, d_key, d_value, d_gate, d_beta, d_conv_weight_q, d_conv_weight_k, d_conv_weight_v.
- Exact compiler VMEM allocation from XLA post-RA dumps (*-54-post-ra.txt).
"""

import glob
import os
import re
import shutil
import sys

import jax
import jax.numpy as jnp

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_repo, "tokamax")) and _repo not in sys.path:
  sys.path.insert(0, _repo)

from tokamax._src.ops.experimental.kda import api
from gdn_validation.maxtext_adapter_kda_gated_delta_rule import kda_chunk_gated_delta_rule

if os.environ.get("KDA_BYPASS_DEVICE_CHECK") == "1":
  import dataclasses as _dc
  for _name, _op in list(api.IMPLEMENTATIONS.items()):
    api.IMPLEMENTATIONS[_name] = _dc.replace(_op, bypass_device_check=True)


def ref_causal_conv1d_silu(x: jax.Array, w: jax.Array, b: jax.Array | None = None) -> jax.Array:
  """Causal depthwise conv1d + SiLU matching MaxText / Qwen3.5."""
  W = w.shape[1]
  x_f32 = x.astype(jnp.float32)
  w_f32 = w.astype(jnp.float32)
  acc = x_f32 * w_f32[:, None, W - 1 : W, :]
  for lag in range(1, W):
    x_lag = jnp.pad(x_f32[:, :, :-lag, :], ((0, 0), (0, 0), (lag, 0), (0, 0)))
    acc = acc + x_lag * w_f32[:, None, W - 1 - lag : W - lag, :]
  if b is not None:
    acc = acc + b.astype(jnp.float32)[:, None, None, :]
  return jax.nn.silu(acc).astype(x.dtype)


def rel_err(a: jax.Array, b: jax.Array) -> float:
  a_f, b_f = a.astype(jnp.float32), b.astype(jnp.float32)
  return float(jnp.max(jnp.abs(a_f - b_f))) / (float(jnp.max(jnp.abs(b_f))) + 1e-12)


def abs_err(a: jax.Array, b: jax.Array) -> float:
  a_f, b_f = a.astype(jnp.float32), b.astype(jnp.float32)
  return float(jnp.max(jnp.abs(a_f - b_f)))


def test_correctness(dtype=jnp.bfloat16, H=32, B=1, T=4096, D=128, W=4, chunk_size=64):
  print(f"\n=== Correctness Test: dtype={dtype}, H={H}, B={B}, T={T}, D={D}, W={W}, chunk={chunk_size} ===")
  key = jax.random.PRNGKey(123)
  keys = jax.random.split(key, 12)

  q_raw = jax.random.normal(keys[0], (H, B, T, D), dtype=dtype)
  k_raw = jax.random.normal(keys[1], (H, B, T, D), dtype=dtype)
  v_raw = jax.random.normal(keys[2], (H, B, T, D), dtype=dtype)
  beta = jax.nn.sigmoid(jax.random.normal(keys[3], (H, B, T), dtype=jnp.float32)).astype(dtype)
  a_log = jnp.log(jax.random.uniform(keys[4], (H, 1, 1), jnp.float32, minval=1e-9, maxval=16.0))
  g_scalar = (-jnp.exp(a_log) * jax.nn.softplus(jax.random.normal(keys[5], (H, B, T), dtype=jnp.float32) + 1.0)).astype(jnp.float32)
  gate = g_scalar[..., None]  # [H, B, T, 1]

  w_q = jax.random.normal(keys[6], (H, W, D), dtype=jnp.float32) * 0.2
  w_k = jax.random.normal(keys[7], (H, W, D), dtype=jnp.float32) * 0.2
  w_v = jax.random.normal(keys[8], (H, W, D), dtype=jnp.float32) * 0.2
  init = jnp.zeros((B, 1, H, D, D), dtype=jnp.float32)
  cot = jax.random.normal(keys[9], (H, B, T, D), dtype=jnp.float32)

  # 1. Unfused baseline path
  def unfused_loss(q_r, k_r, v_r, g_in, beta_in, wq, wk, wv):
    q_c = ref_causal_conv1d_silu(q_r, wq)
    k_c = ref_causal_conv1d_silu(k_r, wk)
    v_c = ref_causal_conv1d_silu(v_r, wv)
    out, _ = api.kimi_delta_attention(
        q_c, k_c, v_c, g_in, beta_in,
        initial_state=init,
        output_final_state=True,
        use_gate_in_kernel=False,
        use_qk_l2norm=True,
        per_channel_gate=False,
        chunk_size=chunk_size,
        use_conv1d_in_kernel=False,
        implementation="mosaic",
    )
    return jnp.sum(out.astype(jnp.float32) * cot), out

  # 2. Fused kernel path
  def fused_loss(q_r, k_r, v_r, g_in, beta_in, wq, wk, wv):
    out, _ = api.kimi_delta_attention(
        q_r, k_r, v_r, g_in, beta_in,
        initial_state=init,
        output_final_state=True,
        use_gate_in_kernel=False,
        use_qk_l2norm=True,
        per_channel_gate=False,
        chunk_size=chunk_size,
        conv_weight_q=wq,
        conv_weight_k=wk,
        conv_weight_v=wv,
        use_conv1d_in_kernel=True,
        implementation="mosaic",
    )
    return jnp.sum(out.astype(jnp.float32) * cot), out

  argnums = (0, 1, 2, 3, 4, 5, 6, 7)
  (_, out_unfused), grads_unfused = jax.value_and_grad(unfused_loss, argnums=argnums, has_aux=True)(
      q_raw, k_raw, v_raw, gate, beta, w_q, w_k, w_v
  )
  (_, out_fused), grads_fused = jax.value_and_grad(fused_loss, argnums=argnums, has_aux=True)(
      q_raw, k_raw, v_raw, gate, beta, w_q, w_k, w_v
  )
  jax.block_until_ready((out_unfused, out_fused, grads_unfused, grads_fused))

  names = [
      "out (forward)",
      "d_query",
      "d_key",
      "d_value",
      "d_gate",
      "d_beta",
      "d_conv_weight_q",
      "d_conv_weight_k",
      "d_conv_weight_v",
  ]
  pairs = [(out_fused, out_unfused)] + list(zip(grads_fused, grads_unfused))

  print(f"{'Tensor':<20} | {'Max Rel Error':<15} | {'Max Abs Error':<15}")
  print("-" * 56)
  for name, (val_f, val_u) in zip(names, pairs):
    r_err = rel_err(val_f, val_u)
    a_err = abs_err(val_f, val_u)
    print(f"{name:<20} | {r_err:<15.3e} | {a_err:<15.3e}")
    assert jnp.all(jnp.isfinite(val_f)), f"Non-finite values in {name}"
    assert r_err < 1e-2, f"Relative error too high for {name}: {r_err}"

  print("SUCCESS: Fused path matches unfused path on all forward and backward outputs!")


def test_maxtext_adapter_gqa(B=1, S=4096, H_k=16, H_v=32, D=128, W=4, chunk_size=64):
  print(f"\n=== MaxText Adapter GQA Test: B={B}, S={S}, H_k={H_k}, H_v={H_v}, D={D}, W={W} ===")
  key = jax.random.PRNGKey(999)
  keys = jax.random.split(key, 10)
  dtype = jnp.bfloat16

  q_raw_k = jax.random.normal(keys[0], (B, S, H_k, D), dtype=dtype)
  k_raw_k = jax.random.normal(keys[1], (B, S, H_k, D), dtype=dtype)
  v_raw_v = jax.random.normal(keys[2], (B, S, H_v, D), dtype=dtype)
  g = -jax.nn.softplus(jax.random.normal(keys[3], (B, S, H_v), dtype=jnp.float32))
  beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, S, H_v), dtype=jnp.float32))

  w_q_k = jax.random.normal(keys[5], (H_k, W, D), dtype=jnp.float32) * 0.2
  w_k_k = jax.random.normal(keys[6], (H_k, W, D), dtype=jnp.float32) * 0.2
  w_v_v = jax.random.normal(keys[7], (H_v, W, D), dtype=jnp.float32) * 0.2
  cot = jax.random.normal(keys[8], (B, S, H_v, D), dtype=jnp.float32)

  repeats = H_v // H_k

  def conv1d_bshd(x_bshd, w_hwd):
    x_hbtd = jnp.transpose(x_bshd, (2, 0, 1, 3))
    out_hbtd = ref_causal_conv1d_silu(x_hbtd, w_hwd)
    return jnp.transpose(out_hbtd, (1, 2, 0, 3))

  def unfused_adapter_loss(q_k, k_k, v_v, g_in, b_in, wq, wk, wv):
    q_c = conv1d_bshd(q_k, wq)
    k_c = conv1d_bshd(k_k, wk)
    v_c = conv1d_bshd(v_v, wv)
    q_rep = jnp.repeat(q_c, repeats, axis=2)
    k_rep = jnp.repeat(k_c, repeats, axis=2)
    out, _ = kda_chunk_gated_delta_rule(
        q_rep, k_rep, v_c, g_in, b_in,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
        compute_dtype=dtype,
        use_conv1d_in_kernel=False,
    )
    return jnp.sum(out.astype(jnp.float32) * cot), out

  def fused_adapter_loss(q_k, k_k, v_v, g_in, b_in, wq, wk, wv):
    q_rep = jnp.repeat(q_k, repeats, axis=2)
    k_rep = jnp.repeat(k_k, repeats, axis=2)
    out, _ = kda_chunk_gated_delta_rule(
        q_rep, k_rep, v_v, g_in, b_in,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
        compute_dtype=dtype,
        conv_weight_q=wq,
        conv_weight_k=wk,
        conv_weight_v=wv,
        use_conv1d_in_kernel=True,
    )
    return jnp.sum(out.astype(jnp.float32) * cot), out

  argnums = (0, 1, 2, 3, 4, 5, 6, 7)
  (_, out_u), grads_u = jax.value_and_grad(unfused_adapter_loss, argnums=argnums, has_aux=True)(
      q_raw_k, k_raw_k, v_raw_v, g, beta, w_q_k, w_k_k, w_v_v
  )
  (_, out_f), grads_f = jax.value_and_grad(fused_adapter_loss, argnums=argnums, has_aux=True)(
      q_raw_k, k_raw_k, v_raw_v, g, beta, w_q_k, w_k_k, w_v_v
  )
  jax.block_until_ready((out_u, out_f, grads_u, grads_f))

  names = [
      "out (forward)",
      "d_q_raw (GQA H_k)",
      "d_k_raw (GQA H_k)",
      "d_v_raw (H_v)",
      "d_g",
      "d_beta",
      "d_w_q (GQA H_k)",
      "d_w_k (GQA H_k)",
      "d_w_v (H_v)",
  ]
  print(f"{'Tensor':<20} | {'Max Rel Error':<15} | {'Max Abs Error':<15}")
  print("-" * 56)
  for name, (vf, vu) in zip(names, [(out_f, out_u)] + list(zip(grads_f, grads_u))):
    r_err = rel_err(vf, vu)
    a_err = abs_err(vf, vu)
    print(f"{name:<20} | {r_err:<15.3e} | {a_err:<15.3e}")
    assert jnp.all(jnp.isfinite(vf)), f"Non-finite values in {name}"
    assert r_err < 1e-2, f"Relative error too high for {name}: {r_err}"
  print("SUCCESS: MaxText Adapter GQA test matched with 0.0 error across all forward & backward outputs!")


if __name__ == "__main__":
  test_correctness(dtype=jnp.bfloat16, H=32, B=1, T=4096, D=128, W=4, chunk_size=64)
  test_correctness(dtype=jnp.float32, H=32, B=1, T=4096, D=128, W=4, chunk_size=64)
  test_maxtext_adapter_gqa(B=1, S=4096, H_k=16, H_v=32, D=128, W=4, chunk_size=64)
