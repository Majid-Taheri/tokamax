#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rigorous Method B Leave-One-Out measurement with explicit XLA DCE protection.

This script benchmarks `_fused_dhu_wy_intra_cumsum_pallas_jit` across four modes:
  1. `full`     : Unmodified kernel (Dhu + WY + Intra + reverse cumsum).
  2. `no_dhu`   : Omit `compute_dhu_recurrence`, with explicit runtime-dynamic
                  guards so `compute_wy_backward` and `compute_intra_backward`
                  cannot be constant-folded or dead-code eliminated by XLA.
  3. `no_wy`    : Omit `compute_wy_backward`, with explicit runtime-dynamic
                  guards (`dAkk_local = bA`, `b_dvb = bdv + bv`, etc.) so all
                  8 matmuls in `compute_intra_backward` and `compute_dhu_recurrence`
                  are 100% preserved.
  4. `no_intra` : Omit `compute_intra_backward`, with explicit concatenation guard
                  `dq_total = dq_acc + jnp.concatenate([dAkk_local, bdAqk], axis=-1)`
                  so all [MB, 64, 64] elements of `dAkk_local` (and all 3 matmuls
                  computing it inside `compute_wy_backward`) directly feed HBM
                  output `dq_ref[:, 0, 0]`.

We test two configurations on Cloud TPU v7x (GhostFish):
  - Config A (Production 397B per-device config):
      H=64, B=1, T=16384, BT=64, NT=256, K=128, V=128, per_channel_gate=False,
      dtype=bfloat16, mini_batch=None (selects MB=16, grid=(4, 1, 256)).
  - Config B (Explicit mini_batch=2 config that produced the 16.401 ms figure):
      Same shapes, but mini_batch=2 (MB=2, grid=(32, 1, 256)).
"""

import builtins
import functools
from functools import partial
import os
import statistics
import sys
import time

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp

print = functools.partial(builtins.print, flush=True)

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_repo, "tokamax")) and _repo not in sys.path:
  sys.path.insert(0, _repo)

from tokamax._src.ops.experimental.kda import pallas_mosaic_tpu_bwd_kernel as bwd_mod


def make_dce_protected_kernel(mode: str):
  """Returns a Pallas kernel function for `mode` with explicit XLA DCE guards."""

  def _kernel(
      chunk_seg_ids_ref,
      q_ref,
      k_ref,
      v_ref,
      v_new_ref,
      qg_ref,
      kg_ref,
      w_ref,
      g_ref,
      beta_ref,
      A_ref,
      h_ref,
      do_ref,
      dv0_ref,
      dAqk_ref,
      dht_ref,
      dq_ref,
      dk_ref,
      dv_ref,
      db_ref,
      dg_ref,
      dh0_ref,
      dh_tmp_ref,
      *,
      BT,
      K,
      V,
      NT,
      scale,
      MB,
      per_channel_gate=False,
  ):
    head_group = pl.program_id(0)
    batch_idx = pl.program_id(1)
    rev_c = pl.program_id(2)
    chunk_id = NT - 1 - rev_c
    _, seq_idx, _, is_first_chunk, is_last_chunk = (
        bwd_mod._chunk_segment_metadata(chunk_seg_ids_ref, batch_idx, chunk_id, NT)
    )
    precision = (
        None if q_ref.dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST
    )

    @pl.when(is_last_chunk)
    def _():
      dh_tmp_ref[:] = dht_ref[:, 0, 0, :].astype(dh_tmp_ref.dtype)

    dh = dh_tmp_ref[:].astype(jnp.float32)
    bq = q_ref[:, 0, 0].astype(jnp.float32)
    bk = k_ref[:, 0, 0].astype(jnp.float32)
    bv = v_ref[:, 0, 0].astype(jnp.float32)
    bvn = v_new_ref[:, 0, 0].astype(jnp.float32)
    bqg = qg_ref[:, 0, 0].astype(jnp.float32)
    bkg = kg_ref[:, 0, 0]
    bw = w_ref[:, 0, 0].astype(jnp.float32)
    if g_ref.shape[-2] == 1:
      bg = g_ref[:, 0, 0, 0].astype(jnp.float32)[..., None]
    else:
      bg = g_ref[:, 0, 0].astype(jnp.float32)
    _gate_narrow = bg.shape[-1] != K
    if _gate_narrow:
      bg = jnp.broadcast_to(bg, bg.shape[:-1] + (K,))
    g_exp_last = jnp.exp2(bg[:, BT - 1, :])
    bb = beta_ref[:, 0, 0, 0].astype(jnp.float32)
    bA = A_ref[:, 0, 0].astype(jnp.float32)
    bh = h_ref[:, 0, 0].astype(jnp.float32)
    bdo = do_ref[:, 0, 0]
    bdv0 = dv0_ref[:, 0, 0].astype(jnp.float32)
    bdAqk = dAqk_ref[:, 0, 0].astype(jnp.float32)

    # =========================================================================
    # STEP 1: compute_dhu_recurrence
    # =========================================================================
    if mode != "no_dhu":
      bdv, dh_new = bwd_mod.compute_dhu_recurrence(
          bkg,
          dh,
          bdv0,
          dh_tmp_ref[:],
          g_exp_last,
          bqg,
          bw,
          bdo,
          scale,
          precision,
      )
      dh_tmp_ref[:] = dh_new
    else:
      # DCE Guard for `no_dhu`:
      # Do NOT use jnp.zeros_like(bdv0) or jnp.zeros_like(dh), because XLA's
      # AlgebraicSimplifier constant-folds 0 @ X -> 0 and deletes 7 of 8 matmuls
      # inside `compute_wy_backward`. Instead, pass runtime-dynamic VMEM inputs
      # bdv0 + bkg (shape [MB, BT, K]) and dh + concat([bqg, bw]) (shape [MB, K, V]),
      # which also keeps the HBM DMA loads of dv0_ref, kg_ref, qg_ref, w_ref live.
      bdv = bdv0 + bkg.astype(jnp.float32)
      dh_new = dh + jnp.concatenate([bqg, bw], axis=1)
      dh_tmp_ref[:] = dh_new

    # =========================================================================
    # STEP 2: compute_wy_backward
    # =========================================================================
    if mode != "no_wy":
      dq_acc, dk_acc, b_dvb, db_acc, dg_acc, dAkk_local = (
          bwd_mod.compute_wy_backward(
              bdo, bdv, bvn, bv, bh, dh, bq, bk, bg, bb, bA, scale, precision
          )
      )
    else:
      # DCE Guard for `no_wy`:
      # 1. `dAkk_local` feeds 4 of the 8 dot_general matmuls in `compute_intra_backward`.
      #    Passing compile-time zeros deletes half of `compute_intra_backward`.
      #    Pass runtime-dynamic `bA` ([MB, BT, BT]) so all 8 matmuls execute intact.
      # 2. Route `bdv` (from `compute_dhu_recurrence`) + `bv` into `b_dvb` -> `dv_ref`
      #    so `compute_dhu_recurrence` and `v_ref` DMA are 100% live.
      # 3. Route `bvn` and `bh` into `dq_acc` and `dk_acc` so `v_new_ref` and `h_ref`
      #    DMA loads are 100% live.
      dAkk_local = bA
      b_dvb = bdv + bv
      dq_acc = bvn
      dk_acc = bh[:, :BT, :]
      db_acc = bb
      dg_acc = bg

    # =========================================================================
    # STEP 3: compute_intra_backward + reverse cumsum
    # =========================================================================
    if mode != "no_intra":
      dq_total, dk_total, db_total, dg_total = bwd_mod.compute_intra_backward(
          bq,
          bk,
          bg,
          bb,
          bdAqk,
          dAkk_local,
          dq_acc,
          dk_acc,
          db_acc,
          dg_acc,
          precision=precision,
          per_channel_gate=per_channel_gate,
      )
      dg_reverse_cumsum = bwd_mod.compute_reverse_cumsum_dg(dg_total)
    else:
      # DCE Guard for `no_intra`:
      # `compute_wy_backward` produces `dq_acc`, `dk_acc`, `b_dvb`, `db_acc`,
      # `dg_acc`, and `dAkk_local` ([MB, 64, 64]).
      # Without `compute_intra_backward`, `dAkk_local` and `bdAqk` ([MB, 64, 64])
      # have no consumer, allowing XLA DCE to eliminate the 3 matmuls inside
      # `compute_wy_backward` that produce `dAkk_local` (lines 592-595, 608-609)
      # and the HBM DMA load of `dAqk_ref`.
      # Since K = 128 = 2 * BT (64), concatenating `dAkk_local` ([MB, 64, 64])
      # and `bdAqk` ([MB, 64, 64]) along axis -1 yields shape [MB, 64, 128],
      # matching `dq_acc` ([MB, 64, 128]) exactly.
      # Adding this directly to `dq_acc` ensures every single element [m, i, j]
      # of `dAkk_local` and `bdAqk` is written to HBM output `dq_ref[:, 0, 0]`.
      dakk_daqk_guard = jnp.concatenate([dAkk_local, bdAqk], axis=-1)
      dq_total = dq_acc + dakk_daqk_guard
      dk_total = dk_acc
      db_total = db_acc
      dg_total = dg_acc
      dg_reverse_cumsum = bwd_mod.compute_reverse_cumsum_dg(dg_total)

    dq_ref[:, 0, 0] = dq_total.astype(dq_ref.dtype)
    dk_ref[:, 0, 0] = dk_total.astype(dk_ref.dtype)
    dv_ref[:, 0, 0] = (b_dvb * bb[:, :, None]).astype(dv_ref.dtype)
    db_ref[:, 0, 0, 0] = db_total.astype(db_ref.dtype)
    if _gate_narrow:
      dg_reverse_cumsum = jnp.sum(dg_reverse_cumsum, axis=-1, keepdims=True)
    if dg_ref.shape[-2] == 1:
      dg_ref[:, 0, 0, 0] = dg_reverse_cumsum[..., 0].astype(dg_ref.dtype)
    else:
      dg_ref[:, 0, 0] = dg_reverse_cumsum.astype(dg_ref.dtype)

    @pl.when(is_first_chunk)
    def _():
      dh0_ref[:, 0, 0, :] = dh_tmp_ref[:].astype(dh0_ref.dtype)

  return _kernel


def run_config(
    label: str,
    H: int = 64,
    B: int = 1,
    T: int = 16384,
    K: int = 128,
    V: int = 128,
    BT: int = 64,
    mini_batch: int | None = None,
    iters: int = 20,
):
  NT = T // BT
  print("=" * 80)
  print(f"BENCHMARK CONFIGURATION: {label}")
  print(f"  Device               : {jax.devices()[0]} ({jax.devices()[0].device_kind})")
  print(f"  Batch size (B)       : {B}")
  print(f"  Number of heads (H)  : {H}")
  print(f"  Sequence length (T)  : {T}")
  print(f"  Chunk size (BT)      : {BT}")
  print(f"  Number of chunks (NT): {NT}")
  print(f"  Head dim (K, V)      : K={K}, V={V}")
  print(f"  Gate mode            : scalar gate (per_channel_gate=False, GW=1)")
  print(f"  Dtype                : bfloat16 inputs/outputs, float32 state/accumulators")
  print(f"  mini_batch setting   : {mini_batch} (effective MB={'16 (auto)' if mini_batch is None else mini_batch})")
  eff_mb = 16 if mini_batch is None else mini_batch
  print(f"  Pallas Grid          : (H // MB, B, NT) = ({H // eff_mb}, {B}, {NT})")
  print(f"  Dimension semantics  : ('parallel', 'parallel', 'arbitrary')")
  print("=" * 80)

  key = jax.random.PRNGKey(42)
  keys = jax.random.split(key, 16)
  dtype = jnp.bfloat16

  q = jax.random.normal(keys[0], (H, B, T, K), dtype=dtype)
  k = jax.random.normal(keys[1], (H, B, T, K), dtype=dtype)
  v = jax.random.normal(keys[2], (H, B, T, V), dtype=dtype)
  v_new = jax.random.normal(keys[3], (H, B, T, V), dtype=dtype)
  qg = jax.random.normal(keys[4], (H, B, T, K), dtype=dtype)
  kg = jax.random.normal(keys[5], (H, B, T, K), dtype=dtype)
  w = jax.random.normal(keys[6], (H, B, T, K), dtype=dtype)
  g = -jax.nn.softplus(
      jax.random.normal(keys[7], (H, B, T, 1), dtype=jnp.float32)
  )
  beta = jax.nn.sigmoid(
      jax.random.normal(keys[8], (H, B, T), dtype=jnp.float32)
  )
  A = jax.random.normal(keys[9], (H, B, T, BT), dtype=jnp.float32)
  h = jax.random.normal(keys[10], (H, B, NT, K, V), dtype=jnp.float32)
  do = jax.random.normal(keys[11], (H, B, T, V), dtype=dtype)
  dv0 = jax.random.normal(keys[12], (H, B, T, V), dtype=jnp.float32)
  dAqk = jax.random.normal(keys[13], (H, B, T, BT), dtype=jnp.float32)
  dht = jax.random.normal(keys[14], (B, 1, H, K, V), dtype=jnp.float32)
  scale = K**-0.5

  args = (q, k, v, v_new, qg, kg, w, g, beta, A, h, do, dv0, dAqk, dht)
  kwargs = dict(
      scale=scale,
      chunk_size=BT,
      use_exp2=True,
      mini_batch=mini_batch,
      return_dh0=True,
      max_num_segments=1,
      per_channel_gate=False,
  )

  orig_kernel = bwd_mod._fused_dhu_wy_intra_cumsum_kernel
  results = {}

  for mode in ["full", "no_dhu", "no_wy", "no_intra"]:
    bwd_mod._fused_dhu_wy_intra_cumsum_kernel = make_dce_protected_kernel(mode)
    bwd_mod._fused_dhu_wy_intra_cumsum_pallas_jit.clear_cache()

    # Warmup (2 runs to absorb XLA compilation and buffer allocation)
    for _ in range(2):
      out = bwd_mod._fused_dhu_wy_intra_cumsum_pallas_jit(*args, **kwargs)
      jax.block_until_ready(out)

    raw_ts = []
    for i in range(iters):
      t0 = time.perf_counter()
      out = bwd_mod._fused_dhu_wy_intra_cumsum_pallas_jit(*args, **kwargs)
      jax.block_until_ready(out)
      dt_ms = (time.perf_counter() - t0) * 1e3
      raw_ts.append(dt_ms)

    med = statistics.median(raw_ts)
    mean = statistics.mean(raw_ts)
    stdev = statistics.stdev(raw_ts)
    mn, mx = min(raw_ts), max(raw_ts)
    results[mode] = med

    raw_str = ", ".join(f"{x:.3f}" for x in raw_ts)
    print(f"\nMode: {mode}")
    print(f"  Raw %timeit ({iters} iters, ms): [{raw_str}]")
    print(
        f"  Stats: median = {med:.3f} ms | mean = {mean:.3f} ms | std = {stdev:.3f} ms"
        f" | min = {mn:.3f} ms | max = {mx:.3f} ms"
    )

  bwd_mod._fused_dhu_wy_intra_cumsum_kernel = orig_kernel
  bwd_mod._fused_dhu_wy_intra_cumsum_pallas_jit.clear_cache()

  t_full = results["full"]
  m_dhu = t_full - results["no_dhu"]
  m_wy = t_full - results["no_wy"]
  m_intra = t_full - results["no_intra"]
  sum_marginal = m_dhu + m_wy + m_intra
  rem_shared = t_full - sum_marginal

  print("\n" + "-" * 80)
  print(f"FINAL MEASURED CONTRIBUTION ({label}):")
  print(f"  Full Kernel Time (T_full)           : {t_full:8.3f} ms (100.0%)")
  print(
      f"  compute_dhu_recurrence (T - no_dhu) : {m_dhu:8.3f} ms"
      f" ({m_dhu / t_full * 100:5.1f}% of T_full | {m_dhu / sum_marginal * 100:5.1f}% of marginal compute)"
  )
  print(
      f"  compute_wy_backward    (T - no_wy)  : {m_wy:8.3f} ms"
      f" ({m_wy / t_full * 100:5.1f}% of T_full | {m_wy / sum_marginal * 100:5.1f}% of marginal compute)"
  )
  print(
      f"  compute_intra_backward (T - no_intra): {m_intra:8.3f} ms"
      f" ({m_intra / t_full * 100:5.1f}% of T_full | {m_intra / sum_marginal * 100:5.1f}% of marginal compute)"
  )
  print(
      f"  Sum of 3 marginal compute costs     : {sum_marginal:8.3f} ms"
      f" ({sum_marginal / t_full * 100:5.1f}% of T_full)"
  )
  print(
      f"  Remaining shared overhead (DMA/loop): {rem_shared:8.3f} ms"
      f" ({rem_shared / t_full * 100:5.1f}% of T_full)"
  )
  print("-" * 80 + "\n")
  return results


def main():
  # 1. Run exact production 397B per-device configuration (mini_batch=None -> MB=16)
  run_config(
      "Config 1: Production 397B per-device config (mini_batch=None -> MB=16)",
      mini_batch=None,
  )
  # 2. Run explicit mini_batch=2 configuration (MB=2) that produced the 16.401 ms baseline
  run_config(
      "Config 2: Explicit mini_batch=2 config (MB=2, 16.401 ms baseline)",
      mini_batch=2,
  )


if __name__ == "__main__":
  main()
