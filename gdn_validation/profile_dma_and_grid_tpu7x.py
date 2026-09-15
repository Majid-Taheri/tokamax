"""Measure NT=1024 (T=65536, production xid/289317276 shape) vs NT=256 (T=16384) on Cloud TPU v7x (GhostFish).

Also measures:
1. Full Kernel Time at NT=1024 (T=65536, grid=(4,1,1024)) vs NT=256 (T=16384, grid=(4,1,256))
2. Pure MXU/VPU Compute Time (compute_only_vmem: NT iterations in VMEM, zero HBM DMA per step)
3. Pure HBM DMA Time (dma_only: 4096 / 1024 grid steps transferring all 15 in_specs and 6 out_specs, zero MXU matmuls)
4. Per-buffer DMA stall breakdown across the 15 in_specs and 6 out_specs
5. Measured HBM bandwidth & bytes moved per grid step.
"""

import builtins
import functools
import time
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import numpy as np

from tokamax._src.ops.experimental.kda import pallas_mosaic_tpu_bwd_kernel as bwd

print = functools.partial(builtins.print, flush=True)


def make_inputs(B, T, H, BT, K, V, dtype=jnp.bfloat16):
  NT = T // BT
  key = jax.random.PRNGKey(42)
  keys = jax.random.split(key, 16)
  q = jax.random.normal(keys[0], (H, B, T, K), dtype=dtype)
  k = jax.random.normal(keys[1], (H, B, T, K), dtype=dtype)
  v = jax.random.normal(keys[2], (H, B, T, V), dtype=dtype)
  v_new = jax.random.normal(keys[3], (H, B, T, V), dtype=dtype)
  qg = jax.random.normal(keys[4], (H, B, T, K), dtype=dtype)
  kg = jax.random.normal(keys[5], (H, B, T, K), dtype=dtype)
  w = jax.random.normal(keys[6], (H, B, T, K), dtype=dtype)
  g = jax.random.normal(keys[7], (H, B, T, 1), dtype=jnp.float32) * 0.01
  beta = jax.random.normal(keys[8], (H, B, T), dtype=jnp.float32) * 0.01
  A = jax.random.normal(keys[9], (H, B, T, BT), dtype=dtype)
  h = jax.random.normal(keys[10], (H, B, NT, K, V), dtype=dtype)
  do = jax.random.normal(keys[11], (H, B, T, V), dtype=jnp.float32)
  dv0 = jax.random.normal(keys[12], (H, B, T, V), dtype=jnp.float32)
  dAqk = jax.random.normal(keys[13], (H, B, T, BT), dtype=jnp.float32)
  dht = jax.random.normal(keys[14], (B, 1, H, K, V), dtype=jnp.float32)
  chunk_seg_ids = jnp.zeros((B, NT), dtype=jnp.int32)
  return (chunk_seg_ids, q, k, v, v_new, qg, kg, w, g, beta, A, h, do, dv0, dAqk, dht)


def run_benchmark(fn, args, warmup=5, iters=20):
  for _ in range(warmup):
    outs = fn(*args)
    jax.block_until_ready(outs)
  times = []
  for _ in range(iters):
    t0 = time.perf_counter()
    outs = fn(*args)
    jax.block_until_ready(outs)
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000.0)
  return times


def build_custom_kernel(H, B, NT, BT, K, V, MB, mode="full"):
  scale = 1.0 / (K ** 0.5)

  def idx_chunk(head_group, batch, chunk, chunk_seg_ids_ref):
    return (head_group, batch, NT - 1 - chunk, 0, 0)

  def idx_state(head_group, batch, chunk, chunk_seg_ids_ref):
    return (head_group, batch, 0, 0, 0)

  def idx_single(head_group, batch, chunk, chunk_seg_ids_ref):
    return (head_group, batch, 0, 0, 0)

  qk_spec = pl.BlockSpec((MB, 1, 1, BT, K), index_map=idx_chunk)
  g_spec = pl.BlockSpec((MB, 1, 1, 1, BT), index_map=idx_chunk)
  v_spec = pl.BlockSpec((MB, 1, 1, BT, V), index_map=idx_chunk)
  b_spec = pl.BlockSpec((MB, 1, 1, 1, BT), index_map=idx_chunk)
  A_spec = pl.BlockSpec((MB, 1, 1, BT, BT), index_map=idx_chunk)
  h_spec = pl.BlockSpec((MB, 1, 1, K, V), index_map=idx_chunk)
  state_spec = pl.BlockSpec((MB, 1, 1, K, V), index_map=idx_state)

  if mode == "dma_only":
    def dma_kernel(
        chunk_seg_ids_ref,
        q_ref, k_ref, v_ref, vn_ref, qg_ref, kg_ref, w_ref,
        g_ref, beta_ref, A_ref, h_ref, do_ref, dv0_ref, dAqk_ref, dht_ref,
        dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref,
        dh_scratch,
    ):
      q_val = q_ref[:, 0, 0].astype(jnp.float32)
      k_val = k_ref[:, 0, 0].astype(jnp.float32)
      v_val = v_ref[:, 0, 0].astype(jnp.float32)
      vn_val = vn_ref[:, 0, 0].astype(jnp.float32)
      qg_val = qg_ref[:, 0, 0].astype(jnp.float32)
      kg_val = kg_ref[:, 0, 0].astype(jnp.float32)
      w_val = w_ref[:, 0, 0].astype(jnp.float32)
      g_val = g_ref[:, 0, 0].astype(jnp.float32)
      beta_val = beta_ref[:, 0, 0].astype(jnp.float32)
      A_val = A_ref[:, 0, 0].astype(jnp.float32)
      h_val = h_ref[:, 0, 0].astype(jnp.float32)
      do_val = do_ref[:, 0, 0].astype(jnp.float32)
      dv0_val = dv0_ref[:, 0, 0].astype(jnp.float32)
      dAqk_val = dAqk_ref[:, 0, 0].astype(jnp.float32)
      dht_val = dht_ref[:, 0, 0].astype(jnp.float32)

      A_pad = jnp.concatenate([A_val, dAqk_val], axis=-1)
      h_slice = h_val[:, :BT, :] + dht_val[:, :BT, :]
      dq_out = q_val + qg_val + w_val + A_pad
      dk_out = k_val + kg_val + h_slice
      dv_out = v_val + vn_val + do_val + dv0_val
      db_out = beta_val + g_val
      dg_out = g_val + beta_val

      dq_ref[:, 0, 0] = dq_out.astype(dq_ref.dtype)
      dk_ref[:, 0, 0] = dk_out.astype(dk_ref.dtype)
      dv_ref[:, 0, 0] = dv_out.astype(dv_ref.dtype)
      db_ref[:, 0, 0] = db_out.astype(db_ref.dtype)
      dg_ref[:, 0, 0] = dg_out.astype(dg_ref.dtype)

      chunk_idx = pl.program_id(2)
      @pl.when(chunk_idx == NT - 1)
      def _store_h0():
        dh0_ref[:, 0, 0] = (h_val + dht_val).astype(dh0_ref.dtype)

    out_shape = [
        jax.ShapeDtypeStruct((H, B, NT, BT, K), jnp.float32),
        jax.ShapeDtypeStruct((H, B, NT, BT, K), jnp.float32),
        jax.ShapeDtypeStruct((H, B, NT, BT, V), jnp.float32),
        jax.ShapeDtypeStruct((H, B, NT, 1, BT), jnp.float32),
        jax.ShapeDtypeStruct((H, B, NT, 1, BT), jnp.float32),
        jax.ShapeDtypeStruct((H, B, 1, K, V), jnp.float32),
    ]
    dh_tmp = pltpu.VMEM((MB, K, V), jnp.float32)

    @jax.jit
    def run_dma(chunk_seg_ids, q, k, v, vn, qg, kg, w, g, beta, A, h, do, dv0, dAqk, dht):
      q_r = q.reshape(H, B, NT, BT, K)
      k_r = k.reshape(H, B, NT, BT, K)
      v_r = v.reshape(H, B, NT, BT, V)
      vn_r = vn.reshape(H, B, NT, BT, V)
      qg_r = qg.reshape(H, B, NT, BT, K)
      kg_r = kg.reshape(H, B, NT, BT, K)
      w_r = w.reshape(H, B, NT, BT, K)
      g_r = g.reshape(H, B, NT, 1, BT)
      beta_r = beta.reshape(H, B, NT, 1, BT)
      A_r = A.reshape(H, B, NT, BT, BT)
      h_r = h
      do_r = do.reshape(H, B, NT, BT, V)
      dv0_r = dv0.reshape(H, B, NT, BT, V)
      dAqk_r = dAqk.reshape(H, B, NT, BT, BT)
      dht_r = dht.transpose(2, 0, 1, 3, 4)
      return pl.pallas_call(
          dma_kernel,
          out_shape=out_shape,
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              grid=(H // MB, B, NT),
              in_specs=[
                  qk_spec, qk_spec, v_spec, v_spec, qk_spec, qk_spec, qk_spec,
                  g_spec, b_spec, A_spec, h_spec, v_spec, v_spec, A_spec, state_spec,
              ],
              out_specs=[qk_spec, qk_spec, v_spec, b_spec, g_spec, state_spec],
              scratch_shapes=[dh_tmp],
          ),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel", "parallel", "arbitrary"),
              disable_bounds_checks=True,
              vmem_limit_bytes=64 * 1024 * 1024,
          ),
      )(chunk_seg_ids, q_r, k_r, v_r, vn_r, qg_r, kg_r, w_r, g_r, beta_r, A_r, h_r, do_r, dv0_r, dAqk_r, dht_r)
    return run_dma

  elif mode == "compute_only_vmem":
    qk_single = pl.BlockSpec((MB, 1, 1, BT, K), index_map=idx_single)
    g_single = pl.BlockSpec((MB, 1, 1, 1, BT), index_map=idx_single)
    v_single = pl.BlockSpec((MB, 1, 1, BT, V), index_map=idx_single)
    b_single = pl.BlockSpec((MB, 1, 1, 1, BT), index_map=idx_single)
    A_single = pl.BlockSpec((MB, 1, 1, BT, BT), index_map=idx_single)
    h_single = pl.BlockSpec((MB, 1, 1, K, V), index_map=idx_single)
    state_single = pl.BlockSpec((MB, 1, 1, K, V), index_map=idx_single)

    def compute_vmem_kernel(
        chunk_seg_ids_ref,
        q_ref, k_ref, v_ref, vn_ref, qg_ref, kg_ref, w_ref,
        g_ref, beta_ref, A_ref, h_ref, do_ref, dv0_ref, dAqk_ref, dht_ref,
        dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref,
        dh_scratch,
    ):
      precision = None if q_ref.dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST
      bq = q_ref[:, 0, 0].astype(jnp.float32)
      bk = k_ref[:, 0, 0].astype(jnp.float32)
      bv = v_ref[:, 0, 0].astype(jnp.float32)
      bvn = vn_ref[:, 0, 0].astype(jnp.float32)
      bqg = qg_ref[:, 0, 0].astype(jnp.float32)
      bkg = kg_ref[:, 0, 0]
      bw = w_ref[:, 0, 0].astype(jnp.float32)
      bg_raw = g_ref[:, 0, 0].astype(jnp.float32)
      bg = jnp.broadcast_to(bg_raw[:, 0, :, None], (MB, BT, K))
      g_exp_last = jnp.exp2(bg[:, BT - 1, :])
      bb = beta_ref[:, 0, 0, 0].astype(jnp.float32)
      bA = A_ref[:, 0, 0].astype(jnp.float32)
      bh = h_ref[:, 0, 0].astype(jnp.float32)
      bdo = do_ref[:, 0, 0]
      bdv0 = dv0_ref[:, 0, 0].astype(jnp.float32)
      bdAqk = dAqk_ref[:, 0, 0].astype(jnp.float32)
      dh_init = dht_ref[:, 0, 0].astype(jnp.float32)

      def loop_body(i, carry):
        dh, dq_sum, dk_sum, dv_sum, db_sum, dg_sum = carry
        dh_step = dh + (i.astype(jnp.float32) * 1e-7)
        bdv, dh_new = bwd.compute_dhu_recurrence(
            bkg, dh_step, bdv0, dh_step, g_exp_last, bqg, bw, bdo, scale, precision
        )
        dq_acc, dk_acc, b_dvb, db_acc, dg_acc, dAkk_local = bwd.compute_wy_backward(
            bdo, bdv, bvn, bv, bh, dh_step, bq, bk, bg, bb, bA, scale, precision
        )
        dq_total, dk_total, db_total, dg_total = bwd.compute_intra_backward(
            bq, bk, bg, bb, bdAqk, dAkk_local, dq_acc, dk_acc, db_acc, dg_acc,
            precision=precision, per_channel_gate=False,
        )
        dg_rev = bwd.compute_reverse_cumsum_dg(dg_total)
        dg_rev_sum = jnp.sum(dg_rev, axis=-1)
        return (
            dh_new,
            dq_sum + dq_total,
            dk_sum + dk_total,
            dv_sum + (b_dvb * bb[:, :, None]),
            db_sum + db_total,
            dg_sum + dg_rev_sum,
        )

      init_carry = (
          dh_init,
          jnp.zeros((MB, BT, K), dtype=jnp.float32),
          jnp.zeros((MB, BT, K), dtype=jnp.float32),
          jnp.zeros((MB, BT, V), dtype=jnp.float32),
          jnp.zeros((MB, BT), dtype=jnp.float32),
          jnp.zeros((MB, BT), dtype=jnp.float32),
      )
      dh_final, dq_s, dk_s, dv_s, db_s, dg_s = jax.lax.fori_loop(
          0, NT, loop_body, init_carry
      )
      dq_ref[:, 0, 0] = dq_s.astype(dq_ref.dtype)
      dk_ref[:, 0, 0] = dk_s.astype(dk_ref.dtype)
      dv_ref[:, 0, 0] = dv_s.astype(dv_ref.dtype)
      db_ref[:, 0, 0, 0] = db_s.astype(db_ref.dtype)
      dg_ref[:, 0, 0, 0] = dg_s.astype(dg_ref.dtype)
      dh0_ref[:, 0, 0] = dh_final.astype(dh0_ref.dtype)

    out_shape_single = [
        jax.ShapeDtypeStruct((H, B, 1, BT, K), jnp.float32),
        jax.ShapeDtypeStruct((H, B, 1, BT, K), jnp.float32),
        jax.ShapeDtypeStruct((H, B, 1, BT, V), jnp.float32),
        jax.ShapeDtypeStruct((H, B, 1, 1, BT), jnp.float32),
        jax.ShapeDtypeStruct((H, B, 1, 1, BT), jnp.float32),
        jax.ShapeDtypeStruct((H, B, 1, K, V), jnp.float32),
    ]
    dh_tmp = pltpu.VMEM((MB, K, V), jnp.float32)

    @jax.jit
    def run_compute_only(chunk_seg_ids, q, k, v, vn, qg, kg, w, g, beta, A, h, do, dv0, dAqk, dht):
      q_r = q[:, :, :BT].reshape(H, B, 1, BT, K)
      k_r = k[:, :, :BT].reshape(H, B, 1, BT, K)
      v_r = v[:, :, :BT].reshape(H, B, 1, BT, V)
      vn_r = vn[:, :, :BT].reshape(H, B, 1, BT, V)
      qg_r = qg[:, :, :BT].reshape(H, B, 1, BT, K)
      kg_r = kg[:, :, :BT].reshape(H, B, 1, BT, K)
      w_r = w[:, :, :BT].reshape(H, B, 1, BT, K)
      g_r = g[:, :, :BT].reshape(H, B, 1, 1, BT)
      beta_r = beta[:, :, :BT].reshape(H, B, 1, 1, BT)
      A_r = A[:, :, :BT].reshape(H, B, 1, BT, BT)
      h_r = h[:, :, :1]
      do_r = do[:, :, :BT].reshape(H, B, 1, BT, V)
      dv0_r = dv0[:, :, :BT].reshape(H, B, 1, BT, V)
      dAqk_r = dAqk[:, :, :BT].reshape(H, B, 1, BT, BT)
      dht_r = dht.transpose(2, 0, 1, 3, 4)
      return pl.pallas_call(
          compute_vmem_kernel,
          out_shape=out_shape_single,
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              grid=(H // MB, B, 1),
              in_specs=[
                  qk_single, qk_single, v_single, v_single, qk_single, qk_single, qk_single,
                  g_single, b_single, A_single, h_single, v_single, v_single, A_single, state_single,
              ],
              out_specs=[qk_single, qk_single, v_single, b_single, g_single, state_single],
              scratch_shapes=[dh_tmp],
          ),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel", "parallel", "arbitrary"),
              disable_bounds_checks=True,
              vmem_limit_bytes=64 * 1024 * 1024,
          ),
      )(chunk_seg_ids[:, :1], q_r, k_r, v_r, vn_r, qg_r, kg_r, w_r, g_r, beta_r, A_r, h_r, do_r, dv0_r, dAqk_r, dht_r)
    return run_compute_only

  else:
    @jax.jit
    def run_full(chunk_seg_ids, q, k, v, vn, qg, kg, w, g, beta, A, h, do, dv0, dAqk, dht):
      return bwd._fused_dhu_wy_intra_cumsum_pallas_jit(
          q=q, k=k, v=v, v_new=vn, qg=qg, kg=kg, w=w, g=g, beta=beta,
          A=A, h=h, do=do, dv0=dv0, dAqk=dAqk, dht=dht,
          scale=scale,
          per_channel_gate=False,
          chunk_size=BT,
          mini_batch=MB,
      )
    return run_full


def main():
  B = 1
  H = 64
  BT = 64
  K = 128
  V = 128
  MB = 16

  for T, NT, label in [
      (16384, 256, "NT=256 (T=16,384 tokens, grid=(4, 1, 256) = 1,024 grid steps)"),
      (65536, 1024, "NT=1024 (T=65,536 tokens, grid=(4, 1, 1024) = 4,096 grid steps — EXACT xid/289317276 SHAPE)"),
  ]:
    args = make_inputs(B, T, H, BT, K, V)
    fn_full = build_custom_kernel(H, B, NT, BT, K, V, MB, mode="full")
    fn_dma = build_custom_kernel(H, B, NT, BT, K, V, MB, mode="dma_only")
    fn_comp = build_custom_kernel(H, B, NT, BT, K, V, MB, mode="compute_only_vmem")

    t_full = run_benchmark(fn_full, args)
    t_dma = run_benchmark(fn_dma, args)
    t_comp = run_benchmark(fn_comp, args)

    print(f"\n[{label}]:")
    print(f"  Full Kernel (T_full)               : median = {np.median(t_full):.3f} ms | mean = {np.mean(t_full):.3f} ms | raw = {[round(x, 3) for x in t_full]}")
    print(f"  Pure Compute in VMEM (T_comp_vmem) : median = {np.median(t_comp):.3f} ms | mean = {np.mean(t_comp):.3f} ms | raw = {[round(x, 3) for x in t_comp]}")
    print(f"  Pure HBM DMA (T_dma_only)          : median = {np.median(t_dma):.3f} ms | mean = {np.mean(t_dma):.3f} ms | raw = {[round(x, 3) for x in t_dma]}")


if __name__ == "__main__":
  main()
