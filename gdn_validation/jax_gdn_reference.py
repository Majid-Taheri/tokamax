#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The pure-JAX Gated Delta Net, before and after MaxText PR #4577.

Copied verbatim from AI-Hypercomputer/maxtext main
(`src/maxtext/models/qwen3.py`, `jax_chunk_gated_delta_rule`) so the benchmark
does not need MaxText installed. Two variants:

  jax_gdn_solve_triangular   main as it stands -- jax.scipy.linalg.solve_triangular
  jax_gdn_log_depth          with PR #4577 applied -- invert_unit_lower_triangular_log_depth

PR #4577 replaces the triangular solve with a log-depth Newton-Schulz iteration
carrying a custom VJP. Its author measured, on TPU v6e at chunk_size 128,
forward 96.96 -> 23 ms and backward 127 -> 52 ms. That makes it the baseline
any kernel should be compared against; comparisons against the unoptimised
path overstate the kernel's advantage.

`invert_unit_lower_triangular_log_depth` and the l2norm helper below are
reproduced from the same sources.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import Array
from jax import lax


def l2norm(x, dim=-1, eps=1e-6):
  """MaxText's maxtext.layers.normalizations.l2norm."""
  return x * jax.lax.rsqrt(jnp.sum(x * x, axis=dim, keepdims=True) + eps)


# ---------------------------------------------------------------- PR #4577
@jax.custom_vjp
def invert_unit_lower_triangular_log_depth(S):
  """Computes (I + S)^-1 for strictly lower triangular S, log-depth."""
  chunk_size = S.shape[-1]
  S_strict = jnp.tril(S, k=-1)
  identity = jnp.eye(chunk_size, dtype=S.dtype)
  A = identity - S_strict
  E = jnp.tril(S_strict @ S_strict, k=-1)
  steps = int(math.ceil(math.log2(chunk_size)))
  for _ in range(steps - 1):
    A = jnp.tril(A + A @ E)
    E = jnp.tril(E @ E, k=-1)
  return A


@functools.partial(jax.named_call, name="invert_triangular_fwd")
def _invert_fwd(S):
  A = invert_unit_lower_triangular_log_depth(S)
  return A, A


@functools.partial(jax.named_call, name="invert_triangular_bwd")
def _invert_bwd(res, g):
  A = res
  return (jnp.tril(-(A.mT @ g @ A.mT), k=-1),)


invert_unit_lower_triangular_log_depth.defvjp(_invert_fwd, _invert_bwd)


# ------------------------------------- PR #4577 with explicit HIGHEST matmuls
# On TPU, float32 matmuls default to reduced precision. The log-depth method
# squares repeatedly (E = tril(E @ E)), so that error compounds -- five
# squarings at chunk 64, seven at chunk 256. `solve_triangular` has no such
# iteration, which is why the two diverge on TPU but agree on CPU. This variant
# is identical except that every matmul in the iteration asks for HIGHEST.
_HI = jax.lax.Precision.HIGHEST


@jax.custom_vjp
def invert_unit_lower_triangular_log_depth_hi(S):
  chunk_size = S.shape[-1]
  S_strict = jnp.tril(S, k=-1)
  identity = jnp.eye(chunk_size, dtype=S.dtype)
  A = identity - S_strict
  E = jnp.tril(jnp.matmul(S_strict, S_strict, precision=_HI), k=-1)
  steps = int(math.ceil(math.log2(chunk_size)))
  for _ in range(steps - 1):
    A = jnp.tril(A + jnp.matmul(A, E, precision=_HI))
    E = jnp.tril(jnp.matmul(E, E, precision=_HI), k=-1)
  return A


def _invert_hi_fwd(S):
  A = invert_unit_lower_triangular_log_depth_hi(S)
  return A, A


def _invert_hi_bwd(res, g):
  A = res
  return (jnp.tril(-jnp.matmul(jnp.matmul(A.mT, g, precision=_HI), A.mT,
                               precision=_HI), k=-1),)


invert_unit_lower_triangular_log_depth_hi.defvjp(_invert_hi_fwd, _invert_hi_bwd)


# ---------------------------------------------------------- main, unchanged
def jax_gdn_solve_triangular(
    query: Array,
    key: Array,
    value: Array,
    g: Array,
    beta: Array,
    chunk_size: int = 64,
    initial_state: None | Array = None,
    use_qk_norm_in_gdn: bool = False,
    cp_axis: None | str = None,
    compute_dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[Array, None | Array]:
  """Optimized JAX implementation of Gated Delta Rule."""
  # =========================================================================
  # STAGE 1: PREPARATION & PADDING
  # =========================================================================
  initial_dtype = query.dtype

  if use_qk_norm_in_gdn:
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)

  g = g.astype(jnp.float32)

  # 2. Cast inputs to the requested compute_dtype (cfg.dtype) to save memory/compute
  query = query.astype(compute_dtype)
  key = key.astype(compute_dtype)
  value = value.astype(compute_dtype)
  beta = beta.astype(compute_dtype)

  # Scale Query (keep in compute_dtype)
  scale = jax.lax.rsqrt(jnp.array(query.shape[-1], dtype=jnp.float32)).astype(compute_dtype)
  query = query * scale

  B, seq_len, H, K_dim = key.shape
  V_dim = value.shape[-1]

  pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size
  if pad_len > 0:

    def pad_fn(x, val=0.0):
      return jnp.pad(x, ((0, 0), (0, pad_len)) + ((0, 0),) * (x.ndim - 2), constant_values=val)

    query = pad_fn(query)
    key = pad_fn(key)
    value = pad_fn(value)
    g = pad_fn(g)
    beta = pad_fn(beta)

  num_chunks = query.shape[1] // chunk_size

  # Helper: (B, S, H, D) -> (B, N, H, C, D)
  def to_chunk(x):
    return x.reshape(B, num_chunks, chunk_size, H, -1).transpose(0, 1, 3, 2, 4)

  # Helper for scalars: (B, S, H) -> (B, N, H, C)
  def to_chunk_scalar(x):
    return x.reshape(B, num_chunks, chunk_size, H).transpose(0, 1, 3, 2)

  q_c = to_chunk(query)
  k_c = to_chunk(key)
  v_c = to_chunk(value)
  g_c = to_chunk_scalar(g)
  beta_c = to_chunk_scalar(beta)

  # =========================================================================
  # STAGE 2: INTRA-CHUNK PRE-COMPUTATION (Parallel)
  # =========================================================================

  # Cumulative decay (Must be float32)
  g_cumsum = jnp.cumsum(g_c, axis=-1)
  k_beta = k_c * beta_c[..., None]

  # S Matrix Calculation
  S = jnp.matmul(k_beta, k_c.swapaxes(-1, -2), precision=jax.lax.Precision.HIGHEST)
  S = S.astype(jnp.float32)

  # Apply mask BEFORE exp to prevent 'inf' gradients
  g_diff = g_cumsum[..., :, None] - g_cumsum[..., None, :]
  mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=-1)
  g_diff = jnp.where(mask, g_diff, -1e30)

  S = S * jnp.exp(g_diff)
  S = jnp.where(mask, S, 0.0)

  # Inversion (A) - Strictly float32
  identity = jnp.eye(chunk_size, dtype=jnp.float32)
  identity_broadcasted = jnp.broadcast_to(identity, S.shape)

  A = jax.scipy.linalg.solve_triangular(identity + S, identity_broadcasted, lower=True, unit_diagonal=True)

  # 5. WY Factors
  v_beta = v_c * beta_c[..., None]
  u_chunks = jnp.matmul(A, v_beta.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
  u_chunks = u_chunks.astype(compute_dtype)

  k_beta_g = k_beta.astype(jnp.float32) * jnp.exp(g_cumsum)[..., None]
  w_chunks = jnp.matmul(A, k_beta_g, precision=jax.lax.Precision.HIGHEST)
  w_chunks = w_chunks.astype(compute_dtype)

  # =========================================================================
  # STAGE 3: INTER-CHUNK RECURRENCE (Scan)
  # =========================================================================
  scan_perm_vec = (1, 0, 2, 3, 4)
  scan_perm_scl = (1, 0, 2, 3)

  w_scan = w_chunks.transpose(scan_perm_vec)
  u_scan = u_chunks.transpose(scan_perm_vec)
  k_scan = k_c.transpose(scan_perm_vec)
  q_scan = q_c.transpose(scan_perm_vec)
  g_scan = g_cumsum.transpose(scan_perm_scl)

  if initial_state is None:
    h_init = jnp.zeros((B, H, K_dim, V_dim), dtype=jnp.float32)
  else:
    h_init = initial_state.astype(jnp.float32)

  xs = (w_scan, u_scan, q_scan, k_scan, g_scan)

  def scan_body(h, args):
    w, u, q, k, g = args
    prec = jax.lax.Precision.HIGHEST

    # --- Output Computation ---
    # 1. Inter-chunk: q(dtype) * exp(g)(f32) -> f32
    q_g = q.astype(jnp.float32) * jnp.exp(g)[..., None]
    attn_inter = jnp.matmul(q_g, h, precision=prec)

    # 2. Delta Rule Subtraction (v_prime and v_new)
    # w serves as k_cumdecay, u serves as value_intra
    v_prime = jnp.matmul(w.astype(jnp.float32), h, precision=prec)
    v_new = u.astype(jnp.float32) - v_prime

    # 3. Intra-chunk: q(dtype) @ k(dtype) -> f32
    attn = jnp.matmul(q, k.swapaxes(-1, -2), precision=prec)
    attn = attn.astype(jnp.float32)

    # Mask before exp
    g_diff = g[..., :, None] - g[..., None, :]
    mask_intra = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool))
    g_diff = jnp.where(mask_intra, g_diff, -1e30)

    attn_i = attn * jnp.exp(g_diff)
    attn_i = jnp.where(mask_intra, attn_i, 0.0)

    # Note: We do NOT multiply attn_i by beta here. The Delta rule mathematically
    # absorbed beta inside v_new (via u).

    # 4. Combine Core Output
    term2 = jnp.matmul(attn_i, v_new, precision=prec)
    o_c = attn_inter + term2

    # --- State Update ---
    g_i_last_exp = jnp.exp(g[..., -1, None, None])
    h_new = h * g_i_last_exp

    # Apply Delta Rule K decay to state
    g_diff_exp_state = jnp.exp(g[..., -1, None] - g)[..., None]
    k_i_g_diff = k.astype(jnp.float32) * g_diff_exp_state

    update_term = jnp.matmul(k_i_g_diff.swapaxes(-1, -2), v_new, precision=prec)
    h_new = h_new + update_term

    return h_new, o_c

  if cp_axis is None:
    final_h, o_chunks = lax.scan(scan_body, h_init, xs)
  else:
    # Sequence is sharded over cp_axis, so a sequential scan over chunks is not
    # available. Fold the local chunks into one affine map, exchange those, then
    # replay locally from the correct incoming state. See kernels/attention/gdn_cp.py.
    A_loc, B_loc = gdn_cp.compose_local(w_scan, u_scan, k_scan, g_scan)
    h_in, final_h = gdn_cp.incoming_state(A_loc, B_loc, h_init, cp_axis)
    _, o_chunks = lax.scan(scan_body, h_in, xs)

  # =========================================================================
  # STAGE 4: FINALIZATION
  # =========================================================================
  o = o_chunks.transpose(1, 0, 3, 2, 4)
  o = o.reshape(B, -1, H, V_dim)

  if pad_len > 0:
    o = o[:, :seq_len, :, :]

  o = o.astype(initial_dtype)

  return o, (final_h if initial_state is not None else None)



# ------------------------------------------- main + PR #4577 applied
def jax_gdn_log_depth(
    query: Array,
    key: Array,
    value: Array,
    g: Array,
    beta: Array,
    chunk_size: int = 64,
    initial_state: None | Array = None,
    use_qk_norm_in_gdn: bool = False,
    cp_axis: None | str = None,
    compute_dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[Array, None | Array]:
  """Optimized JAX implementation of Gated Delta Rule."""
  # =========================================================================
  # STAGE 1: PREPARATION & PADDING
  # =========================================================================
  initial_dtype = query.dtype

  if use_qk_norm_in_gdn:
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)

  g = g.astype(jnp.float32)

  # 2. Cast inputs to the requested compute_dtype (cfg.dtype) to save memory/compute
  query = query.astype(compute_dtype)
  key = key.astype(compute_dtype)
  value = value.astype(compute_dtype)
  beta = beta.astype(compute_dtype)

  # Scale Query (keep in compute_dtype)
  scale = jax.lax.rsqrt(jnp.array(query.shape[-1], dtype=jnp.float32)).astype(compute_dtype)
  query = query * scale

  B, seq_len, H, K_dim = key.shape
  V_dim = value.shape[-1]

  pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size
  if pad_len > 0:

    def pad_fn(x, val=0.0):
      return jnp.pad(x, ((0, 0), (0, pad_len)) + ((0, 0),) * (x.ndim - 2), constant_values=val)

    query = pad_fn(query)
    key = pad_fn(key)
    value = pad_fn(value)
    g = pad_fn(g)
    beta = pad_fn(beta)

  num_chunks = query.shape[1] // chunk_size

  # Helper: (B, S, H, D) -> (B, N, H, C, D)
  def to_chunk(x):
    return x.reshape(B, num_chunks, chunk_size, H, -1).transpose(0, 1, 3, 2, 4)

  # Helper for scalars: (B, S, H) -> (B, N, H, C)
  def to_chunk_scalar(x):
    return x.reshape(B, num_chunks, chunk_size, H).transpose(0, 1, 3, 2)

  q_c = to_chunk(query)
  k_c = to_chunk(key)
  v_c = to_chunk(value)
  g_c = to_chunk_scalar(g)
  beta_c = to_chunk_scalar(beta)

  # =========================================================================
  # STAGE 2: INTRA-CHUNK PRE-COMPUTATION (Parallel)
  # =========================================================================

  # Cumulative decay (Must be float32)
  g_cumsum = jnp.cumsum(g_c, axis=-1)
  k_beta = k_c * beta_c[..., None]

  # S Matrix Calculation
  S = jnp.matmul(k_beta, k_c.swapaxes(-1, -2), precision=jax.lax.Precision.HIGHEST)
  S = S.astype(jnp.float32)

  # Apply mask BEFORE exp to prevent 'inf' gradients
  g_diff = g_cumsum[..., :, None] - g_cumsum[..., None, :]
  mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=-1)
  g_diff = jnp.where(mask, g_diff, -1e30)

  S = S * jnp.exp(g_diff)
  S = jnp.where(mask, S, 0.0)

  # Cast to float32 explicitly as you were doing before
  S = S.astype(jnp.float32)

  # Inversion (A) - Replaces solve_triangular entirely
  A = invert_unit_lower_triangular_log_depth(S)

  # 5. WY Factors
  v_beta = v_c * beta_c[..., None]
  u_chunks = jnp.matmul(A, v_beta.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
  u_chunks = u_chunks.astype(compute_dtype)

  k_beta_g = k_beta.astype(jnp.float32) * jnp.exp(g_cumsum)[..., None]
  w_chunks = jnp.matmul(A, k_beta_g, precision=jax.lax.Precision.HIGHEST)
  w_chunks = w_chunks.astype(compute_dtype)

  # =========================================================================
  # STAGE 3: INTER-CHUNK RECURRENCE (Scan)
  # =========================================================================
  scan_perm_vec = (1, 0, 2, 3, 4)
  scan_perm_scl = (1, 0, 2, 3)

  w_scan = w_chunks.transpose(scan_perm_vec)
  u_scan = u_chunks.transpose(scan_perm_vec)
  k_scan = k_c.transpose(scan_perm_vec)
  q_scan = q_c.transpose(scan_perm_vec)
  g_scan = g_cumsum.transpose(scan_perm_scl)

  if initial_state is None:
    h_init = jnp.zeros((B, H, K_dim, V_dim), dtype=jnp.float32)
  else:
    h_init = initial_state.astype(jnp.float32)

  xs = (w_scan, u_scan, q_scan, k_scan, g_scan)

  def scan_body(h, args):
    w, u, q, k, g = args
    prec = jax.lax.Precision.HIGHEST

    # --- Output Computation ---
    # 1. Inter-chunk: q(dtype) * exp(g)(f32) -> f32
    q_g = q.astype(jnp.float32) * jnp.exp(g)[..., None]
    attn_inter = jnp.matmul(q_g, h, precision=prec)

    # 2. Delta Rule Subtraction (v_prime and v_new)
    # w serves as k_cumdecay, u serves as value_intra
    v_prime = jnp.matmul(w.astype(jnp.float32), h, precision=prec)
    v_new = u.astype(jnp.float32) - v_prime

    # 3. Intra-chunk: q(dtype) @ k(dtype) -> f32
    attn = jnp.matmul(q, k.swapaxes(-1, -2), precision=prec)
    attn = attn.astype(jnp.float32)

    # Mask before exp
    g_diff = g[..., :, None] - g[..., None, :]
    mask_intra = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool))
    g_diff = jnp.where(mask_intra, g_diff, -1e30)

    attn_i = attn * jnp.exp(g_diff)
    attn_i = jnp.where(mask_intra, attn_i, 0.0)

    # Note: We do NOT multiply attn_i by beta here. The Delta rule mathematically
    # absorbed beta inside v_new (via u).

    # 4. Combine Core Output
    term2 = jnp.matmul(attn_i, v_new, precision=prec)
    o_c = attn_inter + term2

    # --- State Update ---
    g_i_last_exp = jnp.exp(g[..., -1, None, None])
    h_new = h * g_i_last_exp

    # Apply Delta Rule K decay to state
    g_diff_exp_state = jnp.exp(g[..., -1, None] - g)[..., None]
    k_i_g_diff = k.astype(jnp.float32) * g_diff_exp_state

    update_term = jnp.matmul(k_i_g_diff.swapaxes(-1, -2), v_new, precision=prec)
    h_new = h_new + update_term

    return h_new, o_c

  if cp_axis is None:
    final_h, o_chunks = lax.scan(scan_body, h_init, xs)
  else:
    # Sequence is sharded over cp_axis, so a sequential scan over chunks is not
    # available. Fold the local chunks into one affine map, exchange those, then
    # replay locally from the correct incoming state. See kernels/attention/gdn_cp.py.
    A_loc, B_loc = gdn_cp.compose_local(w_scan, u_scan, k_scan, g_scan)
    h_in, final_h = gdn_cp.incoming_state(A_loc, B_loc, h_init, cp_axis)
    _, o_chunks = lax.scan(scan_body, h_in, xs)

  # =========================================================================
  # STAGE 4: FINALIZATION
  # =========================================================================
  o = o_chunks.transpose(1, 0, 3, 2, 4)
  o = o.reshape(B, -1, H, V_dim)

  if pad_len > 0:
    o = o[:, :seq_len, :, :]

  o = o.astype(initial_dtype)

  return o, (final_h if initial_state is not None else None)



# ------------------- main + PR #4577, HIGHEST precision in the iteration
def jax_gdn_log_depth_hi(
    query: Array,
    key: Array,
    value: Array,
    g: Array,
    beta: Array,
    chunk_size: int = 64,
    initial_state: None | Array = None,
    use_qk_norm_in_gdn: bool = False,
    cp_axis: None | str = None,
    compute_dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[Array, None | Array]:
  """Optimized JAX implementation of Gated Delta Rule."""
  # =========================================================================
  # STAGE 1: PREPARATION & PADDING
  # =========================================================================
  initial_dtype = query.dtype

  if use_qk_norm_in_gdn:
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)

  g = g.astype(jnp.float32)

  # 2. Cast inputs to the requested compute_dtype (cfg.dtype) to save memory/compute
  query = query.astype(compute_dtype)
  key = key.astype(compute_dtype)
  value = value.astype(compute_dtype)
  beta = beta.astype(compute_dtype)

  # Scale Query (keep in compute_dtype)
  scale = jax.lax.rsqrt(jnp.array(query.shape[-1], dtype=jnp.float32)).astype(compute_dtype)
  query = query * scale

  B, seq_len, H, K_dim = key.shape
  V_dim = value.shape[-1]

  pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size
  if pad_len > 0:

    def pad_fn(x, val=0.0):
      return jnp.pad(x, ((0, 0), (0, pad_len)) + ((0, 0),) * (x.ndim - 2), constant_values=val)

    query = pad_fn(query)
    key = pad_fn(key)
    value = pad_fn(value)
    g = pad_fn(g)
    beta = pad_fn(beta)

  num_chunks = query.shape[1] // chunk_size

  # Helper: (B, S, H, D) -> (B, N, H, C, D)
  def to_chunk(x):
    return x.reshape(B, num_chunks, chunk_size, H, -1).transpose(0, 1, 3, 2, 4)

  # Helper for scalars: (B, S, H) -> (B, N, H, C)
  def to_chunk_scalar(x):
    return x.reshape(B, num_chunks, chunk_size, H).transpose(0, 1, 3, 2)

  q_c = to_chunk(query)
  k_c = to_chunk(key)
  v_c = to_chunk(value)
  g_c = to_chunk_scalar(g)
  beta_c = to_chunk_scalar(beta)

  # =========================================================================
  # STAGE 2: INTRA-CHUNK PRE-COMPUTATION (Parallel)
  # =========================================================================

  # Cumulative decay (Must be float32)
  g_cumsum = jnp.cumsum(g_c, axis=-1)
  k_beta = k_c * beta_c[..., None]

  # S Matrix Calculation
  S = jnp.matmul(k_beta, k_c.swapaxes(-1, -2), precision=jax.lax.Precision.HIGHEST)
  S = S.astype(jnp.float32)

  # Apply mask BEFORE exp to prevent 'inf' gradients
  g_diff = g_cumsum[..., :, None] - g_cumsum[..., None, :]
  mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=-1)
  g_diff = jnp.where(mask, g_diff, -1e30)

  S = S * jnp.exp(g_diff)
  S = jnp.where(mask, S, 0.0)

  # Cast to float32 explicitly as you were doing before
  S = S.astype(jnp.float32)

  # Inversion (A) - Replaces solve_triangular entirely
  A = invert_unit_lower_triangular_log_depth_hi(S)

  # 5. WY Factors
  v_beta = v_c * beta_c[..., None]
  u_chunks = jnp.matmul(A, v_beta.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
  u_chunks = u_chunks.astype(compute_dtype)

  k_beta_g = k_beta.astype(jnp.float32) * jnp.exp(g_cumsum)[..., None]
  w_chunks = jnp.matmul(A, k_beta_g, precision=jax.lax.Precision.HIGHEST)
  w_chunks = w_chunks.astype(compute_dtype)

  # =========================================================================
  # STAGE 3: INTER-CHUNK RECURRENCE (Scan)
  # =========================================================================
  scan_perm_vec = (1, 0, 2, 3, 4)
  scan_perm_scl = (1, 0, 2, 3)

  w_scan = w_chunks.transpose(scan_perm_vec)
  u_scan = u_chunks.transpose(scan_perm_vec)
  k_scan = k_c.transpose(scan_perm_vec)
  q_scan = q_c.transpose(scan_perm_vec)
  g_scan = g_cumsum.transpose(scan_perm_scl)

  if initial_state is None:
    h_init = jnp.zeros((B, H, K_dim, V_dim), dtype=jnp.float32)
  else:
    h_init = initial_state.astype(jnp.float32)

  xs = (w_scan, u_scan, q_scan, k_scan, g_scan)

  def scan_body(h, args):
    w, u, q, k, g = args
    prec = jax.lax.Precision.HIGHEST

    # --- Output Computation ---
    # 1. Inter-chunk: q(dtype) * exp(g)(f32) -> f32
    q_g = q.astype(jnp.float32) * jnp.exp(g)[..., None]
    attn_inter = jnp.matmul(q_g, h, precision=prec)

    # 2. Delta Rule Subtraction (v_prime and v_new)
    # w serves as k_cumdecay, u serves as value_intra
    v_prime = jnp.matmul(w.astype(jnp.float32), h, precision=prec)
    v_new = u.astype(jnp.float32) - v_prime

    # 3. Intra-chunk: q(dtype) @ k(dtype) -> f32
    attn = jnp.matmul(q, k.swapaxes(-1, -2), precision=prec)
    attn = attn.astype(jnp.float32)

    # Mask before exp
    g_diff = g[..., :, None] - g[..., None, :]
    mask_intra = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool))
    g_diff = jnp.where(mask_intra, g_diff, -1e30)

    attn_i = attn * jnp.exp(g_diff)
    attn_i = jnp.where(mask_intra, attn_i, 0.0)

    # Note: We do NOT multiply attn_i by beta here. The Delta rule mathematically
    # absorbed beta inside v_new (via u).

    # 4. Combine Core Output
    term2 = jnp.matmul(attn_i, v_new, precision=prec)
    o_c = attn_inter + term2

    # --- State Update ---
    g_i_last_exp = jnp.exp(g[..., -1, None, None])
    h_new = h * g_i_last_exp

    # Apply Delta Rule K decay to state
    g_diff_exp_state = jnp.exp(g[..., -1, None] - g)[..., None]
    k_i_g_diff = k.astype(jnp.float32) * g_diff_exp_state

    update_term = jnp.matmul(k_i_g_diff.swapaxes(-1, -2), v_new, precision=prec)
    h_new = h_new + update_term

    return h_new, o_c

  if cp_axis is None:
    final_h, o_chunks = lax.scan(scan_body, h_init, xs)
  else:
    # Sequence is sharded over cp_axis, so a sequential scan over chunks is not
    # available. Fold the local chunks into one affine map, exchange those, then
    # replay locally from the correct incoming state. See kernels/attention/gdn_cp.py.
    A_loc, B_loc = gdn_cp.compose_local(w_scan, u_scan, k_scan, g_scan)
    h_in, final_h = gdn_cp.incoming_state(A_loc, B_loc, h_init, cp_axis)
    _, o_chunks = lax.scan(scan_body, h_in, xs)

  # =========================================================================
  # STAGE 4: FINALIZATION
  # =========================================================================
  o = o_chunks.transpose(1, 0, 3, 2, 4)
  o = o.reshape(B, -1, H, V_dim)

  if pad_len > 0:
    o = o[:, :seq_len, :, :]

  o = o.astype(initial_dtype)

  return o, (final_h if initial_state is not None else None)

