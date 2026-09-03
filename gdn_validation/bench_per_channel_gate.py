#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure the scalar-gate path against the per-channel path.

So far the claim that `per_channel_gate=False` is cheaper rests on an op count,
not a measurement: same matmul FLOPs, roughly 10x fewer exp2, no per-sub-block
[BT, K] temporaries. This times it.

The gate is deliberately MILD (A_max=0.5). A steep gate makes the per-channel
path return NaN, and timing a kernel that is producing garbage tells you
nothing. Both paths do the same work here.

Shapes default to one shard of the real training config -- H=32, B=4, T=4096,
D=128, which is what MaxText's shard_map hands the kernel per device. Override
with env vars if that does not fit:

    KDA_H=8 KDA_B=1 KDA_T=1024 python3 bench_per_channel_gate.py
"""

import os
import statistics
import sys
import time

import jax
import jax.numpy as jnp

H = int(os.environ.get("KDA_H", "32"))
B = int(os.environ.get("KDA_B", "4"))
T = int(os.environ.get("KDA_T", "4096"))
D = int(os.environ.get("KDA_D", "128"))
CHUNK = 64
ITERS = int(os.environ.get("KDA_ITERS", "10"))
DTYPE = {"f32": jnp.float32, "bf16": jnp.bfloat16}[os.environ.get("KDA_DTYPE", "bf16")]

api = None
for _p in ("tokamax._src.ops.experimental.kda.api",
           "tokamax.ops.experimental.kda.api"):
  try:
    api = __import__(_p, fromlist=["api"])
    break
  except ImportError:
    continue
if api is None:
  sys.exit("KDA op not importable.")

import inspect
_HAS_FLAG = "per_channel_gate" in inspect.signature(
    api.kimi_delta_attention).parameters
if not _HAS_FLAG:
  sys.exit("This tokamax has no per_channel_gate -- patch it first.")

# After the gate-shape change the scalar path takes [H, B, T, 1]. Detect which
# contract this build uses so the same script measures both.
_SCALAR_GATE_IS_NARROW = os.environ.get("KDA_NARROW_GATE", "auto")


def inputs(per_channel, a_max=0.5):
  k = jax.random.split(jax.random.PRNGKey(0), 6)
  n = lambda i, s: jax.random.normal(k[i], s, jnp.float32)
  q = jax.nn.silu(n(0, (H, B, T, D))).astype(DTYPE)
  key = jax.nn.silu(n(1, (H, B, T, D))).astype(DTYPE)
  v = jax.nn.silu(n(2, (H, B, T, D))).astype(DTYPE)
  beta = jax.nn.sigmoid(n(3, (H, B, T)))
  a_log = jnp.log(jax.random.uniform(k[4], (H, 1, 1), jnp.float32,
                                     minval=1e-9, maxval=a_max))
  g = -jnp.exp(a_log) * jax.nn.softplus(n(5, (H, B, T)) + 1.0)
  width = 1 if (not per_channel and _narrow()) else D
  gate = jnp.broadcast_to(g[..., None], g.shape + (width,)).astype(jnp.float32)
  init = jnp.zeros((B, 1, H, D, D), jnp.float32)
  return q, key, v, gate, beta, init


def _narrow():
  if _SCALAR_GATE_IS_NARROW in ("1", "true", "yes"):
    return True
  if _SCALAR_GATE_IS_NARROW in ("0", "false", "no"):
    return False
  # auto: try a tiny call with a width-1 gate and see if it is accepted
  if not hasattr(_narrow, "_cached"):
    try:
      h, b, t, d = 1, 1, CHUNK, D
      api.kimi_delta_attention(
          jnp.zeros((h, b, t, d), jnp.float32), jnp.zeros((h, b, t, d), jnp.float32),
          jnp.zeros((h, b, t, d), jnp.float32), jnp.zeros((h, b, t, 1), jnp.float32),
          jnp.zeros((h, b, t), jnp.float32),
          initial_state=jnp.zeros((b, 1, h, d, d), jnp.float32),
          output_final_state=True, use_gate_in_kernel=False,
          use_qk_l2norm=True, per_channel_gate=False, implementation="mosaic")
      _narrow._cached = True
    except Exception:
      _narrow._cached = False
  return _narrow._cached


def timed(fn, args, label):
  """Median wall time over ITERS, after a warmup that absorbs compilation."""
  try:
    jax.block_until_ready(fn(*args))
  except Exception as e:  # noqa: BLE001
    return None, f"{type(e).__name__}: {str(e)[:60]}"
  ts = []
  for _ in range(ITERS):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    ts.append((time.perf_counter() - t0) * 1e3)
  return statistics.median(ts), None


def main():
  print(f"device: {jax.devices()[0]}")
  print(f"shapes: H={H} B={B} T={T} D={D} chunk={CHUNK} dtype={DTYPE.__name__}")
  print(f"iters:  {ITERS} (median reported)   gate: mild, A_max=0.5")
  print(f"scalar gate width: {'1 (narrow)' if _narrow() else str(D) + ' (broadcast)'}\n")

  results = {}
  for per_channel in (True, False):
    q, key, v, gate, beta, init = inputs(per_channel)
    gate_mb = gate.size * 4 / 2**20

    def fwd(q, key, v, gate, beta):
      out, _ = api.kimi_delta_attention(
          q, key, v, gate, beta, initial_state=init, output_final_state=True,
          use_gate_in_kernel=False, use_qk_l2norm=True,
          per_channel_gate=per_channel, implementation="mosaic")
      return out

    cot = jax.random.normal(jax.random.PRNGKey(9), (H, B, T, D), jnp.float32)
    loss = lambda *x: jnp.sum(fwd(*x).astype(jnp.float32) * cot)
    grad = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))
    fwd_j = jax.jit(fwd)

    args = (q, key, v, gate, beta)
    t_f, err_f = timed(fwd_j, args, "fwd")
    t_b, err_b = timed(grad, args, "fwd+bwd")
    results[per_channel] = (t_f, t_b, gate_mb)
    tag = "per_channel" if per_channel else "scalar     "
    print(f"  {tag}  gate tensor {gate_mb:7.1f} MB   "
          f"fwd {('%8.2f ms' % t_f) if t_f else err_f}   "
          f"fwd+bwd {('%8.2f ms' % t_b) if t_b else err_b}")

  tf_p, tb_p, mb_p = results[True]
  tf_s, tb_s, mb_s = results[False]
  print()
  if tf_p and tf_s:
    print(f"  forward   speedup {tf_p / tf_s:5.2f}x   ({tf_p:.2f} -> {tf_s:.2f} ms)")
  if tb_p and tb_s:
    print(f"  fwd+bwd   speedup {tb_p / tb_s:5.2f}x   ({tb_p:.2f} -> {tb_s:.2f} ms)")
    if tf_p and tf_s:
      print(f"  backward  speedup {(tb_p - tf_p) / (tb_s - tf_s):5.2f}x  "
            f"(by subtraction, so noisier than the two above)")
  print(f"  gate tensor {mb_p:.1f} MB -> {mb_s:.1f} MB  ({mb_p / mb_s:.0f}x smaller)"
        if mb_s < mb_p else
        f"  gate tensor unchanged at {mb_p:.1f} MB (shape change not applied)")


if __name__ == "__main__":
  main()
