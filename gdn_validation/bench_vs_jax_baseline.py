#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KDA against the pure-JAX Gated Delta Net, before and after MaxText PR #4577.

Earlier comparisons of this kernel were made against MaxText main's
`jax_chunk_gated_delta_rule`, which inverts (I+S) with
`jax.scipy.linalg.solve_triangular`. PR #4577 replaces that with a log-depth
Newton-Schulz iteration and reports forward 96.96 -> 23 ms, backward 127 -> 52
ms on v6e. Any speedup quoted against the old path is therefore overstated.

Four contenders, same inputs, same shapes:

  jax + solve_triangular   MaxText main today
  jax + log-depth          main with PR #4577 -- the baseline to beat
  KDA per_channel_gate=True    stock kernel behaviour
  KDA per_channel_gate=False   the scalar-gate path

The gate is mild (A_max=0.5) so every contender computes the same thing; at
Qwen3.5's real gate the stock KDA path returns NaN and timing it would be
meaningless.

Correctness is checked first and reported alongside, because a fast wrong
answer is worthless. Shapes come from one shard of the training config;
override with KDA_H / KDA_B / KDA_T / KDA_DTYPE / KDA_CHUNK / KDA_ITERS.
"""

import inspect
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
CHUNK = int(os.environ.get("KDA_CHUNK", "64"))
ITERS = int(os.environ.get("KDA_ITERS", "10"))
DTYPE = {"f32": jnp.float32, "bf16": jnp.bfloat16}[os.environ.get("KDA_DTYPE", "bf16")]

import os as _os
_repo = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _os.path.isdir(_os.path.join(_repo, "tokamax")) and _repo not in sys.path:
  sys.path.insert(0, _repo)
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import jax_gdn_reference as ref  # noqa: E402

api = None
_errs = []
for _p in ("tokamax._src.ops.experimental.kda.api",
           "tokamax.ops.experimental.kda.api"):
  try:
    api = __import__(_p, fromlist=["api"])
    break
  except ImportError as e:
    _errs.append(f"{_p}: {e}")
if api is None:
  sys.exit("KDA op not importable. Tried:\n  " + "\n  ".join(_errs))

if _os.environ.get("KDA_BYPASS_DEVICE_CHECK") == "1":
  import dataclasses as _dc
  for _n, _o in list(api.IMPLEMENTATIONS.items()):
    api.IMPLEMENTATIONS[_n] = _dc.replace(_o, bypass_device_check=True)
  print("NOTE: device check bypassed (KDA_BYPASS_DEVICE_CHECK=1)")


def make(a_max=0.5, gate_width=None):
  """MaxText layout for the JAX paths: [B, T, H, D]. KDA wants head-first."""
  k = jax.random.split(jax.random.PRNGKey(0), 6)
  n = lambda i, s: jax.random.normal(k[i], s, jnp.float32)
  q = jax.nn.silu(n(0, (B, T, H, D))).astype(DTYPE)
  key = jax.nn.silu(n(1, (B, T, H, D))).astype(DTYPE)
  v = jax.nn.silu(n(2, (B, T, H, D))).astype(DTYPE)
  beta = jax.nn.sigmoid(n(3, (B, T, H)))
  a_log = jnp.log(jax.random.uniform(k[4], (H,), jnp.float32,
                                     minval=1e-9, maxval=a_max))
  g = -jnp.exp(a_log) * jax.nn.softplus(n(5, (B, T, H)) + 1.0)
  init = jnp.zeros((B, H, D, D), jnp.float32)
  return q, key, v, g, beta, init, (gate_width or D)


def jax_path(fn):
  """MaxText's signature, returning just the output."""
  def run(q, key, v, g, beta, init):
    out, _ = fn(q, key, v, g, beta, chunk_size=CHUNK, initial_state=init,
                use_qk_norm_in_gdn=True, compute_dtype=DTYPE)
    return out
  return run


# Only the scalar path can take a chunk size; a per-channel gate is pinned to
# 64 upstream. Passing it for both would raise, so ask only when it is legal.
_KDA_TAKES_CHUNK = "chunk_size" in inspect.signature(
    api.kimi_delta_attention).parameters


def kda_path(per_channel, width):
  """Same maths through KDA: transpose to head-first, gate to `width`."""
  hf = lambda x: jnp.transpose(x, (2, 0, 1, 3))
  extra = {}
  if _KDA_TAKES_CHUNK and not per_channel:
    extra["chunk_size"] = CHUNK
  def run(q, key, v, g, beta, init):
    gh = jnp.transpose(g.astype(jnp.float32), (2, 0, 1))
    gate = jnp.broadcast_to(gh[..., None], gh.shape + (width,))
    out, _ = api.kimi_delta_attention(
        hf(q), hf(key), hf(v), gate,
        jnp.transpose(beta.astype(jnp.float32), (2, 0, 1)),
        initial_state=init.astype(jnp.float32)[:, None],
        output_final_state=True, use_gate_in_kernel=False,
        use_qk_l2norm=True, per_channel_gate=per_channel,
        implementation="mosaic", **extra)
    return jnp.transpose(out, (1, 2, 0, 3))     # back to [B, T, H, D]
  return run


def timed(fn, args):
  try:
    jax.block_until_ready(fn(*args))
  except Exception as e:  # noqa: BLE001
    return None, f"{type(e).__name__}: {str(e).replace(chr(10), ' | ')[:90]}"
  ts = []
  for _ in range(ITERS):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    ts.append((time.perf_counter() - t0) * 1e3)
  return statistics.median(ts), None


def rel(a, b):
  a, b = a.astype(jnp.float32), b.astype(jnp.float32)
  return float(jnp.max(jnp.abs(a - b))) / (float(jnp.max(jnp.abs(b))) + 1e-12)


def main():
  print(f"device: {jax.devices()[0]}")
  print(f"shapes: H={H} B={B} T={T} D={D} chunk={CHUNK} "
        f"dtype={DTYPE.__name__} iters={ITERS}\n")

  contenders = [
      ("jax + solve_triangular  (main)", jax_path(ref.jax_gdn_solve_triangular), D),
      ("jax + log-depth   (PR #4577)", jax_path(ref.jax_gdn_log_depth), D),
      ("jax + log-depth, HIGHEST", jax_path(ref.jax_gdn_log_depth_hi), D),
      ("KDA per_channel=True  (chunk 64)", kda_path(True, D), D),
      ("KDA per_channel=False (chunk %d)" % (CHUNK if _KDA_TAKES_CHUNK else 64), kda_path(False, D), D),
      ("KDA width-1 gate      (chunk %d)" % (CHUNK if _KDA_TAKES_CHUNK else 64), kda_path(False, 1), 1),
  ]

  q, key, v, g, beta, init, _ = make()
  args = (q, key, v, g, beta, init)
  cot = jax.random.normal(jax.random.PRNGKey(9), (B, T, H, D), jnp.float32)

  # PR #4577 is the reference for both correctness and speed.
  baseline_fn = jax_path(ref.jax_gdn_log_depth)
  base_out = jax.block_until_ready(baseline_fn(*args))

  print(f"{'contender':<32} {'vs PR#4577':>11} {'fwd':>10} {'fwd+bwd':>11} "
        f"{'fwd x':>7} {'total x':>8}")
  print("-" * 84)

  results = {}
  base_f = base_b = None
  for name, fn, _w in contenders:
    try:
      out = jax.block_until_ready(fn(*args))
      bad = int(jnp.sum(~jnp.isfinite(out.astype(jnp.float32))))
      acc = f"NaN {100.0 * bad / out.size:.0f}%" if bad else f"{rel(out, base_out):.1e}"
    except Exception as e:  # noqa: BLE001
      acc = type(e).__name__

    loss = lambda *x: jnp.sum(fn(*x).astype(jnp.float32) * cot)
    tf, ef = timed(jax.jit(fn), args)
    tb, eb = timed(jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4))), args)
    results[name] = (tf, tb)
    if name.startswith("jax + log-depth   (PR"):   # the PR row only
      base_f, base_b = tf, tb

    fs = f"{base_f / tf:.2f}x" if (tf and base_f) else "-"
    bs = f"{base_b / tb:.2f}x" if (tb and base_b) else "-"
    print(f"{name:<32} {acc:>11} "
          f"{(f'{tf:.2f} ms' if tf else (ef or '-')[:10]):>10} "
          f"{(f'{tb:.2f} ms' if tb else (eb or '-')[:11]):>11} {fs:>7} {bs:>8}")

  print("\n'fwd x' and 'total x' are speedups over PR #4577; above 1.00x means")
  print("faster than the optimised JAX baseline. 'vs PR#4577' is max relative")
  print("error against it, so it doubles as a correctness check.")


if __name__ == "__main__":
  main()
