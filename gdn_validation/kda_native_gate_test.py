#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the gate-range NaN also hit KDA used as KDA, not as GDN?

Everything measured so far used use_gate_in_kernel=False -- the GDN adapter
hands the kernel an already-activated log gate. KDA's native mode is a
different branch:

  * the kernel activates the gate itself from a_log and delta_time_bias
  * _resolve_safe_gate returns `not use_gate_in_kernel or lower_bound is not
    None`, so native mode with no lower_bound gives safe_gate=FALSE, which
    moves the reference row from BC//2=8 to 0. Untested by everything else.

And native mode has two gate regimes (reference.py:46):

    lower_bound is None ->  g = -A * softplus(g_f)          unbounded below
    lower_bound = L     ->  g = L * sigmoid(A * g_f)        bounded in [L, 0]

The bounded one cannot produce a steep gate no matter what A is, so it should
be immune. The unbounded one is the same formula GDN uses, so it should not be.
This checks both instead of assuming.
"""

import sys

import jax
import jax.numpy as jnp

H, B, T, D = 4, 1, 512, 128
CHUNK = 64
LN2 = 0.6931471805599453

api = None
_errs = []
for _p in ("tokamax._src.ops.experimental.kda.api",
           "tokamax.ops.experimental.kda.api",
           "tokamax.experimental.kda.api"):
  try:
    api = __import__(_p, fromlist=["api"])
    break
  except ImportError as e:
    _errs.append(f'{_p}: {e}')
    continue
if api is None:
  sys.exit("KDA op not importable. Tried:\n  " + "\n  ".join(_errs))


def inputs(a_max):
  k = jax.random.split(jax.random.PRNGKey(0), 6)
  n = lambda i, s: jax.random.normal(k[i], s, jnp.float32)
  q = jax.nn.silu(n(0, (H, B, T, D)))
  key = jax.nn.silu(n(1, (H, B, T, D)))
  v = jax.nn.silu(n(2, (H, B, T, D)))
  beta = jax.nn.sigmoid(n(3, (H, B, T)))
  raw_gate = n(5, (H, B, T, D))                      # pre-activation
  a_log = jnp.log(jax.random.uniform(k[4], (H,), jnp.float32,
                                     minval=1e-9, maxval=a_max))
  dt_bias = jnp.ones((H * D,), jnp.float32)
  init = jnp.zeros((B, 1, H, D, D), jnp.float32)
  return q, key, v, raw_gate, beta, a_log, dt_bias, init


def activated(raw_gate, a_log, dt_bias, lower_bound):
  """Mirror reference.py:_activate_gate so we can report the real magnitude."""
  g_f = raw_gate + dt_bias.reshape(H, 1, 1, D)
  A = jnp.exp(a_log).reshape(H, 1, 1, 1)
  if lower_bound is None:
    return -A * jax.nn.softplus(g_f)
  return lower_bound * jax.nn.sigmoid(A * g_f)


def verdict(out):
  bad = int(jnp.sum(~jnp.isfinite(out.astype(jnp.float32))))
  return (f"NaN/Inf {100.0 * bad / out.size:5.1f}%" if bad
          else f"finite ({float(jnp.max(jnp.abs(out.astype(jnp.float32)))):.3e})")


def run(a_max, lower_bound):
  q, key, v, raw_gate, beta, a_log, dt_bias, init = inputs(a_max)
  g = activated(raw_gate, a_log, dt_bias, lower_bound)
  cum = jnp.cumsum(g.reshape(H, B, T // CHUNK, CHUNK, D), axis=3)
  drop = float(-jnp.min(cum)) / LN2

  row = {}
  for impl in ("mosaic", "xla"):
    try:
      out, _ = api.kimi_delta_attention(
          q, key, v, raw_gate, beta,
          a_log=a_log, delta_time_bias=dt_bias,
          initial_state=init, output_final_state=True,
          use_gate_in_kernel=True,          # <-- KDA as KDA
          use_qk_l2norm=True,
          lower_bound=lower_bound,
          implementation=impl)
      row[impl] = verdict(jax.block_until_ready(out))
    except Exception as e:  # noqa: BLE001
      row[impl] = f"RAISED {type(e).__name__}"

  lb = "None" if lower_bound is None else f"{lower_bound}"
  print(f"{a_max:>7.1f} {lb:>12} {float(jnp.min(g)):>12.2f} {drop / 64:>11.1f} "
        f" {row['mosaic']:>22}  {row['xla']:>22}")
  return "NaN" in row["mosaic"]


def main():
  print(f"device: {jax.devices()[0]}")
  print(f"shapes: H={H} B={B} T={T} D={D}   use_gate_in_kernel=True\n")
  print("safe_gate resolves to FALSE when lower_bound is None (ref row 0, not 8).\n")
  print(f"{'A_max':>7} {'lower_bound':>12} {'min g/token':>12} {'avg log2/tok':>11}"
        f"  {'mosaic':>22}  {'xla reference':>22}")
  print("-" * 95)

  unbounded = [run(a, None) for a in (0.5, 8.0, 16.0)]
  print()
  bounded = [run(a, -1.0) for a in (0.5, 16.0, 64.0)]

  print("\n" + "=" * 95)
  print(f"unbounded (lower_bound=None): {'FAILS' if any(unbounded) else 'clean'}")
  print(f"bounded   (lower_bound=-1.0): {'FAILS' if any(bounded) else 'clean'}")


if __name__ == "__main__":
  main()
