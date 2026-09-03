#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the new per_channel_gate=False path in the KDA Mosaic forward.

Four things have to hold, and only the fourth is the new behaviour:

  1. per_channel_gate=True, mild gate   -> unchanged, matches the XLA reference
  2. per_channel_gate=True, steep gate  -> still NaN (untouched code path)
  3. per_channel_gate=False, mild gate  -> matches the XLA reference
  4. per_channel_gate=False, steep gate -> FINITE and matches the reference

1 and 2 are the regression check: the KDA path must behave exactly as before.
3 proves the regrouping is algebraically right, not just NaN-free. 4 is the fix.

The reference is Tokamax's own XLA implementation, which never factors the
gate, so it is a valid oracle for both groupings.
"""

import sys

import jax
import jax.numpy as jnp

H, B, T, D = 4, 1, 512, 128
CHUNK = 64
LN2 = 0.6931471805599453

import os as _os
# These scripts live in <repo>/gdn_validation/. Put the repo root on the path
# so `import tokamax` resolves to the checkout directly, rather than through
# the editable install's finder -- which reports tokamax._src as a namespace
# package and then cannot find tokamax._src.jaxtyping.
_repo = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _os.path.isdir(_os.path.join(_repo, "tokamax")) and _repo not in sys.path:
  sys.path.insert(0, _repo)

api = None
_errs = []
for _p in ("tokamax._src.ops.experimental.kda.api",
           "tokamax.ops.experimental.kda.api"):
  try:
    api = __import__(_p, fromlist=["api"])
    break
  except ImportError as e:
    _errs.append(f'{_p}: {e}')
    continue
if api is None:
  sys.exit("KDA op not importable. Tried:\n  " + "\n  ".join(_errs))

import inspect
if "per_channel_gate" not in inspect.signature(api.kimi_delta_attention).parameters:
  sys.exit("This tokamax does not have per_channel_gate -- the patch did not "
           "land in the container's site-packages.")


def inputs(a_max):
  k = jax.random.split(jax.random.PRNGKey(0), 6)
  n = lambda i, s: jax.random.normal(k[i], s, jnp.float32)
  q = jax.nn.silu(n(0, (H, B, T, D)))
  key = jax.nn.silu(n(1, (H, B, T, D)))
  v = jax.nn.silu(n(2, (H, B, T, D)))
  beta = jax.nn.sigmoid(n(3, (H, B, T)))
  a_log = jnp.log(jax.random.uniform(k[4], (H, 1, 1), jnp.float32,
                                     minval=1e-9, maxval=a_max))
  g = -jnp.exp(a_log) * jax.nn.softplus(n(5, (H, B, T)) + 1.0)
  # Scalar gate broadcast across key channels -- identical values, which is
  # exactly the case per_channel_gate=False is for.
  gate = jnp.broadcast_to(g[..., None], g.shape + (D,)).astype(jnp.float32)
  init = jnp.zeros((B, 1, H, D, D), jnp.float32)
  return q, key, v, gate, beta, init, g


def call(q, key, v, gate, beta, init, impl, per_channel):
  out, _ = api.kimi_delta_attention(
      q, key, v, gate, beta,
      initial_state=init, output_final_state=True,
      use_gate_in_kernel=False, use_qk_l2norm=True,
      per_channel_gate=per_channel,
      implementation=impl)
  return out


def rel(a, b):
  a, b = a.astype(jnp.float32), b.astype(jnp.float32)
  return float(jnp.max(jnp.abs(a - b))) / (float(jnp.max(jnp.abs(b))) + 1e-12)


def main():
  print(f"device: {jax.devices()[0]}")
  print(f"shapes: H={H} B={B} T={T} D={D} chunk={CHUNK}\n")
  print(f"{'A_max':>7} {'per_channel':>12} {'avg log2/tok':>13}  "
        f"{'mosaic':>22}  {'vs xla reference':>18}")
  print("-" * 82)

  ok = True
  for a_max in (0.5, 16.0):
    q, key, v, gate, beta, init, g = inputs(a_max)
    cum = jnp.cumsum(g.reshape(H, B, T // CHUNK, CHUNK), axis=3)
    avg = float(-jnp.min(cum)) / LN2 / CHUNK

    ref = jax.block_until_ready(
        call(q, key, v, gate, beta, init, "xla", True))

    for per_channel in (True, False):
      try:
        out = jax.block_until_ready(
            call(q, key, v, gate, beta, init, "mosaic", per_channel))
        bad = int(jnp.sum(~jnp.isfinite(out.astype(jnp.float32))))
        if bad:
          verdict, err = f"NaN/Inf {100.0 * bad / out.size:5.1f}%", "-"
        else:
          verdict, err = "finite", f"{rel(out, ref):.2e}"
      except Exception as e:  # noqa: BLE001
        verdict, err = f"RAISED {type(e).__name__}", "-"
      print(f"{a_max:>7.1f} {str(per_channel):>12} {avg:>13.1f}  "
            f"{verdict:>22}  {err:>18}")

      # Expectations
      if not per_channel:
        if "finite" not in verdict:
          print(f"        ^^ FAIL: scalar path should never NaN")
          ok = False
        elif float(err) > 5e-2:
          print(f"        ^^ FAIL: scalar path disagrees with the reference")
          ok = False
    print()

  print("=" * 82)
  print("PASS: scalar path is finite and matches the reference at both gates."
        if ok else "FAIL: see the marked rows above.")
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()
