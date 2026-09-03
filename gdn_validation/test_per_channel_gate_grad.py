#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate per_channel_gate in BOTH directions of the KDA Mosaic kernel.

Forward was already confirmed on v5: the scalar path is finite at Qwen3.5's
gate where the per-channel path gives NaN in half its outputs, and it matches
Tokamax's XLA reference to ~1.8e-03.

This adds the backward. The defect there is in the diagonal-block term, which
re-centres on the block max:

    g_max = max over the block
    row_d = exp2(g_b - g_max)      # <= 0, safe
    col_d = exp2(g_max - g_b)      # >= 0, overflows on a steep gate

The off-diagonal row/col pair paths are already safe -- both of their exponents
are <= 0 by construction -- so only this one term needed changing. For a scalar
gate row_d[r] * col_d[t] is exactly exp2(g[r] - g[t]), so the difference is
formed once as a [BC, BC] matrix and folded into the cotangents before
contracting: no re-centring, nothing to overflow.

Four rows again, and the last one is the new result:

  per_channel=True,  mild  -> unchanged, matches the reference (regression)
  per_channel=False, mild  -> matches the reference (regrouping is correct)
  per_channel=True,  steep -> still NaN (untouched path)
  per_channel=False, steep -> FINITE and matches the reference (the fix)
"""

import sys

import jax
import jax.numpy as jnp

H, B, T, D = 4, 1, 512, 128
CHUNK = 64
LN2 = 0.6931471805599453
NAMES = ("d_query", "d_key", "d_value", "d_gate", "d_beta")

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

# Upstream restricts the KDA Mosaic kernel to TPU generation >= 6:
#   supported_on() -> device.platform == "tpu" and generation >= 6
# The branch this work started from had no such guard, which is why it ran on
# v5. Set KDA_BYPASS_DEVICE_CHECK=1 to test on older hardware anyway. Do that
# knowingly: the numerics here are hardware-independent, but performance
# numbers from an unsupported generation mean nothing.
if _os.environ.get("KDA_BYPASS_DEVICE_CHECK") == "1":
  import dataclasses as _dc
  for _name, _op in list(api.IMPLEMENTATIONS.items()):
    api.IMPLEMENTATIONS[_name] = _dc.replace(_op, bypass_device_check=True)
  print("NOTE: device check bypassed (KDA_BYPASS_DEVICE_CHECK=1)")


import inspect
if "per_channel_gate" not in inspect.signature(api.kimi_delta_attention).parameters:
  sys.exit("This tokamax has no per_channel_gate -- the patch did not land.")


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
  gate = jnp.broadcast_to(g[..., None], g.shape + (D,)).astype(jnp.float32)
  init = jnp.zeros((B, 1, H, D, D), jnp.float32)
  return (q, key, v, gate, beta), init, g


def make(init, impl, per_channel):
  def f(q, key, v, gate, beta):
    out, _ = api.kimi_delta_attention(
        q, key, v, gate, beta,
        initial_state=init, output_final_state=True,
        use_gate_in_kernel=False, use_qk_l2norm=True,
        per_channel_gate=per_channel, implementation=impl)
    return out
  return f


def rel(a, b):
  a, b = a.astype(jnp.float32), b.astype(jnp.float32)
  return float(jnp.max(jnp.abs(a - b))) / (float(jnp.max(jnp.abs(b))) + 1e-12)


def main():
  print(f"device: {jax.devices()[0]}")
  print(f"shapes: H={H} B={B} T={T} D={D} chunk={CHUNK}\n")

  ok = True
  for a_max in (0.5, 16.0):
    args, init, g = inputs(a_max)
    cum = jnp.cumsum(g.reshape(H, B, T // CHUNK, CHUNK), axis=3)
    avg = float(-jnp.min(cum)) / LN2 / CHUNK
    cot = jax.random.normal(jax.random.PRNGKey(9), (H, B, T, D), jnp.float32)
    argn = (0, 1, 2, 3, 4)
    loss = lambda f: (lambda *x: jnp.sum(f(*x).astype(jnp.float32) * cot))

    print(f"=== A_max={a_max}  (min g {float(jnp.min(g)):.2f} nats/token, "
          f"{avg:.1f} log2/token avg) ===")
    ref_f = jax.block_until_ready(make(init, "xla", True)(*args))
    ref_g = jax.block_until_ready(
        jax.grad(loss(make(init, "xla", True)), argnums=argn)(*args))

    for per_channel in (True, False):
      tag = f"  per_channel={str(per_channel):<5}"
      try:
        fn = make(init, "mosaic", per_channel)
        out = jax.block_until_ready(fn(*args))
        grads = jax.block_until_ready(jax.grad(loss(fn), argnums=argn)(*args))
      except Exception as e:  # noqa: BLE001
        print(f"{tag} RAISED {type(e).__name__}: {str(e).replace(chr(10), ' | ')[:400]}")
        if not per_channel:
          ok = False
        continue

      bits = []
      worst = 0.0
      bad_any = False
      for nm, gr, rf in zip(("out",) + NAMES, (out,) + grads, (ref_f,) + ref_g):
        bad = int(jnp.sum(~jnp.isfinite(gr.astype(jnp.float32))))
        if bad:
          bits.append(f"{nm}=NaN{100 * bad // gr.size}%")
          bad_any = True
        else:
          r = rel(gr, rf)
          worst = max(worst, r)
          bits.append(f"{nm}={r:.1e}")
      print(f"{tag} " + "  ".join(bits))

      if not per_channel:
        if bad_any:
          print("           ^^ FAIL: scalar path must never produce NaN")
          ok = False
        elif worst > 5e-2:
          print(f"           ^^ FAIL: disagrees with the reference ({worst:.2e})")
          ok = False
    print()

  print("=" * 78)
  print("PASS: scalar path is finite and matches the reference, forward and "
        "backward." if ok else "FAIL: see the marked rows above.")
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()
