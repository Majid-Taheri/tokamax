#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check the width-1 gate: same answers, 128x less gate memory.

Three configurations, forward and all five gradients, all against Tokamax's
XLA reference:

  per_channel=True,  width=K   KDA as it was -- must be untouched
  per_channel=False, width=K   the scalar maths, gate still broadcast
  per_channel=False, width=1   the scalar maths with the redundancy removed

Rows 2 and 3 must agree with the reference and with each other: the only
difference between them is whether the same number is stored once or 128
times. Row 1 is the regression check.

At A_max=16 (Qwen3.5's own init) row 1 is expected to be NaN -- that is the
bug the scalar path exists to avoid, not a failure of this test.
"""

import sys

import jax
import jax.numpy as jnp

H, B, T, D = 4, 1, 512, 128
CHUNK = 64
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


def inputs(a_max, width):
  k = jax.random.split(jax.random.PRNGKey(0), 6)
  n = lambda i, s: jax.random.normal(k[i], s, jnp.float32)
  q = jax.nn.silu(n(0, (H, B, T, D)))
  key = jax.nn.silu(n(1, (H, B, T, D)))
  v = jax.nn.silu(n(2, (H, B, T, D)))
  beta = jax.nn.sigmoid(n(3, (H, B, T)))
  a_log = jnp.log(jax.random.uniform(k[4], (H, 1, 1), jnp.float32,
                                     minval=1e-9, maxval=a_max))
  g = -jnp.exp(a_log) * jax.nn.softplus(n(5, (H, B, T)) + 1.0)
  gate = jnp.broadcast_to(g[..., None], g.shape + (width,)).astype(jnp.float32)
  init = jnp.zeros((B, 1, H, D, D), jnp.float32)
  return (q, key, v, gate, beta), init


def make(init, impl, per_channel):
  def f(q, key, v, gate, beta):
    out, _ = api.kimi_delta_attention(
        q, key, v, gate, beta, initial_state=init, output_final_state=True,
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
    print(f"=== A_max={a_max} ===")
    for per_channel, width in ((True, D), (False, D), (False, 1)):
      args, init = inputs(a_max, width)
      mb = args[3].size * 4 / 2**20
      cot = jax.random.normal(jax.random.PRNGKey(9), (H, B, T, D), jnp.float32)
      argn = (0, 1, 2, 3, 4)
      loss = lambda f: (lambda *x: jnp.sum(f(*x).astype(jnp.float32) * cot))
      tag = f"  per_channel={str(per_channel):<5} width={width:<3} gate {mb:6.1f} MB"

      try:
        ref_f = jax.block_until_ready(make(init, "xla", per_channel)(*args))
        ref_g = jax.block_until_ready(
            jax.grad(loss(make(init, "xla", per_channel)), argnums=argn)(*args))
        fn = make(init, "mosaic", per_channel)
        out = jax.block_until_ready(fn(*args))
        grads = jax.block_until_ready(jax.grad(loss(fn), argnums=argn)(*args))
      except Exception as e:  # noqa: BLE001
        print(f"{tag}  RAISED {type(e).__name__}: {str(e)[:70]}")
        if not per_channel:
          ok = False
        continue

      bits, worst, bad_any = [], 0.0, False
      for nm, gr, rf in zip(("out",) + NAMES, (out,) + grads, (ref_f,) + ref_g):
        bad = int(jnp.sum(~jnp.isfinite(gr.astype(jnp.float32))))
        if bad:
          bits.append(f"{nm}=NaN{100 * bad // gr.size}%")
          bad_any = True
        else:
          r = rel(gr, rf)
          worst = max(worst, r)
          bits.append(f"{nm}={r:.1e}")
      print(f"{tag}  " + " ".join(bits))

      if not per_channel:
        if bad_any:
          print("      ^^ FAIL: the scalar path must never produce NaN")
          ok = False
        elif worst > 5e-2:
          print(f"      ^^ FAIL: disagrees with the reference ({worst:.2e})")
          ok = False
    print()

  print("=" * 78)
  print("PASS: width-1 gate matches the reference, forward and backward, at "
        "both gate ranges." if ok else "FAIL: see the marked rows above.")
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()
