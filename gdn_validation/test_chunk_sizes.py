#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the scalar-gate path give the same answer at every chunk size?

MaxText's production config asks for `gdn_chunk_size=128`. Stock KDA is pinned
to 64, so every speed comparison so far forced the JAX path down to 64 to match
-- which is not comparing like with like. The scalar-gate path now takes the
chunk size from the caller.

Chunking must not change the result: the recurrence is identical however it is
split. So every chunk size here must agree with the XLA reference, and with
each other. Anything else is a bug, not a rounding difference.

Two things are checked:

  1. scalar gate at 32/64/128/256 -- all must match the reference
  2. per-channel gate at 128      -- must still be REJECTED, because Kimi Delta
                                     Attention is only validated at 64 and this
                                     change was not meant to alter it
"""

import os as _os
import sys

import jax
import jax.numpy as jnp

H, B, T, D = 4, 1, 1024, 128

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
    _errs.append(f"{_p}: {e}")
if api is None:
  sys.exit("KDA op not importable. Tried:\n  " + "\n  ".join(_errs))

if _os.environ.get("KDA_BYPASS_DEVICE_CHECK") == "1":
  import dataclasses as _dc
  for _n, _o in list(api.IMPLEMENTATIONS.items()):
    api.IMPLEMENTATIONS[_n] = _dc.replace(_o, bypass_device_check=True)
  print("NOTE: device check bypassed (KDA_BYPASS_DEVICE_CHECK=1)")

import inspect
if "chunk_size" not in inspect.signature(api.kimi_delta_attention).parameters:
  sys.exit("This tokamax has no chunk_size parameter -- patch did not land.")


def inputs(a_max=16.0, width=1):
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
  return q, key, v, gate, beta, jnp.zeros((B, 1, H, D, D), jnp.float32)


def run(impl, per_channel, chunk, width=1):
  q, key, v, gate, beta, init = inputs(width=width)
  kw = {} if chunk is None else {"chunk_size": chunk}
  out, _ = api.kimi_delta_attention(
      q, key, v, gate, beta, initial_state=init, output_final_state=True,
      use_gate_in_kernel=False, use_qk_l2norm=True,
      per_channel_gate=per_channel, implementation=impl, **kw)
  return out


def rel(a, b):
  a, b = a.astype(jnp.float32), b.astype(jnp.float32)
  return float(jnp.max(jnp.abs(a - b))) / (float(jnp.max(jnp.abs(b))) + 1e-12)


def main():
  print(f"device: {jax.devices()[0]}")
  print(f"shapes: H={H} B={B} T={T} D={D}   gate: A_max=16 (Qwen3.5's own)\n")
  ok = True

  ref = jax.block_until_ready(run("xla", False, None))

  print("scalar gate, width 1 -- chunking must not change the answer")
  print(f"{'chunk':>7}  {'mosaic':>22}  {'vs XLA reference':>18}")
  print("-" * 54)
  outs = {}
  for chunk in (32, 64, 128, 256):
    try:
      out = jax.block_until_ready(run("mosaic", False, chunk))
      bad = int(jnp.sum(~jnp.isfinite(out.astype(jnp.float32))))
      if bad:
        print(f"{chunk:>7}  {'NaN/Inf %.0f%%' % (100.0*bad/out.size):>22}  {'-':>18}")
        ok = False
        continue
      r = rel(out, ref)
      outs[chunk] = out
      print(f"{chunk:>7}  {'finite':>22}  {r:>18.2e}")
      if r > 5e-2:
        print(f"         ^^ FAIL: disagrees with the reference")
        ok = False
    except Exception as e:  # noqa: BLE001
      msg = str(e).replace(chr(10), " | ")[:70]
      print(f"{chunk:>7}  {'RAISED ' + type(e).__name__:>22}  {msg}")
      ok = False

  if 64 in outs:
    print("\nchunk sizes against each other (64 as the anchor)")
    for c, o in sorted(outs.items()):
      if c != 64:
        r = rel(o, outs[64])
        flag = "" if r < 5e-2 else "   <- FAIL"
        print(f"   {c:>4} vs 64: {r:.2e}{flag}")
        if r >= 5e-2:
          ok = False

  print("\nper-channel gate at 128 must still be rejected")
  try:
    run("mosaic", True, 128, width=D)
    print("   FAIL: it was accepted -- KDA should stay pinned to 64")
    ok = False
  except NotImplementedError as e:
    print(f"   OK: {str(e)[:72]}")
  except Exception as e:  # noqa: BLE001
    print(f"   rejected with {type(e).__name__}: {str(e)[:60]}")

  print("\n" + "=" * 54)
  print("PASS" if ok else "FAIL: see the marked rows above.")
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()
