#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does `batch_first=True` give the same gradients as the head-first path?

The backward kernel has always demanded `[H, B, T, D]`. The model produces
`[B, T, H, D]`. So the wrapper transposes every large tensor in, and every
gradient back out. XProf puts that at 575.76 ms per step against the competing
kernel's 348.95, and it is pure data movement.

`batch_first=True` reads the caller's layout directly through the BlockSpec.
The tile arrives in VMEM as `[BT, MB, D]` and the body wants `[MB, BT, D]`, so
there is one on-chip transpose per operand. That is a VMEM relayout, not free,
but it replaces a full HBM round trip per tensor per layer.

Only the six large tensors move: q, k, v, v_new, do, dv0 in, and dq, dk, dv
out. The gate, beta, A, h and the state stay head-first deliberately -- their
narrow `[.., 1, BT]` layout pads 2x, where a batch-first `[.., BT, MB]` would
put MB=16 on the minor axis and pad 8x.

This compares the two paths on identical inputs. They should agree to
floating-point noise, because the arithmetic is byte-identical and only the
order the bytes are fetched in changes.
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["PALLAS_INTERPRET"] = "1"

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_repo, "tokamax")) and _repo not in sys.path:
  sys.path.insert(0, _repo)

from tokamax._src.ops.experimental.kda import common           # noqa: E402
from tokamax._src.ops.experimental.kda import pallas_mosaic_tpu_bwd_kernel as bwd  # noqa: E402


class _FakeTpu:
  vmem_capacity_bytes = 64 * 1024 * 1024
  num_sublanes = 8
  num_lanes = 128
  generation = 7


common.pltpu.get_tpu_info = lambda *a, **k: _FakeTpu()

H, B, T, K, V, BT = 4, 1, 256, 128, 128, 64
NT = T // BT
NAMES = ("dq", "dk", "dv", "db", "dg", "dh0")


def _inputs(h=H):
  k = jax.random.split(jax.random.PRNGKey(0), 12)
  n = lambda i, s: jax.random.normal(k[i], s, jnp.float32)
  return dict(
      q=n(0, (h, B, T, K)), k=n(1, (h, B, T, K)), v=n(2, (h, B, T, V)),
      v_new=n(3, (h, B, T, V)),
      g=-jnp.abs(n(4, (h, B, T, 1))) * 0.5,
      beta=jax.nn.sigmoid(n(5, (h, B, T))),
      A=n(6, (h, B, T, BT)), h=n(7, (h, B, NT, K, V)),
      do=n(8, (h, B, T, V)), dv0=n(9, (h, B, T, V)),
      dAqk=n(10, (h, B, T, BT)),
  )


def _to_bf(x):
  """`[H, B, T, D]` -> `[B, T, H, D]`, the layout the model actually has."""
  return jnp.transpose(x, (1, 2, 0, 3))


def _from_bf(x):
  return jnp.transpose(x, (2, 0, 1, 3))


def run(batch_first):
  a = _inputs()
  big = ("q", "k", "v", "v_new", "do", "dv0")
  kw = dict(a)
  if batch_first:
    for name in big:
      kw[name] = _to_bf(a[name])
  outs = bwd._fused_dhu_wy_intra_cumsum_pallas_jit(
      qg=None, kg=None, w=None, dht=None,
      scale=K ** -0.5, per_channel_gate=False,
      chunk_size=BT, batch_first=batch_first, **kw)
  outs = list(outs)
  if batch_first:  # dq, dk, dv come back batch-first
    for i in range(3):
      outs[i] = _from_bf(outs[i])
  return outs


def rel(a, b):
  a, b = jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32)
  denom = jnp.maximum(jnp.max(jnp.abs(b)), 1e-6)
  return float(jnp.max(jnp.abs(a - b)) / denom)


def main():
  print("batch_first vs head-first, same inputs, Pallas in interpret mode")
  print(f"H={H} B={B} T={T} K={K} chunk={BT}\n")
  ref = run(False)
  got = run(True)
  ok = True
  for name, r, g in zip(NAMES, ref, got):
    if r is None or g is None:
      print(f"  {name:<5} both None")
      continue
    e = rel(g, r)
    flag = "" if e < 1e-5 else "   <- DISAGREES"
    print(f"  {name:<5} {e:.2e}{flag}")
    ok = ok and e < 1e-5
  print()
  if not ok:
    print("batch_first does NOT match head-first. Do not launch.")
    return 1
  print("batch_first matches head-first.\n")

  # Interpret mode ignores Mosaic's block-shape rules, and the batch-first
  # block puts MB on the second-minor axis where head-first put BT. Mosaic
  # wants that axis to equal the array's or be a multiple of 8, so it has to
  # be checked, not assumed. Production is H=32 with MB=16.
  print("lowering the batch-first path for a TPU:")
  ok2 = True
  for h, mb in ((4, None), (32, 16), (32, 8)):
    a = _inputs(h)
    big = ("q", "k", "v", "v_new", "do", "dv0")
    kw = {n2: (_to_bf(x) if n2 in big else x) for n2, x in a.items()}
    try:
      jax.jit(
          lambda **kk: bwd._fused_dhu_wy_intra_cumsum_pallas_jit(
              qg=None, kg=None, w=None, dht=None, scale=K ** -0.5,
              per_channel_gate=False, chunk_size=BT, batch_first=True,
              mini_batch=mb, **kk)
      ).trace(**kw).lower(lowering_platforms=("tpu",))
      print(f"  lowers    H={h:<3} MB={mb}")
    except Exception as e:  # noqa: BLE001
      print(f"  REJECTED  H={h:<3} MB={mb}")
      print(f"            {str(e).splitlines()[0][:120]}")
      ok2 = False
  print()
  if not ok2:
    print("batch_first does not lower. Do not launch.")
    return 1
  print("batch_first matches head-first and lowers.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
