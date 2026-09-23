#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does native GQA in the backward kernel give the same gradients?

Today the caller expands q and k from the key head count to the value head
count with `jnp.repeat` in HBM, hands all of them to the kernel, gets all of
them back, and sums them down again outside. At H_v=32, H_k=8 that makes q, k,
dq and dk four times larger in memory than they need to be, and XProf charges
the expand, the reduce and the L2-norm VJP riding on them about 195 ms per step.
The competing kernel does all of it in VMEM and pays 9 ms.

`gqa_repeats=R` moves it inside. The kernel takes q and k at the key head
count, broadcasts them up a leading axis in VMEM, and sums dq and dk back down
before writing.

This compares the two on identical inputs. They must agree to floating-point
noise: broadcasting then reducing is the same arithmetic in a different place,
and the sum order over R is the only thing that can differ.
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

H, B, T, K, V, BT = 8, 1, 256, 128, 128, 64   # 8 value heads
NT = T // BT
NAMES = ("dq", "dk", "dv", "db", "dg", "dh0")


def _inputs(h, r):
  """q and k at h // r heads; everything else at h."""
  s = jax.random.split(jax.random.PRNGKey(0), 12)
  n = lambda i, shape: jax.random.normal(s[i], shape, jnp.float32)
  return dict(
      q=n(0, (h // r, B, T, K)), k=n(1, (h // r, B, T, K)),
      v=n(2, (h, B, T, V)), v_new=n(3, (h, B, T, V)),
      g=-jnp.abs(n(4, (h, B, T, 1))) * 0.5,
      beta=jax.nn.sigmoid(n(5, (h, B, T))),
      A=n(6, (h, B, T, BT)), h=n(7, (h, B, NT, K, V)),
      do=n(8, (h, B, T, V)), dv0=n(9, (h, B, T, V)),
      dAqk=n(10, (h, B, T, BT)),
  )


def run(h, r, mb=None):
  a = _inputs(h, r)
  if r == 1:
    kw = a
  else:
    # The reference expands outside, exactly as the caller does today.
    kw = dict(a)
    kw["q"] = jnp.repeat(a["q"], r, axis=0)
    kw["k"] = jnp.repeat(a["k"], r, axis=0)
  outs = list(bwd._fused_dhu_wy_intra_cumsum_pallas_jit(
      qg=None, kg=None, w=None, dht=None, scale=K ** -0.5,
      per_channel_gate=False, chunk_size=BT, mini_batch=mb,
      gqa_repeats=(1 if r == 1 else r), **kw))
  if r != 1:
    return outs
  return outs


def ref_then_reduce(h, r, mb=None):
  """Expand outside, run with gqa_repeats=1, then sum dq/dk down outside."""
  a = _inputs(h, r)
  kw = dict(a)
  kw["q"] = jnp.repeat(a["q"], r, axis=0)
  kw["k"] = jnp.repeat(a["k"], r, axis=0)
  outs = list(bwd._fused_dhu_wy_intra_cumsum_pallas_jit(
      qg=None, kg=None, w=None, dht=None, scale=K ** -0.5,
      per_channel_gate=False, chunk_size=BT, mini_batch=mb,
      gqa_repeats=1, **kw))
  for i in (0, 1):
    x = outs[i]
    outs[i] = x.reshape(h // r, r, *x.shape[1:]).sum(axis=1)
  return outs


def native(h, r, mb=None):
  a = _inputs(h, r)
  return list(bwd._fused_dhu_wy_intra_cumsum_pallas_jit(
      qg=None, kg=None, w=None, dht=None, scale=K ** -0.5,
      per_channel_gate=False, chunk_size=BT, mini_batch=mb,
      gqa_repeats=r, **a))


def rel(a, b):
  a, b = jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32)
  return float(jnp.max(jnp.abs(a - b)) / jnp.maximum(jnp.max(jnp.abs(b)), 1e-6))


def main():
  print("native GQA in the kernel vs expand-and-reduce outside it")
  print(f"H_v={H} B={B} T={T} K={K} chunk={BT}\n")
  ok = True
  for r in (2, 4):
    print(f"  --- gqa_repeats={r}, so {H} value heads from {H // r} key heads ---")
    a, b = ref_then_reduce(H, r), native(H, r)
    for name, x, y in zip(NAMES, b, a):
      if x is None or y is None:
        print(f"    {name:<5} both None")
        continue
      e = rel(x, y)
      flag = "" if e < 1e-5 else "   <- DISAGREES"
      print(f"    {name:<5} {e:.2e}{flag}")
      ok = ok and e < 1e-5

  print("\nlowering for TPU, production shape H_v=32 from H_k=8:")
  for mb in (16, 8):
    a = _inputs(32, 4)
    try:
      jax.jit(lambda **kk: bwd._fused_dhu_wy_intra_cumsum_pallas_jit(
          qg=None, kg=None, w=None, dht=None, scale=K ** -0.5,
          per_channel_gate=False, chunk_size=BT, mini_batch=mb,
          gqa_repeats=4, **kk)).trace(**a).lower(lowering_platforms=("tpu",))
      print(f"  lowers    MB={mb}")
    except Exception as e:  # noqa: BLE001
      print(f"  REJECTED  MB={mb}")
      print(f"            {str(e).splitlines()[0][:120]}")
      ok = False

  print()
  if not ok:
    print("native GQA does NOT match. Do not launch.")
    return 1
  print("native GQA matches and lowers.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
