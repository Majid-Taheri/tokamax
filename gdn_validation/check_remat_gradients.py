#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Do the two backward strategies produce the same gradients?

`rematerialize_for_backward=False` saves the chunk hidden states; `True` drops
them and rebuilds them. Both are meant to compute the same derivative. Nothing
had ever selected `True`, so that was never checked -- and after three fixes
that only made it *run*, a 256-chip job produced a correct loss at step 0 and
NaN at step 1. A right-shaped backward that returns wrong numbers looks exactly
like a working one until the loss diverges.

`PALLAS_INTERPRET=1` runs the Pallas kernels as ordinary JAX, so this compares
real values on a CPU. Interpret mode is slow, hence the small shapes; what
matters is that NT > 1, so the inter-chunk recurrence actually iterates and the
recompute has something to get wrong.

The save path is the reference: it is what trained correctly end to end. Any
disagreement is a bug in the recompute path.
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["PALLAS_INTERPRET"] = "1"

import dataclasses as dc                                        # noqa: E402
import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_repo, "tokamax")) and _repo not in sys.path:
  sys.path.insert(0, _repo)

from tokamax._src.ops.experimental.kda import api              # noqa: E402
from tokamax._src.ops.experimental.kda import common           # noqa: E402
from tokamax._src.ops.experimental.kda import pallas_mosaic_tpu as pmt  # noqa: E402

# The kernels ask the chip how much VMEM it has, to size their mini-batch. A
# CPU cannot answer, and an AbstractMesh only satisfies tracing -- JAX refuses
# to *execute* under one. So answer for it, with a v7's numbers.
#
# Patch the hardware query rather than `get_tpu_limits`: the kernel modules
# import that by name, so replacing it on `common` would not reach them.
# Either way this only picks a tile size, never an arithmetic result, so it
# cannot change a gradient.
class _FakeTpu:
  vmem_capacity_bytes = 64 * 1024 * 1024
  num_sublanes = 8
  num_lanes = 128
  generation = 7


common.pltpu.get_tpu_info = lambda *a, **k: _FakeTpu()

# Small, because interpret mode is slow. NT = T // BT = 4, so the chunk
# recurrence runs four times rather than being a single degenerate step.
H, B, T, K, V, BT = 2, 1, 256, 128, 128, 64

ARG_NAMES = ("query", "key", "value", "gate", "beta")


def inputs(gate_width):
  k = jax.random.split(jax.random.PRNGKey(0), 5)
  n = lambda i, s: jax.random.normal(k[i], s, jnp.float32)
  # A gate decaying at a realistic Gated Delta Net rate, not a mild one: the
  # whole point of the scalar path is that this range is where it matters.
  return (
      jax.nn.silu(n(0, (H, B, T, K))),
      jax.nn.silu(n(1, (H, B, T, K))),
      jax.nn.silu(n(2, (H, B, T, V))),
      -jnp.abs(n(3, (H, B, T, gate_width))) * 0.5,
      jax.nn.sigmoid(n(4, (H, B, T))),
  )


def grads(*, gate_width, per_channel_gate, rematerialize):
  args = inputs(gate_width)
  cfg = pmt.Config(chunk_size=BT, rematerialize_for_backward=rematerialize)
  # Interpret mode runs the kernel as plain JAX, so the device check would
  # refuse on a CPU before any of it executes.
  op = dc.replace(api.IMPLEMENTATIONS["mosaic_tpu"], config=cfg,
                  bypass_device_check=True)
  cot = jax.random.normal(jax.random.PRNGKey(9), (H, B, T, V), jnp.float32)

  def loss(query, key, value, gate, beta):
    out, _ = op(
        query, key, value, gate, beta,
        a_log=None, delta_time_bias=None, scale=None,
        initial_state=None, output_final_state=False,
        use_qk_l2norm=True, use_gate_in_kernel=False,
        per_channel_gate=per_channel_gate, chunk_size=BT,
        segment_ids=None, lower_bound=None,
        context_parallel_metadata=None)
    return jnp.sum(out.astype(jnp.float32) * cot)

  return jax.grad(loss, argnums=(0, 1, 2, 3, 4))(*args)


def rel(a, b):
  a, b = a.astype(jnp.float32), b.astype(jnp.float32)
  return float(jnp.max(jnp.abs(a - b))) / (float(jnp.max(jnp.abs(b))) + 1e-12)


def compare(label, *, gate_width, per_channel_gate, tol=2e-2):
  print(f"\n{label}")
  saved = grads(gate_width=gate_width, per_channel_gate=per_channel_gate,
                rematerialize=False)
  remat = grads(gate_width=gate_width, per_channel_gate=per_channel_gate,
                rematerialize=True)

  ok = True
  for name, s, r in zip(ARG_NAMES, saved, remat):
    bad = int(jnp.sum(~jnp.isfinite(r.astype(jnp.float32))))
    if bad:
      print(f"  d_{name:<6} NaN/Inf in {100.0 * bad / r.size:.0f}% of entries")
      ok = False
      continue
    e = rel(r, s)
    flag = "" if e < tol else "   <- DISAGREES"
    print(f"  d_{name:<6} {e:.2e}{flag}")
    ok = ok and e < tol
  return ok


def main():
  print(f"device: {jax.devices()[0]}   (Pallas in interpret mode)")
  print(f"H={H} B={B} T={T} K={K} chunk={BT} -> NT={T // BT}")
  print("reference is rematerialize=False, the path that trains correctly.")

  results = [
      compare("scalar gate (Gated Delta Net)", gate_width=1,
              per_channel_gate=False),
      compare("per-channel gate (Kimi Delta Attention)", gate_width=K,
              per_channel_gate=True),
  ]

  print()
  if all(results):
    print("both gate modes: rematerialize matches save.")
  else:
    print("rematerialize does NOT reproduce the saved-residual gradients.")
    sys.exit(1)


if __name__ == "__main__":
  main()
