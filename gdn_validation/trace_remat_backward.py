#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace the low-memory backward with a scalar gate, without a TPU.

`rematerialize_for_backward=True` drops the chunk hidden states from the
forward residuals and rebuilds them in the backward. Nothing had ever selected
it, so the recompute path never saw a width-1 gate -- and it assumed the gate
was as wide as the key. On a 256-chip run that surfaced as:

    TypeError: cannot reshape array of shape (64, 1, 65536, 1) (size 4194304)
               into shape (65536, 64, 128) (size 536870912)

That failure happens while JAX is still tracing, before XLA is involved and
long before the kernel would run. So it reproduces on a CPU in a second, and
a whole class of the same mistake can be swept out at once rather than one
256-chip launch per bug.

We only trace: `jax.eval_shape` walks the forward and the custom backward and
builds every intermediate shape, without emitting a Mosaic kernel. Shapes that
disagree raise here exactly as they did on the cluster.

Run with no arguments. Exits non-zero on the first shape error.
"""

import os
import sys
import traceback

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_repo, "tokamax")) and _repo not in sys.path:
  sys.path.insert(0, _repo)

from tokamax._src.ops.experimental.kda import api               # noqa: E402
from tokamax._src.ops.experimental.kda import pallas_mosaic_tpu as pmt  # noqa: E402

# Small but structurally faithful: several chunks per sequence, more than one
# head, and a head dim that differs from the chunk so a wrong axis cannot
# silently pass.
H, B, T, K, V, BT = 4, 1, 256, 128, 128, 64

_MESH = jax.sharding.AbstractMesh(
    (1,), ("x",),
    abstract_device=jax.sharding.AbstractDevice("TPU v6e", 1, "tpu"))


def inputs(gate_width):
  k = jax.random.split(jax.random.PRNGKey(0), 5)
  n = lambda i, s: jax.random.normal(k[i], s, jnp.float32)
  return dict(
      query=n(0, (H, B, T, K)),
      key=n(1, (H, B, T, K)),
      value=n(2, (H, B, T, V)),
      gate=-jnp.abs(n(3, (H, B, T, gate_width))),
      beta=jax.nn.sigmoid(n(4, (H, B, T))),
  )


def trace(*, gate_width, per_channel_gate, rematerialize):
  """Trace forward+backward and return None, or the exception if it fails."""
  import dataclasses as dc
  args = inputs(gate_width)
  # The config lives on the op, not on the call. Pin it rather than leaving it
  # None, or autotuning picks and we would not know which path we traced.
  cfg = pmt.Config(chunk_size=BT, rematerialize_for_backward=rematerialize)
  op = dc.replace(api.IMPLEMENTATIONS["mosaic_tpu"], config=cfg)

  def loss(query, key, value, gate, beta):
    out, _ = op(
        query, key, value, gate, beta,
        a_log=None, delta_time_bias=None, scale=None,
        initial_state=None, output_final_state=False,
        use_qk_l2norm=True, use_gate_in_kernel=False,
        per_channel_gate=per_channel_gate, chunk_size=BT,
        segment_ids=None, lower_bound=None,
        context_parallel_metadata=None)
    return jnp.sum(out.astype(jnp.float32))

  # The kernel asks the hardware how much VMEM it has, which a CPU cannot
  # answer, so name a TPU for it. v6e rather than v7 only because this JAX does
  # not know v7 yet; the chip picks the mini-batch, not the gate width, and the
  # bug we are after is a reshape that ignores the gate width entirely.
  with jax.sharding.use_abstract_mesh(_MESH):
    try:
      jax.eval_shape(jax.grad(loss, argnums=(0, 1, 2, 3, 4)), *args.values())
      return None
    except Exception as e:  # noqa: BLE001
      return e


def main():
  import dataclasses as dc
  # No TPU here, and we are not running the kernel -- only tracing it. The
  # device check would refuse before we get to the shapes we care about.
  for name, op in list(api.IMPLEMENTATIONS.items()):
    api.IMPLEMENTATIONS[name] = dc.replace(op, bypass_device_check=True)

  cases = [
      ("scalar gate, save everything", 1, False, False),
      ("scalar gate, rematerialize", 1, False, True),
      ("per-channel gate, save everything", K, True, False),
      ("per-channel gate, rematerialize", K, True, True),
  ]

  print(f"tracing on {jax.devices()[0]} -- shapes only, no kernel is run")
  print(f"H={H} B={B} T={T} K={K} chunk={BT}\n")

  failures = []
  for label, gw, pcg, remat in cases:
    err = trace(gate_width=gw, per_channel_gate=pcg, rematerialize=remat)
    if err is None:
      print(f"  ok      {label}")
    else:
      print(f"  FAILED  {label}")
      print(f"          {type(err).__name__}: {str(err).splitlines()[0][:150]}")
      failures.append((label, err))

  if failures:
    print(f"\n{len(failures)} of {len(cases)} failed. First traceback:\n")
    traceback.print_exception(failures[0][1])
    sys.exit(1)

  print("\nall four combinations trace cleanly.")


if __name__ == "__main__":
  main()
