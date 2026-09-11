#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Will the kernel compile on a TPU? Asked from a CPU.

The other two checks in this directory answer a different question. Tracing
catches shapes that do not fit together; the gradient check catches wrong
numbers. Neither says anything about whether Mosaic will accept the kernel,
because interpret mode runs the Pallas body as ordinary JAX and never
consults the hardware rules at all.

That gap cost a 256-chip launch. A block layout that saved 11.8 GiB of
padding traced cleanly, produced correct gradients, and was then rejected
outright by Mosaic:

    The Pallas TPU lowering currently requires that the last two dimensions
    of your block shape are divisible by 8 and 128 respectively, or be equal
    to the respective dimensions of the overall array.

The rule is in jax/_src/pallas/mosaic/lowering.py, `_check_block_mappings`:

    (bs0 == as0 or bs0 % 128 == 0) and (bs1 == as1 or bs1 % 8 == 0)

where bs/as are block and array, 0 the last dimension and 1 the one before.

`lower(lowering_platforms=("tpu",))` runs that check, and everything else in
the Mosaic lowering, without a TPU attached. It stops short of codegen, so it
will not find VMEM exhaustion or an MXU width overflow -- those still need
real hardware. It does find every block-shape mistake.

Exits non-zero if any configuration is rejected.
"""

import dataclasses as dc
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_repo, "tokamax")) and _repo not in sys.path:
  sys.path.insert(0, _repo)

from tokamax._src.ops.experimental.kda import api              # noqa: E402
from tokamax._src.ops.experimental.kda import common           # noqa: E402
from tokamax._src.ops.experimental.kda import pallas_mosaic_tpu as pmt  # noqa: E402


class _FakeTpu:
  """A v7's numbers. Only used to size tiles, never to compute anything."""
  vmem_capacity_bytes = 64 * 1024 * 1024
  num_sublanes = 8
  num_lanes = 128
  generation = 7


common.pltpu.get_tpu_info = lambda *a, **k: _FakeTpu()

H, B, T, K, V, BT = 4, 1, 256, 128, 128, 64


def lower(*, gate_width, per_channel_gate, rematerialize, chunk_size):
  """Lower forward and backward for a TPU. Returns None, or the rejection."""
  k = jax.random.split(jax.random.PRNGKey(0), 5)
  n = lambda i, s: jax.random.normal(k[i], s, jnp.float32)
  args = (
      n(0, (H, B, T, K)), n(1, (H, B, T, K)), n(2, (H, B, T, V)),
      -jnp.abs(n(3, (H, B, T, gate_width))), jax.nn.sigmoid(n(4, (H, B, T))),
  )
  cfg = pmt.Config(chunk_size=chunk_size,
                   rematerialize_for_backward=rematerialize)
  op = dc.replace(api.IMPLEMENTATIONS["mosaic_tpu"], config=cfg,
                  bypass_device_check=True)

  def loss(query, key, value, gate, beta):
    out, _ = op(
        query, key, value, gate, beta,
        a_log=None, delta_time_bias=None, scale=None,
        initial_state=None, output_final_state=False,
        use_qk_l2norm=True, use_gate_in_kernel=False,
        per_channel_gate=per_channel_gate, chunk_size=chunk_size,
        segment_ids=None, lower_bound=None,
        context_parallel_metadata=None)
    return jnp.sum(out.astype(jnp.float32))

  try:
    jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4))).trace(*args).lower(
        lowering_platforms=("tpu",))
    return None
  except Exception as e:  # noqa: BLE001
    return e


def main():
  # Chunk 128 is included because the scalar path allows it and the mini-batch
  # clamp that keeps it inside the MXU is easy to undo by accident.
  cases = []
  for chunk in (64, 128):
    for remat in (False, True):
      cases.append((f"scalar gate,      chunk {chunk:>3}, remat={remat!s:<5}",
                    1, False, remat, chunk))
  for remat in (False, True):
    cases.append((f"per-channel gate, chunk  64, remat={remat!s:<5}",
                  K, True, remat, 64))

  print("lowering for TPU from a CPU host -- block shapes, not codegen")
  print(f"H={H} B={B} T={T} K={K}\n")

  failures = []
  for label, gw, pcg, remat, chunk in cases:
    err = lower(gate_width=gw, per_channel_gate=pcg, rematerialize=remat,
                chunk_size=chunk)
    if err is None:
      print(f"  lowers    {label}")
    else:
      print(f"  REJECTED  {label}")
      print(f"            {str(err).splitlines()[0][:130]}")
      failures.append(label)

  print()
  if failures:
    print(f"{len(failures)} of {len(cases)} rejected. Do not launch.")
    sys.exit(1)
  print("all configurations lower. VMEM and MXU limits still need hardware.")


if __name__ == "__main__":
  main()
