#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the real KDA op run its forward kernel once under `jax.checkpoint`?

`probe_remat_custom_vjp.py` established the mechanism on a toy `custom_vjp`:
a name on the residuals and `optimize_remat=False` together keep them, and
neither alone does anything. This checks the same thing on the actual op,
through `tokamax/_src/ops/op.py`, in the arrangement MaxText uses -- the layer
wrapped in `jax.checkpoint` with a `save_only_these_names` policy, and
`jax.shard_map` in between, which the XProf analysis blamed for hiding names.

Counting `pallas_call` in the gradient's jaxpr answers it. Two calls means the
forward was re-run to rebuild the residuals; that second call is the 912.53 ms
the 397B trace charges us. `make_jaxpr` traces without lowering, so this runs
on a CPU in a second and never touches a chip.

Three things are checked, and all three must hold:

  1. KDA runs its forward once, with the policy, with and without shard_map.
  2. It still runs twice with a policy that does *not* name `kda_residuals`,
     because the saving must be the caller's decision, not ours.
  3. An op that has not opted in is byte-for-byte unchanged. `op.py` is shared
     by every Tokamax op and this must not alter any of them.

Correctness is not this script's job -- `check_remat_gradients.py` owns that,
and it is run afterwards.
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
)

import dataclasses as dc                                        # noqa: E402
import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P               # noqa: E402

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_repo, "tokamax")) and _repo not in sys.path:
  sys.path.insert(0, _repo)

from tokamax._src.ops import op as op_lib                       # noqa: E402
from tokamax._src.ops.experimental.kda import api               # noqa: E402
from tokamax._src.ops.experimental.kda import common            # noqa: E402
from tokamax._src.ops.experimental.kda import pallas_mosaic_tpu as pmt  # noqa: E402


# Same reason as `check_remat_gradients.py`: the kernels ask the chip how much
# VMEM it has in order to size their mini-batch, and a CPU cannot answer. This
# only picks a tile size, so it cannot change what is counted here.
class _FakeTpu:
  vmem_capacity_bytes = 64 * 1024 * 1024
  num_sublanes = 8
  num_lanes = 128
  generation = 7


common.pltpu.get_tpu_info = lambda *a, **k: _FakeTpu()

SAVED = "kda_residuals"
H, B, T, K, V, BT = 4, 1, 256, 128, 128, 64


def _count(jaxpr, name, seen=None):
  """Count a primitive across nested jaxprs. `pretty_print` alone misses them."""
  n = 0
  for eqn in jaxpr.eqns:
    if str(eqn.primitive) == name:
      n += 1
    for v in eqn.params.values():
      sub = getattr(v, "jaxpr", v)
      if hasattr(sub, "eqns"):
        n += _count(sub, name)
      elif isinstance(v, (list, tuple)):
        for item in v:
          sub = getattr(item, "jaxpr", item)
          if hasattr(sub, "eqns"):
            n += _count(sub, name)
  return n


def _inputs():
  k = jax.random.split(jax.random.PRNGKey(0), 5)
  n = lambda i, s: jax.random.normal(k[i], s, jnp.float32)
  return (
      jax.nn.silu(n(0, (H, B, T, K))),
      jax.nn.silu(n(1, (H, B, T, K))),
      jax.nn.silu(n(2, (H, B, T, V))),
      -jnp.abs(n(3, (H, B, T, 1))) * 0.5,
      jax.nn.sigmoid(n(4, (H, B, T))),
  )


def _kda(query, key, value, gate, beta):
  op = dc.replace(
      api.IMPLEMENTATIONS["mosaic_tpu"],
      config=pmt.Config(chunk_size=BT, rematerialize_for_backward=False),
      bypass_device_check=True,
  )
  out, _ = op(
      query, key, value, gate, beta,
      a_log=None, delta_time_bias=None, scale=None,
      initial_state=None, output_final_state=False,
      use_qk_l2norm=True, use_gate_in_kernel=False,
      per_channel_gate=False, chunk_size=BT,
      segment_ids=None, lower_bound=None,
      context_parallel_metadata=None,
  )
  return out


def forward_calls(*, saved_names, use_shard_map):
  """How many times does the forward Pallas kernel appear in the gradient?"""
  policy = jax.checkpoint_policies.save_only_these_names(*saved_names)
  mesh = Mesh(np.array(jax.devices()[:4]), ("h",))
  args = _inputs()
  cot = jax.random.normal(jax.random.PRNGKey(9), (H, B, T, V), jnp.float32)

  def inner(*a):
    f = _kda
    if use_shard_map:
      # Shard over heads, as MaxText shards this layer. `check_vma=False`
      # because the Pallas calls declare their outputs with a plain
      # `ShapeDtypeStruct`, which carries no `manual_axis_type`; that is a
      # property of the kernels, not of what is being measured here.
      f = jax.shard_map(
          _kda,
          mesh=mesh,
          in_specs=(P("h"), P("h"), P("h"), P("h"), P("h")),
          out_specs=P("h"),
          check_vma=False,
      )
    return jnp.sum(f(*a).astype(jnp.float32) * cot)

  loss = lambda *a: jax.checkpoint(inner, policy=policy)(*a)
  jaxpr = jax.make_jaxpr(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))(*args)
  # The backward launches its own kernels, so count only the forward's. Its
  # Pallas calls are the ones that also appear when no gradient is taken.
  return _count(jaxpr.jaxpr, "pallas_call")


def forward_only_calls():
  """Pallas calls in the forward alone -- the baseline one 'run' costs."""
  jaxpr = jax.make_jaxpr(lambda *a: jnp.sum(_kda(*a)))(*_inputs())
  return _count(jaxpr.jaxpr, "pallas_call")


def main():
  ok = True
  one = forward_only_calls()
  print(f"forward alone launches {one} Pallas kernel(s). That is one 'run'.\n")

  print("Pallas calls in the gradient, and how many forward runs that is:")
  print(f"  {'policy':<34} {'plain':>14} {'shard_map':>14}")
  print("  " + "-" * 64)

  rows = [
      ("saves 'kda_residuals'", (SAVED,)),
      ("saves something else", ("unrelated_name",)),
  ]
  results = {}
  for label, names in rows:
    cells = []
    for sm in (False, True):
      n = forward_calls(saved_names=names, use_shard_map=sm)
      results[(label, sm)] = n
      cells.append(f"{n}  ({n / one:.2g}x fwd)")
    print(f"  {label:<34} {cells[0]:>14} {cells[1]:>14}")

  print()
  for sm in (False, True):
    where = "with shard_map" if sm else "plain"
    kept = results[("saves 'kda_residuals'", sm)]
    other = results[("saves something else", sm)]
    if kept >= other:
      print(f"  FAIL {where}: naming the residuals saved nothing "
            f"({kept} vs {other} Pallas calls)")
      ok = False
    else:
      print(f"  pass {where}: {other - kept} fewer Pallas call(s) "
            f"when the policy names {SAVED!r}")

  # op.py is shared. An op that has not opted in must be built exactly as
  # before, which means `optimize_remat` stays on.
  print()
  everything = _all_ops()
  opted_in = sorted(c.__name__ for c in everything
                    if c.residuals_checkpoint_name is not None)
  print(f"  {len(everything)} Op subclasses imported, {len(opted_in)} opted in:")
  print(f"    {', '.join(opted_in) or 'none'}")
  unexpected = [n for n in opted_in if "KimiDeltaAttention" not in n]
  if unexpected:
    print(f"  FAIL: an op that should not have opted in did: {unexpected}")
    ok = False
  else:
    print("  pass: every other op keeps optimize_remat=True and is unchanged")

  print()
  print("PASS" if ok else "FAIL")
  return 0 if ok else 1


def _all_ops():
  """Every `Op` subclass that has been imported."""
  seen, stack = set(), [op_lib.Op]
  while stack:
    c = stack.pop()
    if c in seen:
      continue
    seen.add(c)
    stack.extend(c.__subclasses__())
  return seen


if __name__ == "__main__":
  sys.exit(main())
