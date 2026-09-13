#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Can a name-based remat policy keep a custom_vjp's residuals?

At 397B the forward Pallas call runs twice per layer: once in the forward
pass and once again inside `transpose(jvp())/.../checkpoint/rematted_
computation`. That second run costs 912.62 ms per step, which is 60% of our
gap to the MaxKernel baseline.

The cause is not in our kernel. `tokamax/_src/ops/op.py:311` builds every op
as

    f = jax.custom_vjp(f)
    f.defvjp(fwd, bwd, optimize_remat=True)

and `optimize_remat=True` asks JAX to re-run `fwd` under remat rather than
keep its residuals. MaxText then wraps the layer in `jax.checkpoint` with a
policy that saves only a fixed list of names, and the residuals -- q, k, v,
aqk, akk, h, g_cumsum, the rstds -- carry no name, so nothing rescues them.

Four questions, each a separate configuration below:

  1. does the default really recompute?
  2. does `optimize_remat=False` on its own stop it?
  3. does naming the residuals inside fwd stop it?
  4. does doing both stop it?

Counting primitives in the backward jaxpr answers all four in a second, on a
CPU, without a kernel or a chip. A real op would confirm it afterwards; this
only establishes whether a fix exists at all.
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
from jax.ad_checkpoint import checkpoint_name                   # noqa: E402

SAVED = "gdn_residual"


def make_op(*, optimize_remat, name_residuals):
  """A custom_vjp shaped like the real one.

  The detail that matters: output and residual come out of a *single*
  expensive call, as they do from a fused Pallas forward. Model them as two
  independent computations and the experiment does not reproduce -- JAX
  simply defers the residual to the backward and nothing runs twice.
  `jnp.sinh` marks that one call.
  """

  def kernel(x):
    y = jnp.sinh(x)
    return y, y * 2.0

  @jax.custom_vjp
  def op(x):
    return kernel(x)[0]

  def fwd(x):
    y, res = kernel(x)
    if name_residuals:
      res = checkpoint_name(res, SAVED)
    return y, res

  def bwd(res, dy):
    return (dy * res,)

  op.defvjp(fwd, bwd, optimize_remat=optimize_remat)
  return op


def _walk(jaxpr, counts):
  """Count primitives across nested jaxprs, which pretty_print alone misses."""
  for eqn in jaxpr.eqns:
    counts[str(eqn.primitive)] = counts.get(str(eqn.primitive), 0) + 1
    for v in eqn.params.values():
      sub = getattr(v, "jaxpr", v)
      if hasattr(sub, "eqns"):
        _walk(sub, counts)
      elif isinstance(v, (list, tuple)):
        for item in v:
          sub = getattr(item, "jaxpr", item)
          if hasattr(sub, "eqns"):
            _walk(sub, counts)
  return counts


def count_forward_in_backward(*, optimize_remat, name_residuals):
  """How many times does the fused forward run in the whole gradient?

  One means the residuals were kept. Two means the whole forward was run
  again to rebuild them -- which is what the 397B trace shows.
  """
  op = make_op(optimize_remat=optimize_remat, name_residuals=name_residuals)

  # MaxText's arrangement: the layer is wrapped in jax.checkpoint with a
  # policy that saves only named values, and everything else is recomputed.
  policy = jax.checkpoint_policies.save_only_these_names(SAVED)

  @jax.jit
  def loss(x):
    layer = jax.checkpoint(lambda a: jnp.sum(op(a)), policy=policy)
    return layer(x)

  jaxpr = jax.make_jaxpr(jax.grad(loss))(jnp.ones((4,)))
  return _walk(jaxpr.jaxpr, {}).get("sinh", 0)


def main():
  print("counting forward executions in the gradient of a checkpointed layer")
  print("1 = residual was kept.  2 = forward was re-run to rebuild it.\n")

  rows = [
      ("default (what tokamax does today)", True, False),
      ("optimize_remat=False", False, False),
      ("residuals named", True, True),
      ("optimize_remat=False AND residuals named", False, True),
  ]

  results = []
  for label, opt, named in rows:
    n = count_forward_in_backward(optimize_remat=opt, name_residuals=named)
    verdict = "kept" if n <= 1 else "RECOMPUTED"
    print(f"  {n}x  {verdict:<11} {label}")
    results.append((label, n))

  print()
  wins = [label for label, n in results if n <= 1]
  if wins:
    print("A fix exists. Cheapest option that keeps the residual:")
    print(f"  {wins[0]}")
  else:
    print("None of these keep the residual. The recompute is not reachable")
    print("from the policy, and avoiding it needs a different approach --")
    print("hoisting the op out of the checkpointed region, most likely.")


if __name__ == "__main__":
  main()
