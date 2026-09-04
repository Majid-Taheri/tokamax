#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone reproduction: Tokamax's KDA Mosaic kernel returns NaN when the
gate decays quickly.

Run it on one TPU chip:

    python3 kda_gate_nan_repro.py

Needs only `jax` and `tokamax`. No MaxText, no model, no checkpoint. Shapes are
small on purpose (H=4, B=1, T=512, D=128) so the whole thing fits on a single
chip and finishes in a couple of minutes.

WHAT IT SHOWS
-------------
The same op, the same inputs, two implementations. Tokamax's own XLA reference
stays finite. The Mosaic kernel returns NaN once the gate gets steep enough.
Because both sides get byte-identical inputs, nothing about the caller can
explain the difference.

The gate is built exactly the way Qwen3.5's Gated Delta Net builds it:

    A_log = log(Uniform(0, A_max))     dt_bias = 1
    g     = -exp(A_log) * softplus(a + dt_bias)

so sweeping A_max sweeps how fast the state decays per token. Qwen3.5 ships
A_max = 16.

WHY IT HAPPENS
--------------
In pallas_mosaic_tpu_fwd_kernel.py, around line 688, the kernel needs

    Aqk[r,t] = sum_k q[r,k] * k[t,k] * exp2(g[r,k] - g_cumsum[t,k])   for t <= r

The exponent there is always <= 0, so the quantity is always safe. But to batch
the j-loop into a single matmul the kernel factors it around a reference row:

    exp2(g[r] - gn_ref) * exp2(gn_ref - g_cumsum[t])

The product is bounded. The two factors are not. For causal t the second factor
is positive and grows with the cumulative gate drop across the block, so it
overflows to inf while the first underflows to 0, and inf * 0 = NaN.

The code comment there guards the anti-causal rows for exactly this reason, but
the causal direction has the same problem and is not guarded. `safe_gate` moves
the reference row from 0 to BC//2, which halves the exponent range rather than
bounding it -- and it is already enabled here, since it defaults to on whenever
use_gate_in_kernel is False.

BC = 16, so ref_idx = 8 and the split spans 8 tokens of decay. f32 exp2
overflows past 2^128, which predicts the kernel should break once the gate
falls by more than about 128/8 = 16 in log2 units per token. The sweep below
prints the measured drop so you can check that against where it actually
breaks.
"""

import os
import sys

import jax
import jax.numpy as jnp

# Small enough for one chip. head_dim must stay 128; the kernel's chunk is
# fixed at 64 and is not configurable through the public API.
H, B, T, D = 4, 1, 512, 128
CHUNK = 64
LN2 = 0.6931471805599453

# The KDA op has moved around; try the paths it is known to live under. If your
# tree uses a different one, add it here -- everything else in this file is
# plain JAX and does not care where the op came from.
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
_tried = ("tokamax._src.ops.experimental.kda.api",
          "tokamax.ops.experimental.kda.api",
          "tokamax.experimental.kda.api")
for _path in _tried:
  try:
    api = __import__(_path, fromlist=["api"])
    print(f"KDA api imported from: {_path}")
    break
  except ImportError as e:
    _errs.append(f'{_path}: {e}')
    continue
if api is None:
  sys.exit("Could not import the KDA op. Tried:\n  " + "\n  ".join(_tried) +
           "\nAdd your tree's path to _tried. Upstream is openxla/tokamax "
           "PR #1103 (antgroup/tokamax @ antgroup/kda-pallas-kernel, 939da5c).")


def make_inputs(a_max, dtype):
  """Build one set of inputs. Layout is KDA's head-first [H, B, T, *]."""
  k = jax.random.split(jax.random.PRNGKey(0), 6)
  n = lambda i, s: jax.random.normal(k[i], s, jnp.float32)

  # Post-conv activations in a real GDN layer are silu outputs, not gaussians.
  # It makes no difference to this bug, but it keeps the inputs realistic.
  q = jax.nn.silu(n(0, (H, B, T, D)))
  key = jax.nn.silu(n(1, (H, B, T, D)))
  v = jax.nn.silu(n(2, (H, B, T, D)))

  beta = jax.nn.sigmoid(n(3, (H, B, T)))

  # Qwen3.5's own gate, per head and token.
  a_log = jnp.log(jax.random.uniform(k[4], (H, 1, 1), jnp.float32,
                                     minval=1e-9, maxval=a_max))
  g = -jnp.exp(a_log) * jax.nn.softplus(n(5, (H, B, T)) + 1.0)

  # KDA's gate is per key channel; GDN's is one scalar per head and token, so
  # the adapter broadcasts. Exact, and irrelevant to the bug -- every channel
  # carries the same value.
  gate = jnp.broadcast_to(g[..., None], g.shape + (D,))

  # KDA carries a segment axis: [B, N, H, K, V]. GDN packs one sequence per
  # batch row, so N is 1. N must equal max_num_segments.
  init = jnp.zeros((B, 1, H, D, D), jnp.float32)

  c = lambda x: x.astype(dtype)
  return c(q), c(key), c(v), gate.astype(jnp.float32), beta.astype(jnp.float32), init, g


def call_kda(q, key, v, gate, beta, init, impl):
  out, _ = api.kimi_delta_attention(
      q, key, v, gate, beta,
      initial_state=init,
      output_final_state=True,
      # The gate arrives already activated, so the kernel must not apply
      # softplus to it a second time.
      use_gate_in_kernel=False,
      use_qk_l2norm=True,
      implementation=impl,
  )
  return out


def describe(out):
  """Finite or not, and how bad."""
  if out is None:
    return "-"
  bad = int(jnp.sum(~jnp.isfinite(out.astype(jnp.float32))))
  if bad:
    return f"NaN/Inf {100.0 * bad / out.size:5.1f}%"
  return f"finite (max {float(jnp.max(jnp.abs(out.astype(jnp.float32)))):.3e})"


def sweep(dtype, dtype_name):
  print(f"\n{'=' * 78}\nFORWARD, dtype={dtype_name}\n{'=' * 78}")
  print(f"{'A_max':>7} {'min g/token':>12} {'drop/tok':>9} {'chunk drop':>11}  "
        f"{'mosaic':>20}  {'xla reference':>22}")
  print(f"{'':>7} {'(nats)':>12} {'(log2)':>9} {'(log2)':>11}")
  print("-" * 78)

  failed = []
  for a_max in (0.5, 2.0, 8.0, 16.0):
    q, key, v, gate, beta, init, g = make_inputs(a_max, dtype)

    # The cumulative gate inside one 64-token chunk is what reaches exp2.
    cum = jnp.cumsum(g.reshape(H, B, T // CHUNK, CHUNK), axis=3)
    chunk_drop = float(-jnp.min(cum)) / LN2
    per_tok = float(-jnp.min(g)) / LN2

    row = {}
    for impl in ("mosaic", "xla"):
      try:
        row[impl] = describe(jax.block_until_ready(
            call_kda(q, key, v, gate, beta, init, impl)))
      except Exception as e:  # noqa: BLE001 - a crash is a result too
        row[impl] = f"RAISED {type(e).__name__}"

    if "NaN" in row["mosaic"]:
      failed.append(a_max)

    print(f"{a_max:>7.1f} {float(jnp.min(g)):>12.2f} {per_tok:>9.1f} "
          f"{chunk_drop:>11.1f}  {row['mosaic']:>20}  {row['xla']:>22}")
  return failed


def backward_check(dtype, dtype_name):
  """The same defect is in the backward kernel (bwd_kernel.py:840)."""
  print(f"\n{'=' * 78}\nBACKWARD, dtype={dtype_name}\n{'=' * 78}")
  names = ("d_query", "d_key", "d_value", "d_gate", "d_beta")

  for a_max in (0.5, 16.0):
    q, key, v, gate, beta, init, g = make_inputs(a_max, dtype)
    cot = jax.random.normal(jax.random.PRNGKey(9), (H, B, T, D), jnp.float32)

    print(f"\n  A_max={a_max}  (min g = {float(jnp.min(g)):.2f} nats/token)")
    for impl in ("mosaic", "xla"):
      loss = lambda *x: jnp.sum(  # noqa: E731
          call_kda(*x, init, impl).astype(jnp.float32) * cot)
      try:
        grads = jax.block_until_ready(
            jax.grad(loss, argnums=(0, 1, 2, 3, 4))(q, key, v, gate, beta))
        bits = []
        for nm, gr in zip(names, grads):
          bad = int(jnp.sum(~jnp.isfinite(gr.astype(jnp.float32))))
          bits.append(f"{nm}={'NaN%d%%' % (100 * bad // gr.size) if bad else 'ok'}")
        print(f"    {impl:<7} " + "  ".join(bits))
      except Exception as e:  # noqa: BLE001
        print(f"    {impl:<7} RAISED {type(e).__name__}: {str(e).replace(chr(10), ' | ')[:400]}")


def main():
  if os.environ.get("JAX_DEFAULT_MATMUL_PRECISION") == "highest":
    print("WARNING: JAX_DEFAULT_MATMUL_PRECISION=highest makes the KDA backward\n"
          "         fail to COMPILE in bfloat16 ('Bad rhs type: 256, 256').\n"
          "         That is a harness artefact, not this bug. Unset it.\n")

  devs = jax.devices()
  print(f"device: {devs[0]}   ({len(devs)} visible)")
  print(f"shapes: H={H} B={B} T={T} head_dim={D}  chunk={CHUNK}")
  print("\nBoth implementations receive byte-identical inputs.")

  failed = sweep(jnp.float32, "float32")
  sweep(jnp.bfloat16, "bfloat16")
  backward_check(jnp.float32, "float32")

  print(f"\n{'=' * 78}")
  if failed:
    print("RESULT: reproduced.")
    print(f"  The Mosaic kernel returns NaN at A_max = {failed}, while Tokamax's")
    print("  own XLA reference stays finite on the same inputs.")
    print("  Qwen3.5's Gated Delta Net ships A_max = 16, so it lands in the")
    print("  failing range and training gives loss = nan at step 0.")
    print("\n  Cause: pallas_mosaic_tpu_fwd_kernel.py:688 splits an exponent that")
    print("  is always <= 0 into two factors, one of which overflows to inf while")
    print("  the other underflows to 0. inf * 0 = NaN. The backward repeats the")
    print("  pattern at pallas_mosaic_tpu_bwd_kernel.py:840.")
    sys.exit(1)
  print("RESULT: not reproduced -- the Mosaic kernel stayed finite everywhere.")
  print("  If this happens, the kernel has likely been fixed since 939da5c.")


if __name__ == "__main__":
  main()
