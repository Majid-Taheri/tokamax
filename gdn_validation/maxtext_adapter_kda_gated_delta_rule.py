"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Gated Delta Rule backed by Tokamax's KDA Pallas kernel.

`kda_chunk_gated_delta_rule` is a drop-in for
`maxtext.models.qwen3.jax_chunk_gated_delta_rule`: same signature, same input
and output layouts. The pure-JAX version has no kernel in either direction;
this routes to a real Pallas TPU forward and custom-VJP backward.

  Kernel: https://github.com/openxla/tokamax  (KDA, experimental)

WHY THIS IS A DROP-IN AND NOT A REWRITE
---------------------------------------
GDN is the per-head-scalar special case of KDA's per-channel gate. Everything
else already lines up:

  * the gate activation is literally the same function. KDA's `_activate_gate`
    is `-exp(a_log) * softplus(gate + delta_time_bias)`; qwen3.py builds
    `g = -exp(A_log) * softplus(a + dt_bias)`. This function receives `g`
    already activated, so neither side has to compute it.
  * the gate is in NATURAL log space on both sides. KDA's reference applies it
    as `state * jnp.exp(g)`; the log2/exp2 in KDA's design doc is internal to
    the Pallas kernel and does not reach the API. No rescaling.
  * the caller has already expanded GQA (query/key repeated to num_v_heads) and
    already applied the causal conv, so this function does neither.

So the conversion is a layout change plus one broadcast.

VALIDATION
----------
Measured on TPU v5 against a token-by-token GDN reference (itself checked
against `jax_chunk_gated_delta_rule` at 2.1e-07), at the training shapes
num_k_heads=16, num_v_heads=32, head_dim=128, T=4096, batch=4, float32,
matmul precision `highest`:

    forward                     3.1e-06
    all seven gradients         3.1e-06 .. 1.7e-05
    packed varlen (segment_ids) 3.1e-06, cross-segment leakage exactly 0

Speed at those shapes on 4 chips of v5p, delta rule only:

    forward + backward   ~1.85x faster than jax_chunk_gated_delta_rule
                         at a matched chunk size of 64

That ratio is measured on one machine at one configuration; it is not
comparable to layer timings from a large sharded run.

KNOWN DIFFERENCES FROM THE PURE-JAX PATH
----------------------------------------
1. `chunk_size` is honoured when tokamax supports it. Stock KDA pins the
   Mosaic config to 64; the scalar-gate build accepts any power of two from
   16 to 512 and this adapter forwards whatever `gdn_chunk_size` asks for.
   Against a tokamax without that support the request is dropped and a warning
   is emitted, rather than silently doing something else. Chunking never
   changes the result -- the recurrence is identical however it is split --
   only speed and memory.
2. Validated in float32. bfloat16 is supported by the kernel but has not been
   checked against a reference here.
3. Requires `tokamax`. The import is deferred so that MaxText still imports
   cleanly without it; the error only appears if this path is selected.
"""

import functools
import logging

import jax
import jax.numpy as jnp
from jax import Array

_warned_chunk_size = False


@functools.lru_cache(maxsize=1)
def _supports_chunk_size() -> bool:
  """Whether the installed tokamax lets the caller choose the chunk size.

  Stock KDA is pinned to 64. The scalar-gate build accepts any power of two,
  which matters because MaxText's production config asks for 128 -- comparing
  a kernel at 64 against the JAX path at 128 is not comparing like with like.
  """
  import inspect  # pylint: disable=g-import-not-at-top

  return "chunk_size" in inspect.signature(
      _kda_api().kimi_delta_attention).parameters


@functools.lru_cache(maxsize=1)
def _supports_per_channel_gate() -> bool:
  """Whether the installed tokamax takes a scalar gate.

  `per_channel_gate=False` is what fixes the NaN on Qwen3.5's gate range, so
  this is not merely an optimisation. Stock tokamax does not have it; see
  github.com/Majid-Taheri/tokamax branch gdn-scalar-gate.
  """
  import inspect  # pylint: disable=g-import-not-at-top

  has = "per_channel_gate" in inspect.signature(
      _kda_api().kimi_delta_attention).parameters
  if not has:
    logging.warning(
        "This tokamax has no per_channel_gate. Falling back to a broadcast "
        "gate, which is 128x larger and returns NaN once the gate decays "
        "faster than about 11 nats per token -- Qwen3.5 needs about 21."
    )
  return has


def _kda_api():
  """Import Tokamax's KDA API, with an actionable message if it is missing."""
  try:
    from tokamax._src.ops.experimental.kda import api  # pylint: disable=g-import-not-at-top

    return api
  except ImportError as e:
    raise ImportError(
        "use_kda_gdn_kernel=True requires the Tokamax KDA op, which is not "
        "installed. Install tokamax, or set use_kda_gdn_kernel=false to use "
        "the pure-JAX jax_chunk_gated_delta_rule."
    ) from e


def kda_chunk_gated_delta_rule(
    query: Array,
    key: Array,
    value: Array,
    g: Array,
    beta: Array,
    chunk_size: int = 64,
    initial_state: None | Array = None,
    use_qk_norm_in_gdn: bool = False,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    implementation: str | None = None,
) -> tuple[Array, None | Array]:
  """Gated Delta Rule via the KDA Pallas kernel.

  Signature matches `jax_chunk_gated_delta_rule`.

  Args:
    query: `[batch, seq, num_v_heads, head_k_dim]`, GQA already expanded.
    key: same shape as `query`.
    value: `[batch, seq, num_v_heads, head_v_dim]`.
    g: log decay per token and head, `[batch, seq, num_v_heads]`, natural log.
    beta: delta-rule write strength in `[0, 1]`, `[batch, seq, num_v_heads]`.
    chunk_size: forwarded to the kernel when supported; see the module
      docstring. Must be a power of two in [16, 512].
    initial_state: `[batch, num_v_heads, head_k_dim, head_v_dim]`, or None.
    use_qk_norm_in_gdn: L2-normalise query and key inside the kernel.
    compute_dtype: dtype for query/key/value.
    implementation: passed to Tokamax. Defaults to Mosaic with an XLA
      fallback; pass "xla" to force the reference path, which is useful when
      bisecting a numerical difference.

  Returns:
    `(core_attn_out, final_state)` with the same layouts the pure-JAX
    implementation returns.
  """
  global _warned_chunk_size
  if not _supports_chunk_size() and chunk_size != 64 and not _warned_chunk_size:
    _warned_chunk_size = True
    logging.warning(
        "kda_chunk_gated_delta_rule: gdn_chunk_size=%d requested, but this "
        "tokamax pins the kernel to 64. Results are unaffected -- the "
        "recurrence does not depend on how it is chunked -- but timings will "
        "not match a chunk_size=%d run of the pure-JAX path.",
        chunk_size,
        chunk_size,
    )

  api = _kda_api()
  batch, seq_len, num_v_heads, head_k_dim = query.shape
  head_v_dim = value.shape[-1]

  # [B, S, H, D] -> [H, B, S, D]. Tokamax's KDA is head-first throughout.
  to_head_first = lambda x: jnp.transpose(x, (2, 0, 1, 3))
  q = to_head_first(query.astype(compute_dtype))
  k = to_head_first(key.astype(compute_dtype))
  v = to_head_first(value.astype(compute_dtype))

  beta_h = jnp.transpose(beta.astype(jnp.float32), (2, 0, 1))  # [H, B, S]
  g_h = jnp.transpose(g.astype(jnp.float32), (2, 0, 1))  # [H, B, S]

  # GDN's gate is one scalar per (head, token); KDA's is one value per key
  # channel. A tokamax with `per_channel_gate` takes the scalar directly, at
  # width 1, and skips the factorisation that overflows on GDN's gate range --
  # which is what makes Qwen3.5's gate work at all. Without that support we
  # fall back to broadcasting, which is exact but 128x larger and still NaNs
  # above roughly 11 nats/token of decay.
  if _supports_per_channel_gate():
    gate = g_h[..., None]                       # [H, B, T, 1]
    gate_kwargs = {"per_channel_gate": False}
    if _supports_chunk_size():
      gate_kwargs["chunk_size"] = chunk_size
  else:
    gate = jnp.broadcast_to(g_h[..., None], g_h.shape + (head_k_dim,))
    gate_kwargs = {}

  # KDA carries a segment axis: [B, N, H, K, V]. GDN packs one sequence per
  # batch row, so N is 1.
  if initial_state is None:
    init = jnp.zeros((batch, 1, num_v_heads, head_k_dim, head_v_dim), jnp.float32)
  else:
    init = initial_state.astype(jnp.float32)[:, None]

  out, final_state = api.kimi_delta_attention(
      q,
      k,
      v,
      gate,
      beta_h,
      initial_state=init,
      output_final_state=True,
      # `g` arrives already activated by the caller, so the kernel must not
      # apply softplus to it a second time.
      use_gate_in_kernel=False,
      use_qk_l2norm=use_qk_norm_in_gdn,
      **gate_kwargs,
      **({} if implementation is None else {"implementation": implementation}),
  )

  core_attn_out = jnp.transpose(out, (1, 2, 0, 3))  # -> [B, S, H, V]
  next_state = None if final_state is None else final_state[:, 0]  # drop N=1
  return core_attn_out.astype(query.dtype), next_state
