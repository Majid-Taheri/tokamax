# Validation scripts for the scalar-gate (Gated Delta Net) path

Scratch harnesses, not part of the library. Drop this directory before any
upstream pull request.

Run them from anywhere once tokamax is installed:

| script | what it answers |
|---|---|
| `kda_gate_nan_repro.py` | Reproduces the NaN on stock upstream. Exits 1 when it reproduces. |
| `kda_native_gate_test.py` | Does the bug also hit KDA used as KDA? (Yes with an unbounded gate, no with `lower_bound`.) |
| `test_per_channel_gate.py` | Forward: scalar path finite and matching the XLA reference where the per-channel path NaNs. |
| `test_per_channel_gate_grad.py` | Same for all five gradients. |
| `test_narrow_gate.py` | Width-1 gate agrees with width-128, forward and backward. |
| `bench_per_channel_gate.py` | Times both paths. `KDA_H/B/T/DTYPE/ITERS` override the shapes. |

Do not set `JAX_DEFAULT_MATMUL_PRECISION=highest`: it makes the KDA backward
fail to compile in bfloat16, which is a harness artefact and not a kernel bug.
