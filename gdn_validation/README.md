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

## Measured on TPU v5 (2026-09-03)

`KDA_BYPASS_DEVICE_CHECK=1`, H=32 B=4 T=4096 D=128, bf16, mild gate, median of 10:

| | per-channel | scalar, broadcast gate | scalar, width-1 gate |
|---|---|---|---|
| gate tensor | 256.0 MB | 256.0 MB | **2.0 MB** |
| forward | 2.91 ms | 2.83 ms | 3.25 ms |
| fwd+bwd | 9.37 ms | 9.32 ms | 9.81 ms |

Two things this settles, both against the predictions made beforehand:

1. **The regrouping is a correctness fix, not a speedup.** Roughly 10x fewer
   `exp2` and no per-sub-block `[BT, K]` temporaries bought ~1.03x forward and
   nothing overall. Matmuls and memory dominate, and neither changed.
2. **The width-1 gate costs time.** It delivers the 128x memory reduction
   exactly, and is ~10% slower in the forward, ~4% end to end. Widening the
   gate in VMEM immediately after load was tried and made no difference, which
   rules out lane-broadcast cost at the use sites and points at the width-1
   block itself -- Mosaic likely pads it to a full tile anyway, so the HBM
   saving buys nothing on a kernel that is not HBM-bound.

So enable `per_channel_gate=False` for correctness. Pass the gate at width `K`
unless you are memory-bound and can spend the 4%.

Not yet measured on tpu7x, which is the actual target and the only generation
upstream supports.
