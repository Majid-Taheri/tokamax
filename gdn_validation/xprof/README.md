# Where the step time goes

`compare_arms.py` puts two XProf traces side by side and reconciles them to
the wall-clock step time. It is the only thing in this project that located
the gap; everything before it was inference, and the inferences were wrong.

## The two runs

Both on 256 Ghostfish chips, same day, same config: Qwen3.5-397B-A17B,
`cp=4 ep=4`, `per_device_batch_size=0.25`, `T=65536`, `chunk_size=64`.

| arm | kernel | xid | XProf session |
|---|---|---|---|
| A | MaxKernel, the baseline | 288123013 | `majidtaheri-10815935836637529643` |
| B | Tokamax KDA, ours | 289006766 | `majidtaheri-7800865432542923038` |

## Running it

Drop the two HLO op-stat dumps next to the script and run it:

```
gdn_validation/xprof/arm_a_hlo_stats_all.json
gdn_validation/xprof/arm_b_hlo_stats_all.json
```

Each is a list of objects carrying at least `op_name`, `category`,
`occurrences`, `total_self_time_us`, `source_file`, `source_line` and
`tf_op_name`. Self time is summed over all 256 chips, hence the `/256000` to
reach milliseconds per step per chip.

The JSONs are not committed — they are large and tied to two specific
sessions. Re-export them from XProf if you need to re-run this.

## What it found, September 2026

```
total step               A 54,055.88 ms    B 55,588.39 ms    +1,532.51
  the GDN layer          A  4,532.05       B  6,572.66       +2,040.61
  everything else        A 49,523.83       B 49,015.73         -508.10
```

The gap is entirely the GDN layer. Everything else is slightly in our favour.

Inside the layer:

```
Pallas kernels           A  2,736.73       B  3,269.18         +532.44
non-kernel JAX ops       A  1,795.31       B  3,303.48       +1,508.17
```

**Three quarters of the gap is not the kernel.** It is what surrounds it:

| | cost to us |
|---|---|
| causal conv1d as a standalone op, across forward, remat and backward | ~1,545 ms |
| width-1 gate relayouts, `[B,H,T] -> [B,H,T,1]` | ~609 ms |
| pure-JAX Q/K L2-norm and head transposes, run twice | ~320 ms |

The baseline fuses conv1d into its forward kernel and pays almost nothing for
it in the forward.

## The result that changed the plan

Both arms recompute their own forward during the backward: a `custom_vjp`'s
residuals carry no name, so a `save_only_these_names` policy has nothing to
match and rebuilds them.

> **Correction, 14 September.** This section first blamed `shard_map` for
> hiding the names. That is wrong, and it was wrong in a way that would have
> sent someone down the wrong path. Tested both ways
> (`probe_kda_residual_saving.py`): with the residuals named and
> `optimize_remat=False`, the forward is kept whether or not there is a
> `shard_map` in between. The wrapper that really hides a name is
> `custom_batching.custom_vmap`, which `op.py` puts around `fwd` — a tag
> applied inside it is sealed into the `custom_vmap_call`'s jaxpr. Fixed in
> 8cd20fd; the tag is now applied outside that wrapper.

| | recompute cost |
|---|---|
| baseline | 759.66 ms kernel + 283.98 ms non-kernel |
| ours | 912.53 ms kernel + 1,077.50 ms non-kernel |

As a *difference* that is only +152.87 ms on the kernel. But removing ours
outright is worth the full ~1,990 ms, and the baseline keeps paying its
~1,044 ms — which is why stopping the recompute is the first thing to fix and
not the third.

## Two numbers I had wrong before this ran

Recorded because both changed a decision.

**Conv1d.** I estimated 15 ms and argued against fusing it on that basis. It
costs ~1,545 ms. The 15 ms figure was the HBM traffic that fusion saves, not
the cost of running the op.

**The recompute.** I reported 912 ms as gap, which credited us with a cost the
baseline also pays. It is +153 ms as a difference and ~1,990 ms as an absolute
saving. Both framings are useful; conflating them is not.
