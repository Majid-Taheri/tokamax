#!/usr/bin/env python3
"""Side-by-side XProf HLO trace comparison for Qwen3.5-397B:
  - Arm A (Rohan baseline): xid/288123013 (session majidtaheri-10815935836637529643)
  - Arm B (Majid KDA kernel): xid/289006766 (session majidtaheri-7800865432542923038)

Produces:
  1. Top 15 Raw HLO Instructions by Self Time (Side-by-Side)
  2. Top 15 Logical Operation Families by Self Time (Side-by-Side)
  3. Complete Reconciled Step Time & GDN/KDA Layer Breakdown (Forward, Remat, Backward VJP)

INPUTS
------
Two JSON dumps of the XProf HLO op stats, one per arm, next to this file:

    arm_a_hlo_stats_all.json
    arm_b_hlo_stats_all.json

Each is a list of objects with at least: op_name, category, occurrences,
total_self_time_us, source_file, source_line, tf_op_name. `total_self_time_us`
is summed across all 256 chips, hence the /256000 to get ms per step per chip.
Both runs used 256 Ghostfish chips, cp=4, ep=4, batch 0.25, T=65536, chunk 64.

WHY THIS EXISTS
---------------
It is the only thing that located the gap. Everything before it was guesswork,
and the guesses were wrong by two orders of magnitude in one case. What it
found, September 2026:

    total step               A 54,055.88 ms    B 55,588.39 ms    +1,532.51
      the GDN layer          A  4,532.05       B  6,572.66       +2,040.61
      everything else        A 49,523.83       B 49,015.73         -508.10

So the whole gap is in the GDN layer, and inside it:

    Pallas kernels           A  2,736.73       B  3,269.18         +532.44
    non-kernel JAX ops       A  1,795.31       B  3,303.48       +1,508.17

Three quarters of the gap is not the kernel. It is the plumbing around it:
a standalone causal conv1d (~1,545 ms across forward, remat and backward,
which the baseline fuses into its forward kernel), the width-1 gate relayouts
(~609 ms), and pure-JAX L2-norm and transposes run twice.

It also showed the baseline recomputes its own forward -- 759.66 ms -- because
`checkpoint_name` tags inside a `shard_map` are invisible to the outer remat
policy. So our 912.53 ms of the same is only +152.87 ms as a *difference*,
while removing ours entirely is worth the full amount.
"""

import json
import os
import re
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARM_A_JSON = os.path.join(SCRIPT_DIR, "arm_a_hlo_stats_all.json")
ARM_B_JSON = os.path.join(SCRIPT_DIR, "arm_b_hlo_stats_all.json")


def self_ms(op):
  """Converts total_self_time_us across 256 chips into ms per step per chip."""
  return op.get("total_self_time_us", 0.0) / 256000.0


def logical_family(op):
  name = op["op_name"]
  cat = op["category"]
  if "fused_conv1d_gdn_per_seq" in name:
    return "fused_conv1d_gdn_per_seq (Pallas Fwd+Remat)"
  if "gdn_bwd_kernel_group" in name:
    return "gdn_bwd_kernel_group_0..3 (Pallas Bwd)"
  if "_fused_dhu_wy_intra_cumsum_pallas_jit" in name:
    return "_fused_dhu_wy_intra_cumsum (KDA Bwd Kernel 1)"
  if "fused_recompute_w_u_vnew_from_h_pallas" in name:
    return "fused_recompute_w_u_vnew (KDA Bwd Kernel 2)"
  if "chunk_kda_bwd_dAv_kernel" in name:
    return "chunk_kda_bwd_dAv_kernel (KDA Bwd Kernel 3)"
  if any(name.startswith(x) for x in ["shard_map.3069", "shard_map.3070"]) and cat == "custom-call":
    return "PallasMosaicTpuKimiDeltaAttention (KDA Fwd+Remat)"
  if "splash_mha_fwd" in name:
    return "splash_mha_fwd_segmented_residuals"
  if "splash_mha_dkv" in name:
    return "splash_mha_dkv_segmented_no_residuals"
  if "splash_mha_dq" in name:
    return "splash_mha_dq_segmented_no_residuals"
  if name.startswith("gmm_v2") or name.startswith("tgmm_v2"):
    prefix = name.split("-")[0]
    return f"{prefix} (MoE Ragged Dot Kernel)"
  if name.startswith("all-gather"):
    return "all-gather (async-done / wait)"
  if name.startswith("all-reduce"):
    return "all-reduce (async-done / wait)"
  if name.startswith("reduce-scatter"):
    return "reduce-scatter (async-done / wait)"
  if name.startswith("collective-permute"):
    return "collective-permute (ring attn / expert parallel)"
  if name.startswith("gather_offload_async_done"):
    return "gather_offload_async_done (activation offload wait)"
  if name.startswith("gather."):
    return "gather (MoE ragged gather / routing)"
  if ".beta_ref" in name or ".g_ref" in name:
    return "reshape.*.beta_ref / g_ref (KDA gate width-1 relayout)"
  if ".qkv_ref" in name:
    return "reshape/copy.*.qkv_ref (GDN input relayout)"
  base = re.sub(r"\.\d+.*$", "", name)
  return f"{base} ({cat})"


def classify_op_a(op):
  name = op["op_name"]
  cat = op["category"]
  sf = (op.get("source_file") or "").split("/")[-1]
  sl = op.get("source_line", 0)
  tf = op.get("tf_op_name", "")

  if "fused_conv1d_gdn" in name and cat == "custom-call":
    if any(name.endswith(x) for x in [".41", ".42", ".43"]):
      return "1_GDN_FWD_PALLAS_KERNEL"
    else:
      return "2_GDN_REMAT_PALLAS_KERNEL"

  if "gdn_bwd_kernel_group" in name and cat == "custom-call":
    return "3_GDN_BWD_VJP_PALLAS_KERNEL"

  if "gather_offload" in name:
    return "10_NON_GDN_OFFLOAD_AND_OTHER"
  if cat in [
      "async-done",
      "async-start",
      "all-reduce",
      "all-gather",
      "reduce-scatter",
      "collective-permute-start",
      "collective-permute-done",
  ] or any(
      name.startswith(x)
      for x in ["all-gather", "all-reduce", "reduce-scatter", "collective-permute"]
  ):
    return "7_NON_GDN_COLLECTIVES"

  if (sf == "qwen3.py" and sl == 994) or sf == "hybrid_gdn.py":
    if any(k in tf for k in ["ragged-sort", "ragged_gather", "splash", "ring_attention", "gmm"]):
      pass
    else:
      if "transpose(jvp" in tf and "rematted_computation" not in tf:
        return "6_GDN_BWD_VJP_JAX_OPS"
      elif "rematted_computation" in tf:
        return "5_GDN_REMAT_JAX_OPS"
      else:
        return "4_AND_5_SPLIT_A"

  if sf in ["ragged_sort.py", "moe.py", "ops.py"]:
    return "8_NON_GDN_MOE_AND_GMM"
  if sf in ["ring_attention_kernel.py", "attentions.py", "attention_op.py", "tokamax_ring_attention.py"]:
    return "9_NON_GDN_SPLASH_ATTN"
  return "10_NON_GDN_OFFLOAD_AND_OTHER"


def classify_op_b(op):
  name = op["op_name"]
  cat = op["category"]
  sf = (op.get("source_file") or "").split("/")[-1]
  sl = op.get("source_line", 0)
  tf = op.get("tf_op_name", "")

  if cat == "custom-call":
    if any(
        name.startswith(x)
        for x in [
            "shard_map.30691",
            "shard_map.30692",
            "shard_map.30693",
            "shard_map.30694",
            "shard_map.30695",
            "shard_map.30696",
        ]
    ):
      return "1_GDN_FWD_PALLAS_KERNEL"
    if any(
        name.startswith(x)
        for x in [
            "shard_map.30697",
            "shard_map.30698",
            "shard_map.30699",
            "shard_map.30700",
            "shard_map.30701",
            "shard_map.30702",
        ]
    ):
      return "2_GDN_REMAT_PALLAS_KERNEL"
    if any(
        k in name
        for k in [
            "_fused_dhu_wy_intra_cumsum_pallas_jit",
            "fused_recompute_w_u_vnew_from_h_pallas",
            "chunk_kda_bwd_dAv_kernel",
        ]
    ):
      return "3_GDN_BWD_VJP_PALLAS_KERNEL"

  if "gather_offload" in name:
    return "10_NON_GDN_OFFLOAD_AND_OTHER"
  if cat in [
      "async-done",
      "async-start",
      "all-reduce",
      "all-gather",
      "reduce-scatter",
      "collective-permute-start",
      "collective-permute-done",
  ] or any(
      name.startswith(x)
      for x in ["all-gather", "all-reduce", "reduce-scatter", "collective-permute"]
  ):
    if "sparse-core-data-format" not in name:
      return "7_NON_GDN_COLLECTIVES"

  is_gdn_b = False
  if sf in ["op.py", "kda_gated_delta_rule.py", "linear.py"]:
    is_gdn_b = True
  elif sf == "qwen3.py" and sl in [
      1240, 1103, 1104, 1105, 1109, 1111, 1113, 1114, 1115, 1116,
      1119, 1120, 1121, 1122, 1129, 1130, 758, 901, 903, 910,
  ]:
    is_gdn_b = True

  if is_gdn_b:
    if "rematted_computation" in tf:
      return "5_GDN_REMAT_JAX_OPS"
    elif "transpose(jvp" in tf:
      return "6_GDN_BWD_VJP_JAX_OPS"
    else:
      return "4_GDN_FWD_JAX_OPS"

  if sf in ["ragged_sort.py", "moe.py", "ops.py"]:
    return "8_NON_GDN_MOE_AND_GMM"
  if sf in ["ring_attention_kernel.py", "attentions.py", "attention_op.py", "tokamax_ring_attention.py"]:
    return "9_NON_GDN_SPLASH_ATTN"
  return "10_NON_GDN_OFFLOAD_AND_OTHER"


def main():
  with open(ARM_A_JSON) as f:
    ops_a = json.load(f)
  with open(ARM_B_JSON) as f:
    ops_b = json.load(f)

  ops_a.sort(key=self_ms, reverse=True)
  ops_b.sort(key=self_ms, reverse=True)

  tot_a = sum(self_ms(x) for x in ops_a)
  tot_b = sum(self_ms(x) for x in ops_b)

  print("=" * 145)
  print("1. TOP 15 RAW HLO INSTRUCTIONS BY SELF TIME (SIDE-BY-SIDE, SAME STEP)")
  print("=" * 145)
  print(
      f"{'Rk':<3} | {'Arm A Op Name (xid/288123013)':<38} | {'Cat':<11} | {'Occ':>5} | {'Self ms':>8} | {'%':>5} || "
      f"{'Arm B Op Name (xid/289006766)':<38} | {'Cat':<11} | {'Occ':>5} | {'Self ms':>8} | {'%':>5}"
  )
  print("-" * 145)
  for i in range(15):
    oa = ops_a[i]
    ob = ops_b[i]
    print(
        f"{i+1:<3} | {oa['op_name'][:38]:<38} | {oa['category'][:11]:<11} | {oa['occurrences']:>5} | {self_ms(oa):>8.2f} | {self_ms(oa)/tot_a*100:>5.2f} || "
        f"{ob['op_name'][:38]:<38} | {ob['category'][:11]:<11} | {ob['occurrences']:>5} | {self_ms(ob):>8.2f} | {self_ms(ob)/tot_b*100:>5.2f}"
    )

  print("\n" + "=" * 145)
  print("2. TOP 15 LOGICAL OPERATION FAMILIES BY SELF TIME (GROUPED ACROSS CLONES/LAYERS)")
  print("=" * 145)
  fam_a = defaultdict(lambda: [0.0, 0])
  for op in ops_a:
    fam = logical_family(op)
    fam_a[fam][0] += self_ms(op)
    fam_a[fam][1] += op["occurrences"]

  fam_b = defaultdict(lambda: [0.0, 0])
  for op in ops_b:
    fam = logical_family(op)
    fam_b[fam][0] += self_ms(op)
    fam_b[fam][1] += op["occurrences"]

  top_a = sorted(fam_a.items(), key=lambda x: x[1][0], reverse=True)[:15]
  top_b = sorted(fam_b.items(), key=lambda x: x[1][0], reverse=True)[:15]

  print(
      f"{'Rk':<3} | {'Arm A Logical Family (xid/288123013)':<50} | {'Self ms':>8} | {'%':>5} || "
      f"{'Arm B Logical Family (xid/289006766)':<50} | {'Self ms':>8} | {'%':>5}"
  )
  print("-" * 145)
  for i in range(15):
    fa, (msa, _) = top_a[i]
    fb, (msb, _) = top_b[i]
    print(
        f"{i+1:<3} | {fa[:50]:<50} | {msa:8.2f} | {msa/tot_a*100:5.2f} || "
        f"{fb[:50]:<50} | {msb:8.2f} | {msb/tot_b*100:5.2f}"
    )

  print("\n" + "=" * 90)
  print("3. COMPLETE RECONCILED STEP TIME & GDN/KDA LAYER BREAKDOWN (ms / step)")
  print("=" * 90)
  sums_a = defaultdict(float)
  for op in ops_a:
    c = classify_op_a(op)
    if c == "4_AND_5_SPLIT_A":
      sums_a["4_GDN_FWD_JAX_OPS"] += self_ms(op) * 0.5
      sums_a["5_GDN_REMAT_JAX_OPS"] += self_ms(op) * 0.5
    else:
      sums_a[c] += self_ms(op)

  sums_b = defaultdict(float)
  for op in ops_b:
    c = classify_op_b(op)
    sums_b[c] += self_ms(op)

  all_cats = sorted(set(sums_a.keys()) | set(sums_b.keys()))
  print(f"{'Category':<35} | {'Arm A (ms)':>12} | {'Arm B (ms)':>12} | {'Delta (B - A)':>14}")
  print("-" * 80)
  gdn_a = 0.0
  gdn_b = 0.0
  for c in all_cats:
    va = sums_a[c]
    vb = sums_b[c]
    if c.startswith(("1_", "2_", "3_", "4_", "5_", "6_")):
      gdn_a += va
      gdn_b += vb
    print(f"{c:<35} | {va:12.2f} | {vb:12.2f} | {vb - va:+14.2f}")
  print("-" * 80)
  pallas_a = sums_a["1_GDN_FWD_PALLAS_KERNEL"] + sums_a["2_GDN_REMAT_PALLAS_KERNEL"] + sums_a["3_GDN_BWD_VJP_PALLAS_KERNEL"]
  pallas_b = sums_b["1_GDN_FWD_PALLAS_KERNEL"] + sums_b["2_GDN_REMAT_PALLAS_KERNEL"] + sums_b["3_GDN_BWD_VJP_PALLAS_KERNEL"]
  jax_a = sums_a["4_GDN_FWD_JAX_OPS"] + sums_a["5_GDN_REMAT_JAX_OPS"] + sums_a["6_GDN_BWD_VJP_JAX_OPS"]
  jax_b = sums_b["4_GDN_FWD_JAX_OPS"] + sums_b["5_GDN_REMAT_JAX_OPS"] + sums_b["6_GDN_BWD_VJP_JAX_OPS"]
  print(f"{'SUBTOTAL: GDN / KDA PALLAS KERNELS':<35} | {pallas_a:12.2f} | {pallas_b:12.2f} | {pallas_b - pallas_a:+14.2f}")
  print(f"{'SUBTOTAL: GDN / KDA NON-KERNEL JAX':<35} | {jax_a:12.2f} | {jax_b:12.2f} | {jax_b - jax_a:+14.2f}")
  print(f"{'SUBTOTAL: ALL GDN / KDA LAYER OPS':<35} | {gdn_a:12.2f} | {gdn_b:12.2f} | {gdn_b - gdn_a:+14.2f}")
  print(f"{'SUBTOTAL: ALL NON-GDN OPS':<35} | {tot_a - gdn_a:12.2f} | {tot_b - gdn_b:12.2f} | {(tot_b - gdn_b) - (tot_a - gdn_a):+14.2f}")
  print(f"{'EXACT TOTAL STEP TIME (XPROF)':<35} | {tot_a:12.2f} | {tot_b:12.2f} | {tot_b - tot_a:+14.2f}")


if __name__ == "__main__":
  main()
