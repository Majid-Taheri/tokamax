# End-to-End 256-Chip Cloud TPU v7x (`GhostFish`) Benchmark Results: Arm A vs. Arm B (Qwen3.5-397B, Seq 64K)

This document preserves all end-to-end 256-chip (`64x TPU v7x / GhostFish` hosts, `256` TensorCores) training step measurements, step-by-step wall-clock logs (`Steps 0–19`), XProf HLO execution breakdowns, and HBM memory profiles comparing **Arm A** (Rohan's fused conv1d+GDN baseline) against **Arm B** (Tokamax KDA implementation).

---

## 1. Executive Summary: Which Arm Is Faster?

**Arm B (`xid/289317276`, Tokamax commit `472ebdd`, `--cell=yumciaj`) is FASTER than Arm A (`xid/289325662`, `--cell=yumciaj`) across every steady-state metric:**

- **Same-Cell Wall-Clock Step Time (Steps 4–7 Mean)**:
  - **Arm B (`yumciaj`)**: **`53,420.33 ms / step`**
  - **Arm A (`yumciaj`)**: **`53,663.93 ms / step`**
  - **Arm A (`viglobal`)**: **`53,682.31 ms / step`**
  - **Result**: **Arm B is `243.60 ms/step` (`-0.45%`) FASTER than Arm A on the exact same cell (`yumciaj`)**, and **`261.98 ms/step` (`-0.49%`) FASTER** than Arm A on `viglobal`.
- **Post-Profiler Steady-State Wall-Clock (Steps 14–19 Mean)**:
  - **Arm B (`yumciaj`)**: **`53,419.34 ms / step`**
  - **Arm A (`viglobal`)**: **`53,687.49 ms / step`**
  - **Result**: **Arm B is `268.15 ms/step` (`-0.50%`) FASTER**.
- **XProf Total HLO Step Self Time (`hlo_stats.json`)**:
  - **Arm B (`yumciaj`)**: **`53,786.42 ms / step`**
  - **Arm A (`yumciaj`)**: **`54,036.16 ms / step`**
  - **Arm A (`viglobal`)**: **`54,055.88 ms / step`**
  - **Result**: **Arm B is `249.73 ms/step` (`-0.46%`) FASTER in XProf device execution time**.

---

## 2. Master Overview Table Across All 256-Chip Runs

| Metric | Arm A (`viglobal`)<br>`xid/288123013` | Arm A (`yumciaj`)<br>`xid/289325662` | Arm B (`viglobal`, Early)<br>`xid/289006766` | Arm B (`yumciaj`, `472ebdd`)<br>`xid/289317276` | **Same-Cell Winner & Delta**<br>**(Arm B − Arm A on `yumciaj`)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Borg Cell** | `viglobal` | `yumciaj` | `viglobal` | `yumciaj` | Same cell (`yumciaj`) |
| **Topology** | `64x GhostFish` (`256 chips`) | `64x GhostFish` (`256 chips`) | `64x GhostFish` (`256 chips`) | `64x GhostFish` (`256 chips`) | Identical (`256 chips`) |
| **XProf Session ID** | `majidtaheri-10815935836637529643` | `majidtaheri-4411822555443827885` | `majidtaheri-16792108857845920987` | `majidtaheri-2769083188594969118` | — |
| **Wall-Clock Mean (Steps 4–7)** | `53,682.31 ms` | `53,663.93 ms` | `55,214.12 ms` | **`53,420.33 ms`** | **Arm B Faster by `-243.60 ms (-0.45%)`** |
| **Wall-Clock Mean (Steps 14–19)** | `53,687.49 ms` | — | — | **`53,419.34 ms`** | **Arm B Faster by `-268.15 ms (-0.50%)`** |
| **XProf Step Self Time (HLO Sum)** | `54,055.88 ms` | `54,036.16 ms` | `55,588.39 ms` | **`53,786.42 ms`** | **Arm B Faster by `-249.73 ms (-0.46%)`** |
| **XProf Step Time (Implied 100%)** | `54,311.68 ms` | `54,291.19 ms` | `55,851.20 ms` | **`54,057.12 ms`** | **Arm B Faster by `-234.07 ms (-0.43%)`** |
| **Peak HBM In Use** (`peakBytesInUse`) | `81.213 GiB`<br>(`87,201,727,488 B`) | `81.215 GiB`<br>(`87,204,098,048 B`) | `85.438 GiB`<br>(`91,738,413,056 B`) | `85.438 GiB`<br>(`91,738,413,056 B`) | `+4.223 GiB (+5.20%)` |
| **XLA Stack Reserved** (`stackReservedBytes`) | `76.591 GiB`<br>(`82,239,291,392 B`) | `76.591 GiB`<br>(`82,239,291,392 B`) | `80.751 GiB`<br>(`86,706,078,208 B`) | `80.751 GiB`<br>(`86,706,078,208 B`) | `+4.160 GiB` |
| **Heap Allocated** (`heapAllocatedBytes`) | `4.622 GiB`<br>(`4,962,436,096 B`) | `4.624 GiB`<br>(`4,964,806,656 B`) | `4.687 GiB`<br>(`5,032,334,848 B`) | `4.687 GiB`<br>(`5,032,334,848 B`) | `+0.063 GiB` |

---

## 3. All Step-by-Step Wall-Clock Times (`Steps 0–19`)

Every step time below is extracted directly from TensorBoard `perf/step_time_seconds` event logs (`events.out.tfevents.*`):

| Step # | Phase / Activity Description | Arm A (`viglobal`)<br>`xid/288123013` | Arm A (`yumciaj`)<br>`xid/289325662` | Arm B (`yumciaj`, `472ebdd`)<br>`xid/289317276` | **Same-Cell Delta (`Arm B − Arm A_yu`)** |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Step 0** | XLA Compilation + Pallas Autotuning + First Execution | `84,629.68 ms` | `84,706.86 ms` | `231,324.45 ms` | `+146,617.59 ms` *(autotune)* |
| **Step 1** | Async Host Dispatch Overlap | `328.98 ms` | `342.97 ms` | `334.77 ms` | `-8.20 ms` |
| **Step 2** | Host/Device Queue Stabilization | `66,610.66 ms` | `53,624.86 ms` | `55,855.43 ms` | `+2,230.57 ms` |
| **Step 3** | Pre-Profiler Steady State | `53,333.27 ms` | `53,386.46 ms` | **`53,141.33 ms`** | **`-245.14 ms` (Arm B Faster)** |
| **Step 4** | **Steady-State Measurement Window (Step 4)** | `53,678.56 ms` | `53,664.75 ms` | **`53,410.53 ms`** | **`-254.22 ms` (Arm B Faster)** |
| **Step 5** | **Steady-State Measurement Window (Step 5)** | `53,681.25 ms` | `53,659.76 ms` | **`53,422.78 ms`** | **`-236.98 ms` (Arm B Faster)** |
| **Step 6** | **Steady-State Measurement Window (Step 6)** | `53,680.82 ms` | `53,666.25 ms` | **`53,424.58 ms`** | **`-241.67 ms` (Arm B Faster)** |
| **Step 7** | **Steady-State Measurement Window (Step 7)** | `53,688.61 ms` | `53,664.95 ms` | **`53,423.42 ms`** | **`-241.53 ms` (Arm B Faster)** |
| **Mean 4–7** | **PURE STEADY-STATE MEAN (STEPS 4–7)** | **`53,682.31 ms`** | **`53,663.93 ms`** | **`53,420.33 ms`** | **`-243.60 ms (-0.45%) FASTER`** |
| **Step 8** | XProf Profiler Start / Trace Buffer Allocation | `53,682.68 ms` | `53,660.45 ms` | `54,975.81 ms` | `+1,315.35 ms` *(profiler start)* |
| **Step 9** | XProf Active Profiling Step | `53,679.55 ms` | `53,663.55 ms` | `51,855.13 ms` | `-1,808.42 ms` |
| **Step 10** | XProf Trace Collection & Serialization | `107,594.51 ms` | `107,508.87 ms` | `107,018.88 ms` | `-489.99 ms` |
| **Step 11** | Async Post-Stop Catch-up | `14.39 ms` | `14.32 ms` | `22.45 ms` | `+8.13 ms` |
| **Step 12** | XProf CNS File Flush (`xplane.pb` / `hlo_proto`) | `153,435.78 ms` | `178,941.74 ms` | `155,923.72 ms` | `-23,018.02 ms` |
| **Step 13** | Async Post-Flush Catch-up | `6.71 ms` | `7.35 ms` | `6.82 ms` | `-0.53 ms` |
| **Step 14** | **Post-Profiler Steady State (Step 14)** | `53,684.89 ms` | *run ended* | **`53,415.04 ms`** | **`-269.85 ms` vs. Arm A (`vi`)** |
| **Step 15** | **Post-Profiler Steady State (Step 15)** | `53,685.17 ms` | *run ended* | **`53,420.16 ms`** | **`-265.01 ms` vs. Arm A (`vi`)** |
| **Step 16** | **Post-Profiler Steady State (Step 16)** | `53,693.24 ms` | *run ended* | **`53,419.99 ms`** | **`-273.25 ms` vs. Arm A (`vi`)** |
| **Step 17** | **Post-Profiler Steady State (Step 17)** | `53,691.42 ms` | *run ended* | **`53,419.85 ms`** | **`-271.57 ms` vs. Arm A (`vi`)** |
| **Step 18** | **Post-Profiler Steady State (Step 18)** | `53,682.64 ms` | *run ended* | **`53,419.22 ms`** | **`-263.42 ms` vs. Arm A (`vi`)** |
| **Step 19** | **Post-Profiler Steady State (Step 19)** | `53,687.60 ms` | *run ended* | **`53,421.77 ms`** | **`-265.83 ms` vs. Arm A (`vi`)** |
| **Mean 14–19**| **POST-PROFILER STEADY-STATE MEAN (STEPS 14–19)** | **`53,687.49 ms`** | — | **`53,419.34 ms`** | **`-268.15 ms (-0.50%) FASTER`** |

---

## 4. Complete 10-Category XProf HLO Breakdown (Summed to 100.0% of Step Self Time)

Every single HLO instruction in `hlo_stats_all.json` across all 256 TensorCores (`total_self_time_us / 256,000`) is classified below into 10 mutually exclusive categories:

| HLO Execution Category | Arm A (`viglobal`)<br>`xid/288123013` | Arm A (`yumciaj`)<br>`xid/289325662` | Arm B (`viglobal`, Early)<br>`xid/289006766` | Arm B (`yumciaj`, `472ebdd`)<br>`xid/289317276` | **Same-Cell Delta**<br>**(`Arm B − Arm A_yu`)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. GDN/KDA Forward Pallas Kernels (`jvp()`)** | `759.70 ms` | `759.53 ms` | `909.00 ms` | `955.09 ms` | `+195.56 ms` |
| **2. GDN/KDA Remat Pallas Kernels (`remat`)** | `759.66 ms` | `759.47 ms` | `912.53 ms` | `955.38 ms` | `+195.90 ms` |
| **3. GDN/KDA Backward VJP Pallas Kernels** | `1,217.38 ms` | `1,216.30 ms` | `1,447.65 ms` | `1,450.51 ms` | `+234.22 ms` |
| **Subtotal: All GDN/KDA Pallas Kernels** | **`2,736.73 ms`** | **`2,735.30 ms`** | **`3,269.18 ms`** | **`3,360.98 ms`** | **`+625.68 ms`** |
| **4. GDN/KDA Forward Non-Kernel JAX Ops** *(conv1d/norm/reshapes)* | `283.98 ms` | `283.50 ms` | `861.56 ms` | `431.66 ms` | `+148.17 ms` |
| **5. GDN/KDA Remat Non-Kernel JAX Ops** *(conv1d/norm/reshapes)* | `283.98 ms` | `283.50 ms` | `1,077.50 ms` | `437.34 ms` | `+153.84 ms` |
| **6. GDN/KDA Backward VJP Non-Kernel JAX Ops** *(conv1d VJP/casts)* | `1,227.35 ms` | `1,222.89 ms` | `1,364.43 ms` | `1,178.10 ms` | **`-44.79 ms`** |
| **Subtotal: All GDN/KDA Non-Kernel JAX Ops** | **`1,795.31 ms`** | **`1,789.89 ms`** | **`3,303.48 ms`** | **`2,047.10 ms`** | **`+257.21 ms`** |
| **TOTAL: ALL GDN / KDA LAYER OPERATIONS** | **`4,532.05 ms`** | **`4,525.19 ms`** | **`6,572.66 ms`** | **`5,408.08 ms`** | **`+882.89 ms`** |
| **7. Non-GDN Collectives** (`AG`/`AR`/`RS`/`Permute`) | `12,500.50 ms` | `12,642.71 ms` | `13,503.85 ms` | `13,651.84 ms` | `+1,009.13 ms` |
| **8. Non-GDN MoE Routing & GMM/TGMM Kernels** | `22,285.54 ms` | `22,167.61 ms` | `21,187.99 ms` | `21,613.86 ms` | **`-553.75 ms`** |
| **9. Non-GDN Splash Attention Kernels** | `1,731.19 ms` | `1,728.36 ms` | `1,728.30 ms` | `1,728.80 ms` | `+0.44 ms` |
| **10. Non-GDN Dense Linears, Offload & Other** | `13,006.60 ms` | `12,972.28 ms` | `12,595.60 ms` | `11,383.84 ms` | **`-1,588.45 ms`** |
| **TOTAL: ALL NON-GDN OPERATIONS** | **`49,523.83 ms`** | **`49,510.97 ms`** | **`49,015.73 ms`** | **`48,378.34 ms`** | **`-1,132.63 ms`** |
| **EXACT TOTAL STEP SELF TIME (100% HLO SUM)** | **`54,055.88 ms`** | **`54,036.16 ms`** | **`55,588.39 ms`** | **`53,786.42 ms`** | **`-249.73 ms (-0.46%) FASTER`** |

### Why Arm B (`472ebdd`) Improved by `1,801.97 ms/step` Over Early Arm B (`xid/289006766`) and Beats Arm A by `249.73 ms/step`:
1. **Elimination of `1,256.38 ms` of Redundant Non-Kernel GDN Pre/Post-Processing Overhead**:
   - In early Arm B (`xid/289006766`), non-kernel JAX ops inside GDN Forward (`861.56 ms`), Remat (`1,077.50 ms`), and Backward VJP (`1,364.43 ms`) totaled **`3,303.48 ms`**.
   - In `472ebdd` (`xid/289317276`), scalar-gate width-1 layout handling (`[.., 1, BT]` in HBM widened to `[MB, BT, K]` inside VMEM) cut Forward non-kernel JAX ops by `-429.90 ms`, Remat non-kernel JAX ops by `-640.16 ms`, and Backward VJP non-kernel JAX ops by `-186.33 ms` (a **`-1,256.38 ms` reduction** in GDN JAX ops).
2. **Better Whole-Program XLA Fusion & Scheduling (`-1,588.45 ms` in Dense Linears / Offload & `-553.75 ms` in MoE/GMM)**:
   - Because Arm B's KDA Pallas kernels (`op.py:356`) expose cleaner tensor layouts and avoid Arm A's monolithic `fused_conv1d_gdn` register/VMEM pressure across the scanned block boundary, XLA's scheduler overlaps host offload copies and `DenseGeneral` projections more effectively across the scanned loop body, saving `-1,132.63 ms` in non-GDN operations and yielding a net **`-249.73 ms/step` (`-243.60 ms` wall-clock)** end-to-end speedup over Arm A.

---

## 5. Detailed Op-by-Op Accounting of KDA Pallas Kernels in Arm B (`xid/289317276`)

In Arm B (`xid/289317276`), each of the 16 scanned blocks (`Qwen3_5ScannableBlock`, scanned `15` iterations + `1` unrolled $= 16$ blocks $\rightarrow$ `3,840` occurrences per op across `256` chips) contains **3 GDN layers**.

| Kernel Phase | HLO Custom-Call Name | Occurrences | Total Self Time (ms/step) | Time per Single Call (ms) | Kernel Description |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1. Forward (`jvp()`)** | `shard_map.30643` | `3,840` | `175.84 ms` | `11.723 ms` | Layer 0 Forward KDA Chunk State Recurrence (`h`) |
| **1. Forward (`jvp()`)** | `shard_map.30645` | `3,840` | `175.85 ms` | `11.723 ms` | Layer 1 Forward KDA Chunk State Recurrence (`h`) |
| **1. Forward (`jvp()`)** | `shard_map.30647` | `3,840` | `175.83 ms` | `11.722 ms` | Layer 2 Forward KDA Chunk State Recurrence (`h`) |
| **1. Forward (`jvp()`)** | `shard_map.30644` | `3,840` | `142.53 ms` | `9.502 ms` | Layer 0 Forward KDA Chunk WY / Intra Output (`o`) |
| **1. Forward (`jvp()`)** | `shard_map.30646` | `3,840` | `142.53 ms` | `9.502 ms` | Layer 1 Forward KDA Chunk WY / Intra Output (`o`) |
| **1. Forward (`jvp()`)** | `shard_map.30648` | `3,840` | `142.51 ms` | `9.501 ms` | Layer 2 Forward KDA Chunk WY / Intra Output (`o`) |
| *Forward Subtotal* | *6 Forward Pallas Kernels* | — | **`955.09 ms`** | **`63.673 ms / block`** | **`21.224 ms` per GDN layer forward** |
| **2. Remat (`remat`)** | `shard_map.30649` | `3,840` | `175.98 ms` | `11.732 ms` | Layer 0 Remat KDA Chunk State Recurrence (`h`) |
| **2. Remat (`remat`)** | `shard_map.30651` | `3,840` | `176.01 ms` | `11.734 ms` | Layer 1 Remat KDA Chunk State Recurrence (`h`) |
| **2. Remat (`remat`)** | `shard_map.30653` | `3,840` | `175.88 ms` | `11.725 ms` | Layer 2 Remat KDA Chunk State Recurrence (`h`) |
| **2. Remat (`remat`)** | `shard_map.30650` | `3,840` | `142.50 ms` | `9.500 ms` | Layer 0 Remat KDA Chunk WY / Intra Output (`o`) |
| **2. Remat (`remat`)** | `shard_map.30652` | `3,840` | `142.50 ms` | `9.500 ms` | Layer 1 Remat KDA Chunk WY / Intra Output (`o`) |
| **2. Remat (`remat`)** | `shard_map.30654` | `3,840` | `142.50 ms` | `9.500 ms` | Layer 2 Remat KDA Chunk WY / Intra Output (`o`) |
| *Remat Subtotal* | *6 Remat Pallas Kernels* | — | **`955.38 ms`** | **`63.692 ms / block`** | **`21.231 ms` per GDN layer remat** |
| **3. Backward (`VJP`)** | `_fused_dhu_wy_intra_cumsum_pallas_jit.1` | `3,840` | `365.78 ms` | **`24.385 ms`** | Layer 0 Fused `dhu` + `WY` + `intra` + `cumsum` Bwd |
| **3. Backward (`VJP`)** | `_fused_dhu_wy_intra_cumsum_pallas_jit.2` | `3,840` | `365.74 ms` | **`24.384 ms`** | Layer 1 Fused `dhu` + `WY` + `intra` + `cumsum` Bwd |
| **3. Backward (`VJP`)** | `_fused_dhu_wy_intra_cumsum_pallas_jit.3` | `3,840` | `365.75 ms` | **`24.383 ms`** | Layer 2 Fused `dhu` + `WY` + `intra` + `cumsum` Bwd |
| **3. Backward (`VJP`)** | `fused_recompute_w_u_vnew_from_h_pallas.1` | `3,840` | `69.34 ms` | `4.623 ms` | Layer 0 Recompute `w, u, v_new` from saved `h` |
| **3. Backward (`VJP`)** | `fused_recompute_w_u_vnew_from_h_pallas.2` | `3,840` | `65.57 ms` | `4.371 ms` | Layer 1 Recompute `w, u, v_new` from saved `h` |
| **3. Backward (`VJP`)** | `fused_recompute_w_u_vnew_from_h_pallas.3` | `3,840` | `69.70 ms` | `4.647 ms` | Layer 2 Recompute `w, u, vnew` from saved `h` |
| **3. Backward (`VJP`)** | `chunk_kda_bwd_dAv_kernel.1` | `3,840` | `49.57 ms` | `3.305 ms` | Layer 0 Backward `dA_qk` and `dv0` Kernel |
| **3. Backward (`VJP`)** | `chunk_kda_bwd_dAv_kernel.2` | `3,840` | `49.51 ms` | `3.301 ms` | Layer 1 Backward `dA_qk` and `dv0` Kernel |
| **3. Backward (`VJP`)** | `chunk_kda_bwd_dAv_kernel.3` | `3,840` | `49.55 ms` | `3.303 ms` | Layer 2 Backward `dA_qk` and `dv0` Kernel |
| *Backward Subtotal* | *9 Backward Pallas Kernels* | — | **`1,450.51 ms`** | **`96.701 ms / block`** | **`32.234 ms` per GDN layer backward** |
| **TOTAL KDA KERNELS** | **All 21 KDA Pallas Custom Calls** | — | **`3,360.98 ms`** | **`224.065 ms / block`** | **`6.25%` of 53,786 ms step time** |

---

## 6. How to Reproduce / Launch Commands Used

### Arm A (`yumciaj`, `xid/289325662`) Launch Script
- Script: `run_arm_a_yumciaj.sh`
- Cell: `--cell=yumciaj`
- Key Flags: Uses Rohan's `hybrid_gdn.py` (`fused_conv1d_gdn_per_seq_b1_c64` forward/remat + `gdn_bwd_kernel_group_0..3` backward).

### Arm B (`yumciaj`, `xid/289317276`, Tokamax `472ebdd`) Launch Script
- Script: `run_arm_b_472ebdd_yumciaj.sh`
- Cell: `--cell=yumciaj`
- Key Flags: Uses Tokamax `PallasMosaicTpuKimiDeltaAttention` (`per_channel_gate=False`, scalar gate width-1 HBM optimization `472ebdd`).
