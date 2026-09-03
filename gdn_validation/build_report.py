#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the KDA / Gated Delta Net engineering report as a PDF."""

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle)

OUT = os.path.expanduser("~/kda_report/kda_gdn_report.pdf")

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5b6470")
RULE = colors.HexColor("#c8ccd2")
BAND = colors.HexColor("#eef1f5")
GOOD = colors.HexColor("#1a6b3c")
BAD = colors.HexColor("#a32020")
CODEBG = colors.HexColor("#f4f5f7")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                    fontSize=16, leading=20, spaceBefore=14, spaceAfter=7,
                    textColor=INK)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                    fontSize=11.5, leading=15, spaceBefore=11, spaceAfter=5,
                    textColor=INK)
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontName="Helvetica",
                      fontSize=9.6, leading=14.2, spaceAfter=6,
                      alignment=TA_LEFT, textColor=INK)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8.4, leading=12,
                       textColor=MUTED)
CODE = ParagraphStyle("CODE", parent=BODY, fontName="Courier", fontSize=8.1,
                      leading=11, backColor=CODEBG, borderPadding=6,
                      spaceBefore=4, spaceAfter=8, textColor=INK)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8.4, leading=11,
                      spaceAfter=0)
CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")
TITLE = ParagraphStyle("TITLE", parent=H1, fontSize=21, leading=25,
                       spaceAfter=3)
SUB = ParagraphStyle("SUB", parent=BODY, fontSize=11, leading=15,
                     textColor=MUTED, spaceAfter=2)

story = []
A = story.append


def para(t, s=BODY):
  A(Paragraph(t, s))


def code(t):
  esc = (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
         .replace("\n", "<br/>").replace(" ", "&nbsp;"))
  A(Paragraph(esc, CODE))


def table(rows, widths, header=True, align=None):
  data = []
  for r_i, r in enumerate(rows):
    row = []
    for c in r:
      if isinstance(c, Paragraph):
        row.append(c)
      else:
        row.append(Paragraph(str(c), CELLB if (header and r_i == 0) else CELL))
    data.append(row)
  t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
  st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.35, RULE)]
  if header:
    st += [("BACKGROUND", (0, 0), (-1, 0), BAND),
           ("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED)]
  if align:
    for col, a in align.items():
      st.append(("ALIGN", (col, 0), (col, -1), a))
  t.setStyle(TableStyle(st))
  # Small tables must not split across a page: a lone header or a stranded row
  # reads as missing data.
  A(KeepTogether([t]) if len(data) <= 9 else t)
  A(Spacer(1, 7))


def g(s):
  return f'<font color="#1a6b3c"><b>{s}</b></font>'


def b(s):
  return f'<font color="#a32020"><b>{s}</b></font>'


# ============================================================ title
para("KDA as a Gated Delta Net", TITLE)
para("Finding a NaN in Tokamax's Kimi Delta Attention kernel, fixing it, "
     "and measuring the result", SUB)
para("Majid Taheri &nbsp;·&nbsp; 3 September 2026 &nbsp;·&nbsp; "
     "branch <font face='Courier'>Majid-Taheri/tokamax @ gdn-scalar-gate</font>",
     SMALL)
A(Spacer(1, 8))

para("<b>Summary.</b> Routing Qwen3.5's Gated Delta Net through Tokamax's KDA "
     "Pallas kernel produces <font face='Courier'>loss: nan</font> at step 0. "
     "The cause is a numerical defect in the kernel, not in the integration: "
     "to batch a matmul, the kernel splits an exponent that is always "
     "&le;&nbsp;0 into two factors, one of which overflows to infinity while "
     "the other underflows to zero. The defect is present in "
     "<font face='Courier'>openxla/tokamax</font> main today. A fix was "
     "implemented behind a new <font face='Courier'>per_channel_gate</font> "
     "flag, validated forward and backward against the reference "
     "implementation, and benchmarked. The fix is correct; it is not faster.")

para("Two performance predictions made during this work were wrong and are "
     "retracted in section 8. The measurements, not the reasoning, are what "
     "should be relied on.")

# ============================================================ 1
para("1. What went wrong", H1)
para("Qwen3.5's Gated Delta Net (GDN) and Kimi Delta Attention (KDA) implement "
     "the same chunked delta rule. They differ in one respect: KDA's forget "
     "gate carries an independent value for each key channel, while GDN's is a "
     "single scalar per head and token. An adapter was written that broadcasts "
     "GDN's scalar gate across all 128 channels and calls KDA.")
para("At reduced scale this trains correctly. At the full training "
     "configuration it does not:")
table([
    ["Run", "Step 0 loss", "Outcome"],
    ["Pure-JAX control (unchanged path)", "12.923", g("reaches step 9 at 11.901")],
    ["KDA kernel, identical config", b("nan"), b("aborted at step 0")],
], [78 * mm, 30 * mm, 60 * mm])
para("A loss of 12.923 is approximately ln(vocabulary size), which is the "
     "expected value at random initialisation. The control establishes that "
     "the model, data, sharding and optimiser are all sound, so the fault lies "
     "in the kernel path alone.", SMALL)

# ============================================================ 2
para("2. Root cause", H1)
para("In <font face='Courier'>pallas_mosaic_tpu_fwd_kernel.py</font>, the "
     "intra-chunk attention term the kernel needs is")
code("Aqk[r,t] = sum_k q[r,k] * k[t,k] * exp2(g[r,k] - g_cumsum[t,k])"
     "     for t <= r")
para("The gate accumulates negatively, so for every causal pair the exponent "
     "<font face='Courier'>g[r] - g_cumsum[t]</font> is at most zero and the "
     "whole quantity is bounded by one. It is perfectly safe as written.")
para("But the gate depends on the key channel, so it cannot be lifted out of "
     "the sum over <font face='Courier'>k</font>. To fold it into the operands "
     "and still perform a single matmul, the kernel factors the exponent "
     "around a reference row:")
code("ref_idx = BC // 2 if safe_gate else 0\n"
     "gn_ref  = g_i[:, ref_idx : ref_idx + 1, :]\n"
     "\n"
     "q_eg        = q_i  * exp2(g_i - gn_ref)          # first factor\n"
     "diff_j_safe = (gn_ref - g_cumsum) * valid_j      # POSITIVE for causal t\n"
     "k_eng_full  = k_f32 * exp2(diff_j_safe) * valid_j  # second factor")
para("The product of the two factors is bounded. The factors individually are "
     "not. The second grows with the cumulative gate drop across the block and "
     "overflows to infinity; the first underflows to zero. "
     f"{b('inf &times; 0 = NaN')}.")
para("The code comment at that site guards the anti-causal rows for exactly "
     "this reason. The causal direction has the same problem and is not "
     "guarded. The <font face='Courier'>safe_gate</font> option moves the "
     "reference row from index 0 to <font face='Courier'>BC//2</font>, which "
     "halves the exponent range rather than bounding it, and it was already "
     "enabled throughout.")
para("The backward carries the same defect in its diagonal-block term "
     "(<font face='Courier'>pallas_mosaic_tpu_bwd_kernel.py</font>), which "
     "re-centres on the block maximum so that "
     "<font face='Courier'>col_d = exp2(g_max - g_b)</font> is positive. Its "
     "off-diagonal paths are already safe, because both of their exponents are "
     "&le;&nbsp;0 by construction.")

# ============================================================ 3
para("3. Confirming the mechanism", H1)
para("<b>Test A — does the threshold match the arithmetic?</b> "
     "<font face='Courier'>BC = 16</font> and the reference sits at "
     "<font face='Courier'>BC//2 = 8</font>, so the split spans eight tokens "
     "of gate decay against a float32 <font face='Courier'>exp2</font> ceiling "
     "of 2<super>128</super>. That predicts failure once the gate falls by "
     "more than 128/8 = 16 in log2 units per token. Sweeping the gate on "
     "TPU7x:")
table([
    ["Gate scale", "min g / token", "chunk drop (log2)", "avg log2 / token", "Mosaic result"],
    ["0.25 – 21", "-0.25 to -21", "7.9 – 667", "up to 10.4", g("finite, matches reference")],
    ["30", "-30", "953", "14.9", b("NaN 5.8%")],
    ["40", "-40", "1271", "19.9", b("NaN 28.5%")],
    ["60", "-60", "1907", "29.8", b("NaN 46.4%")],
    ["100", "-100", "3178", "49.7", b("NaN 73.2%")],
], [24 * mm, 26 * mm, 30 * mm, 28 * mm, 60 * mm])
para("The edge falls between 10.4 and 14.9 log2 per token, against a predicted "
     "16. The failure is also per-head, not random: "
     "<font face='Courier'>A_log</font> is a per-head parameter, so the "
     "fraction of NaN outputs tracks the fraction of heads whose gate exceeds "
     "the threshold.", SMALL)

para("<b>Test B — is it hardware- or precision-specific?</b> No. A standalone "
     "reproduction on one chip (H=4, B=1, T=512, D=128) gives identical "
     "results in float32 and bfloat16:")
table([
    ["A_max", "min g / token", "chunk drop (log2)", "Mosaic", "XLA reference"],
    ["0.5", "-1.59", "55.8", g("finite"), g("finite")],
    ["2.0", "-6.37", "223.1", g("finite"), g("finite")],
    ["8.0", "-25.50", "892.5", g("finite"), g("finite")],
    ["16.0", "-50.99", "1784.9", b("NaN 50.0%"), g("finite")],
], [18 * mm, 27 * mm, 32 * mm, 38 * mm, 38 * mm])
para("Qwen3.5 initialises <font face='Courier'>A_log = log(Uniform(0, 16))</font> "
     "with <font face='Courier'>dt_bias = 1</font>, which is the last row. The "
     "model lands in the failing range by default. The backward behaves the "
     "same way: at A_max = 16 all five gradients are NaN in 50% of entries "
     "while the XLA reference stays finite.", SMALL)

para("<b>Test C — does the bug also affect KDA used as KDA?</b> Yes, but only "
     "in one of its two gate modes. KDA's own activation "
     "(<font face='Courier'>reference.py</font>) is:")
code("lower_bound is None ->  g = -A * softplus(g_f)        # unbounded below\n"
     "lower_bound = L     ->  g = L * sigmoid(A * g_f)      # saturates at L")
table([
    ["Mode", "A_max", "Mosaic result"],
    ["unbounded (lower_bound = None)", "0.5", g("finite")],
    ["unbounded", "8.0", b("NaN 50%")],
    ["unbounded", "16.0", b("NaN 50%")],
    ["bounded (lower_bound = -1.0)", "0.5 / 16 / 64", g("finite at every scale")],
], [58 * mm, 26 * mm, 84 * mm])
para("The bounded gate is immune by construction: it saturates at "
     "<font face='Courier'>lower_bound</font> per token however large "
     "<font face='Courier'>A</font> becomes, so the exponent can never reach "
     "the overflow range. This very likely explains why the authors never "
     "encountered the defect. It is also why "
     "<font face='Courier'>lower_bound</font> is not a usable workaround for "
     "GDN: <font face='Courier'>L&middot;sigmoid(A&middot;g)</font> is a "
     "different function from <font face='Courier'>-A&middot;softplus(g)</font>, "
     "not a clamp of it, so adopting it would change the model.", SMALL)

# ============================================================ 4
para("4. Isolating the integration from the kernel", H1)
para("Before blaming the kernel, the MaxText integration was bisected to rule "
     "out a wiring error. With the gate initialisation tamed so the forward "
     "stays inside the kernel's range, the KDA path trains correctly at every "
     "reduced configuration:")
table([
    ["Layers", "Batch / device", "Sequence", "Result over 10 steps"],
    ["4", "1", "1024", g("12.913 &rarr; 12.499")],
    ["4", "1", "4096", g("12.925 &rarr; 12.679")],
    ["4", "4", "4096", g("12.923 &rarr; 12.776")],
    ["16", "4", "4096", g("12.922 &rarr; 12.508")],
    ["48", "1", "4096", g("12.923 &rarr; 11.192")],
    ["48", "4", "4096", b("step 0 correct, step 1 nan")],
], [22 * mm, 30 * mm, 26 * mm, 90 * mm])
para("Step 0 matching the pure-JAX control to three decimals establishes that "
     "the adapter's layout, GQA expansion, gate convention, state handling, "
     "bfloat16 casting and shard_map wrapping are all correct. The remaining "
     "failure in the last row is discussed in section 9.", SMALL)

# ============================================================ 5
para("5. The fix", H1)
para("A scalar gate does not need the factorisation at all. When the decay "
     "does not depend on the key channel it comes out of the sum entirely, so "
     "the matmul can run ungated and the decay applied afterwards as a "
     "chunk-by-chunk matrix:")
code("gs    = g_cumsum[:, :, 0]                            # [MB, BT]\n"
     "diff  = gs[:, :, None] - gs[:, None, :]              # [MB, BT, BT]\n"
     "decay = exp2(where(causal, diff, 0.0))               # exponent always <= 0\n"
     "qk    = dot(q, k, contracting over k)                # gate-free matmul\n"
     "Aqk   = where(causal, qk * decay * scale, 0.0)")
para("The exponent is now formed as a single difference, masked to the causal "
     "half where it is guaranteed non-positive, so "
     "<font face='Courier'>exp2</font> lands in (0, 1] and cannot overflow. "
     "Under the mask the two formulations are algebraically identical — this "
     "is a regrouping, not an approximation. The same regrouping replaces the "
     "<font face='Courier'>g_max</font> re-centring in the backward's "
     "diagonal block.")
para("The behaviour is selected by a new op parameter, "
     "<font face='Courier'>per_channel_gate</font>, threaded through "
     "<font face='Courier'>api &rarr; base &rarr; op &rarr; forward and "
     "backward kernels</font>. When true, KDA behaves exactly as before. When "
     "false, the gate may additionally be supplied at width 1 instead of "
     "width 128, and is kept narrow through the input tensor, the "
     "<font face='Courier'>g_cumsum</font> residual, stage 3 and the "
     "backward.")

para("Changes on the branch", H2)
table([
    ["Commit", "What it does"],
    ["Add per_channel_gate", "Scalar-gate forward path"],
    ["Extend to the backward", "Same regrouping in the diagonal-block term"],
    ["Declare static", "per_channel_gate added to chunk_kda_bwd_custom's static_argnames"],
    ["Accept a width-1 gate", "Removes the 128&times; broadcast end to end"],
    ["Validation scripts", "gdn_validation/ — six harnesses plus a README"],
    ["Revert VMEM widening", "Undoes a change justified by a hypothesis later disproved"],
], [46 * mm, 122 * mm])

# ============================================================ 6
para("6. Correctness results", H1)
para("All results below are TPU v5, float32, H=4 B=1 T=512 D=128, comparing "
     "the Mosaic kernel against Tokamax's own XLA reference on byte-identical "
     "inputs. Figures are maximum relative error. A_max = 0.5 is a mild gate; "
     "A_max = 16 is Qwen3.5's actual initialisation.")

para("Forward", H2)
table([
    ["A_max", "per_channel_gate", "Mosaic", "vs XLA reference"],
    ["0.5", "True", g("finite"), "2.62e-03"],
    ["0.5", "False", g("finite"), "1.73e-03"],
    ["16.0", "True", b("NaN 50.0%"), "—"],
    ["16.0", "False", g("finite"), g("1.85e-03")],
], [20 * mm, 38 * mm, 40 * mm, 40 * mm])

para("Backward — all five gradients", H2)
table([
    ["A_max", "mode", "out", "d_query", "d_key", "d_value", "d_gate", "d_beta"],
    ["0.5", "True", "2.6e-03", "4.9e-04", "6.6e-04", "1.7e-03", "8.8e-04", "1.7e-03"],
    ["0.5", "False", "1.7e-03", "5.1e-04", "5.5e-04", "1.7e-03", "8.4e-04", "2.4e-03"],
    ["16.0", "True", b("NaN"), b("NaN"), b("NaN"), b("NaN"), b("NaN"), b("NaN")],
    ["16.0", "False", g("1.8e-03"), g("6.0e-04"), g("1.1e-03"), g("1.9e-03"),
     g("1.6e-03"), g("2.0e-03")],
], [16 * mm, 17 * mm, 21 * mm, 23 * mm, 21 * mm, 23 * mm, 21 * mm, 21 * mm])

para("Width-1 gate", H2)
para("Three configurations, all against the reference. Rows two and three must "
     "agree: the only difference between them is whether the same number is "
     "stored once or 128 times.")
table([
    ["A_max", "per_channel", "gate width", "gate size", "out", "worst gradient"],
    ["16.0", "True", "128", "1.0 MB", b("NaN 50%"), b("NaN 50%")],
    ["16.0", "False", "128", "1.0 MB", "1.8e-03", "2.0e-03"],
    ["16.0", "False", g("1"), g("0.008 MB"), g("1.8e-03"), g("2.0e-03")],
], [17 * mm, 22 * mm, 22 * mm, 24 * mm, 24 * mm, 30 * mm])
para("The two scalar rows agree to the printed digit on every quantity except "
     "<font face='Courier'>d_gate</font> (1.6e-03 against 2.0e-03). That is "
     "expected rather than a discrepancy: with a narrow gate "
     "<font face='Courier'>d_gate</font> is a sum over 128 channels rather "
     "than 128 separate values, so it rounds differently. Both are well within "
     "tolerance of the reference.", SMALL)

para("Regression check", H2)
para("In every table above, the <font face='Courier'>per_channel_gate=True</font> "
     "rows are unchanged from stock upstream, including still failing at "
     "A_max&nbsp;=&nbsp;16. That is the intended behaviour: the KDA path was "
     "not to be altered, and it was not.")

# ============================================================ 7
para("7. Performance results", H1)
para("TPU v5, H=32 B=4 T=4096 D=128, bfloat16, mild gate, median of ten "
     "iterations after warm-up. The gate is deliberately mild so both paths "
     "compute the same thing — timing a kernel that is producing NaN would be "
     "meaningless.")
table([
    ["Configuration", "gate tensor", "forward", "forward + backward"],
    ["per-channel (stock behaviour)", "256.0 MB", "2.91 ms", "9.37 ms"],
    ["scalar regrouping, gate still broadcast", "256.0 MB", "2.83 ms", "9.32 ms"],
    ["scalar regrouping, width-1 gate", g("2.0 MB"), b("3.25 ms"), b("9.81 ms")],
], [66 * mm, 30 * mm, 32 * mm, 40 * mm])
table([
    ["Comparison", "forward", "forward + backward"],
    ["regrouping alone, against stock", "1.03&times;", "1.00&times;"],
    ["width-1 gate, against stock", b("0.90&times;"), b("0.96&times;")],
], [66 * mm, 46 * mm, 56 * mm])
para("The memory objective is met exactly: 128&times; smaller, and the "
     "<font face='Courier'>g_cumsum</font> residual shrinks by the same factor "
     "(visible in the autotuning key as "
     "<font face='Courier'>g_cumsum: f32[32,4,4096,1]</font>). The time cost "
     "is real: roughly 10% in the forward and 4% end to end.", SMALL)

# ============================================================ 8
para("8. Two predictions that were wrong", H1)
para("Both are recorded because the reasoning behind them was plausible and "
     "would otherwise be repeated.")

para("<b>Prediction 1 — the regrouping would be faster.</b> The scalar path "
     "performs the same matmul FLOPs, roughly ten times fewer "
     "<font face='Courier'>exp2</font> operations, and materialises no "
     "per-sub-block temporaries. Measured: 1.03&times; forward, 1.00&times; "
     "overall. The error was reasoning about operation counts without asking "
     "which resource was actually saturated. This kernel is dominated by "
     "matmuls and memory traffic, and the change altered neither; the "
     "transcendentals were never the bottleneck.")

para("<b>Prediction 2 — the width-1 slowdown was a lane-broadcast cost.</b> On "
     "TPU the minor axis is the 128-lane dimension, so a width-1 tensor wastes "
     "the vector registers and every elementwise use appears to need a "
     "broadcast. The backward, which widens the gate in VMEM immediately after "
     "loading it, showed no regression, while the forward, which left it "
     "narrow, was 10% slower — an apparently clean natural experiment. The "
     "forward was changed to widen in VMEM as well. Measured: 3.27 ms became "
     "3.25 ms, which is noise. The hypothesis was wrong, the change was "
     "reverted, and the cost is more likely in the width-1 block itself — "
     "Mosaic probably pads it to a full tile regardless, so the HBM saving "
     "buys nothing on a kernel that is not HBM-bound.")

# ============================================================ 9
para("9. What remains open", H1)
para("<b>The full-configuration NaN is not explained.</b> At 48 layers, batch "
     "4 per device and sequence 4096, MaxText produces a correct step 0 and "
     "then <font face='Courier'>nan</font> at step 1. Ruled out by experiment, "
     "not by argument: gate magnitude (taming to A_max = 0.05 does not help), "
     "dtype, shard_map, <font face='Courier'>remat_policy=full</font>, the "
     "scoped VMEM flag, sequence length, depth alone and batch alone. In "
     "isolation at exactly those per-shard shapes all six gradients are finite "
     "and within bfloat16 noise of the reference. Only the three factors "
     "together fail.")
para("<b>The MaxText adapter has not been pointed at the fix.</b> It still "
     "broadcasts the gate to width 128 and leaves "
     "<font face='Courier'>per_channel_gate</font> at its default. Wiring it "
     "to <font face='Courier'>False</font> and rerunning is the test that "
     "decides whether this work unblocks training.")
para("<b>Nothing has been benchmarked on TPU7x.</b> Every timing here is from "
     "v5, a generation upstream explicitly declares unsupported "
     "(<font face='Courier'>supported_on()</font> requires generation &ge; 6; "
     "the validation scripts honour "
     "<font face='Courier'>KDA_BYPASS_DEVICE_CHECK=1</font>). The correctness "
     "findings are hardware-independent — the defect is a float32 exponent "
     "range problem, and it also reproduces on TPU7x through MaxText — but the "
     "timings are not transferable.")
para("<b>The defect should be reported upstream.</b> It is present in "
     "<font face='Courier'>openxla/tokamax</font> main, not only in the "
     "unmerged pull request, so anyone using KDA with an unbounded gate is "
     "exposed.")

# ============================================================ 10
para("10. Reproducing this", H1)
para("Everything, including the validation harnesses, is on the branch.")
code("git clone -b gdn-scalar-gate \\\n"
     "    https://github.com/Majid-Taheri/tokamax.git tokamax-src\n"
     "cd tokamax-src\n"
     "pip install -e . --no-deps\n"
     "pip install xprof --no-deps        # new dependency on main\n"
     "\n"
     "export KDA_BYPASS_DEVICE_CHECK=1   # only needed below TPU generation 6\n"
     "\n"
     "python3 gdn_validation/test_per_channel_gate.py       # forward\n"
     "python3 gdn_validation/test_per_channel_gate_grad.py  # gradients\n"
     "python3 gdn_validation/test_narrow_gate.py            # width-1 gate\n"
     "python3 gdn_validation/bench_per_channel_gate.py      # timings\n"
     "\n"
     "python3 gdn_validation/kda_gate_nan_repro.py   # bug on stock upstream\n"
     "python3 gdn_validation/kda_native_gate_test.py # KDA's own gate modes")
para("Each test exits 0 on success and 1 on failure. "
     "<font face='Courier'>git checkout main</font> inside the checkout gives "
     "the stock kernel for comparison.", SMALL)

para("Practical notes", H2)
table([
    ["Trap", "Consequence"],
    ["JAX_DEFAULT_MATMUL_PRECISION=highest",
     "Makes the KDA backward fail to compile in bfloat16 "
     "(&ldquo;Bad rhs type: 256, 256&rdquo;). A harness artefact, not a kernel bug."],
    ["jaxtyping binds K from the query",
     "Every annotation on the gate's path — parameters and return tuples, in "
     "both kernels — must declare its own width or a width-1 gate is rejected "
     "while the arithmetic is fine."],
    ["Upstream restricts KDA to generation &ge; 6",
     "TPU v5 needs the documented device-check bypass."],
    ["pip installs openxla main, not the antgroup branch",
     "The two differ; patches written against the wrong base will not apply."],
], [58 * mm, 110 * mm])

# ============================================================ 11
para("11. Conclusion", H1)
para("The KDA Mosaic kernel returns NaN for gates steeper than roughly 11 nats "
     "per token, in both the forward and the backward, because it splits a "
     "bounded exponent into two unbounded factors. Qwen3.5's Gated Delta Net "
     "needs about 21 nats per token at initialisation, so it fails "
     "immediately. The defect is in upstream main and also affects KDA's own "
     "unbounded-gate mode; only its bounded-gate mode is immune.")
para("Specialising the kernel for a scalar gate removes the defect by "
     "construction rather than patching around it, agrees with the reference "
     "to within 2e-03 on every output and gradient at Qwen3.5's actual gate, "
     "leaves the per-channel path untouched, and optionally reduces the gate's "
     "memory by 128&times;. It is not faster — the regrouping is neutral and "
     "the width-1 gate costs about 4% end to end on v5.")
para("The value of this work is that it makes the configuration run at all. "
     "Whether it makes training work is a separate question, and answering it "
     "requires pointing the MaxText adapter at the new flag and repeating the "
     "48-layer run.")

A(Spacer(1, 10))
para("Branch: <font face='Courier'>github.com/Majid-Taheri/tokamax</font>, "
     "branch <font face='Courier'>gdn-scalar-gate</font>, rebased on "
     "<font face='Courier'>openxla/tokamax</font> main. Measurements dated "
     "3 September 2026.", SMALL)


def footer(canvas, doc):
  canvas.saveState()
  canvas.setFont("Helvetica", 7.6)
  canvas.setFillColor(MUTED)
  canvas.drawString(20 * mm, 12 * mm, "KDA as a Gated Delta Net — findings and measurements")
  canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"{doc.page}")
  canvas.setStrokeColor(RULE)
  canvas.setLineWidth(0.4)
  canvas.line(20 * mm, 15 * mm, A4[0] - 20 * mm, 15 * mm)
  canvas.restoreState()


doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                      topMargin=18 * mm, bottomMargin=20 * mm,
                      title="KDA as a Gated Delta Net",
                      author="Majid Taheri")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")
doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])
doc.build(story)
print("wrote", OUT)
