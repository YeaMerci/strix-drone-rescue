# On-Board Inference Boards for a Real-Time Aerial SAR Drone: Jetson Nano vs Coral Dev Board vs Raspberry Pi 4 + Intel NCS2 vs Jetson Xavier NX

> Source report returned from a Claude Deep Research pass (prompt in
> [`hardware-deep-research-prompt.md`](hardware-deep-research-prompt.md)).
> The chapter [`../../04-hardware-and-edge.md`](../../04-hardware-and-edge.md) is
> distilled from this document; figures below are cited there.

## TL;DR
- **The Jetson Nano choice is defensible on cost, power and ecosystem grounds but is not the performance-optimal pick: for YOLOv8n at 1280 px (which small-object detection at 40–50 m altitude actually needs), the Nano delivers only single-digit FPS, while the Jetson Xavier NX — the only one of the four with real 1280 px + INT8 headroom — is roughly 3–5× faster for a ~0.5-minute-larger endurance penalty.**
- **The Coral Dev Board and Raspberry Pi 4 + NCS2 are the two weakest fits and should be rejected: Coral's Edge TPU cannot cleanly map the YOLOv8 head (SiLU crashes the compiler; at 640 px most ops fall back to CPU, collapsing to ~2.6–4 FPS), and the NCS2 is discontinued (last order 28 Feb 2022) and only reaches single-digit YOLOv8n FPS in FP16.**
- **Board choice barely moves endurance — all four cost ~13–18% of hover endurance in our model, a spread of under one minute — so the decision should be driven by throughput/accuracy at the required input resolution, where Xavier NX >> Nano >> NCS2 ≈ Coral.**

## Key Findings
1. **Endurance is dominated by the airframe, not the board.** Under the stated battery/hover assumptions, adding any of the four modules costs 13–18% of hover endurance; the difference *between boards* is only ~0.5 min. SWaP-driven board selection is therefore a weak lever, and real-time capability should dominate the decision.
2. **Only the Xavier NX comfortably clears the ≥10 FPS target with headroom for 1280 px.** Nano is marginal at 640 px (~15–19 FPS inference-only, less end-to-end) and slow at 1280 px. Coral and NCS2 fail the target at 640 px.
3. **Coral is architecturally mismatched to YOLOv8n.** The Edge TPU is INT8-TFLite-only; YOLOv8's SiLU activation crashes the Edge TPU compiler, and at 640 px the detection head forces CPU fallback. It is competitive only at ≤320 px, which is fatal for detecting few-pixel humans from altitude.
4. **The NCS2 path is a dead end in 2026.** Myriad X is FP16-only (accuracy-favorable but no INT8 speedup), throughput is low, and Intel discontinued the product.
5. **All four are effectively legacy hardware in 2026.** Jetson Nano production modules are extended only to Jan 2027 (JetPack 4 EOL Nov 2024); Xavier NX devkit is EOL (modules to Jan 2028); Coral has had no updates since 2021; NCS2 is discontinued. A 2026 greenfield build should use a Jetson Orin Nano Super or a Raspberry Pi 5 + AI HAT+ (Hailo).

## Consolidated Comparison Table

| Metric | Jetson Nano (4GB) | Coral Dev Board | RPi 4 (4GB) + NCS2 | Jetson Xavier NX |
|---|---|---|---|---|
| AI compute | 472 GFLOPS FP16 (~0.5 TOPS) | 4 TOPS INT8 (Edge TPU) | ~1–4 TOPS FP16 (Myriad X) | 21 TOPS INT8 @15W / 14 @10W |
| GPU/accelerator | 128-core Maxwell | Edge TPU ASIC | 16-SHAVE Myriad X VPU | 384-core Volta + 2× DLA |
| RAM | 4 GB LPDDR4, 25.6 GB/s | 1/4 GB LPDDR4 | 4 GB LPDDR4 (Pi4) | 8 GB LPDDR4x, 51.2 GB/s |
| Precision | FP16 only (no INT8 accel) | INT8 only | FP16 / FP32 only | FP32/FP16/INT8 |
| YOLOv8n native? | Yes (TensorRT, convert) | No — head surgery + INT8 QAT, SiLU crashes | Convert to OpenVINO IR FP16 | Yes (TensorRT, convert) |
| YOLOv8n FPS @640 | ~15–19 (TRT FP16, inference-only) | ~2.6–4 (CPU fallback) | ~5–8 (proxy/extrapolated) | ~45–55 (extrapolated) |
| YOLOv8n FPS @1280 | ~3–5 (extrapolated) | fails (max ~320 px) | ~1–2 (extrapolated) | ~15–20 (extrapolated) |
| Board draw under inference | ~10 W | ~5 W (TPU 2 W) | ~6.5 W (Pi 5 W + NCS2 1.5 W) | ~15 W (15 W mode) |
| Flight-ready mass | ~140 g (module+HS+carrier) | ~130 g (est.) | ~134 g (Pi+NCS2+cable) | ~170 g (module+HS/fan+carrier) |
| Cooling | Passive heatsink (fan for 10W) | Heatsink + fan | Passive (Pi); NCS2 passive | Active fan required |
| 2023 price | $149 devkit | ~$130–150 | ~$55 Pi + ~$70–100 NCS2 | ~$399 module |
| 2026 status | Scarce/scalped, modules to Jan 2027 | Abandoned since 2021, low stock | NCS2 discontinued (EOL) | Devkit EOL; modules to Jan 2028 |

## Flight-Time Impact Model

### Assumptions (stated explicitly)
- Battery: 5000 mAh 4S LiPo = 14.8 V × 5.0 Ah = **74 Wh**; usable at 80% DoD = **59.2 Wh**.
- Base airframe (everything except compute module) all-up mass **M₀ = 1500 g**, hover power **P_hover(M₀) = 350 W**.
- **Hover power vs mass:** momentum (actuator-disk) theory gives P_hover ∝ m^1.5. Calibrated k = 350 / 1500^1.5 = 0.006025 W·g^−1.5, so P_hover(M) = 0.006025 · M^1.5.
- Total draw = P_hover(M₀+Δm) + P_board. Endurance T(min) = 59.2 Wh / P_total(W) × 60.

### Results

| Config | Δmass (g) | Board draw (W) | Total mass (g) | Hover power (W) | Total draw (W) | Endurance (min) | Loss vs no-payload |
|---|---|---|---|---|---|---|---|
| No compute payload | 0 | 0 | 1500 | 350.0 | 350.0 | 10.15 | — |
| Coral Dev Board | 130 | 5.0 | 1630 | 396.4 | 401.4 | 8.85 | 12.8% |
| RPi4 + NCS2 | 134 | 6.5 | 1634 | 397.9 | 404.4 | 8.78 | 13.5% |
| Jetson Nano | 140 | 10.0 | 1640 | 400.1 | 410.1 | 8.66 | 14.7% |
| Jetson Xavier NX | 170 | 15.0 | 1670 | 411.2 | 426.2 | 8.33 | 17.9% |

Endurance spread across all four boards is **0.52 min (~31 s)** — a weak discriminator.

## Weighted Trade-Off (YOLOv8n @640 px)

| Board | FPS@640 | FPS/W | FPS/gram | FPS/$ |
|---|---|---|---|---|
| Jetson Nano | 19 | 1.90 | 0.136 | 0.128 ($149) |
| Coral Dev Board | 4 | 0.80 | 0.031 | 0.031 ($130) |
| RPi4 + NCS2 | 6 | 0.92 | 0.045 | 0.048 ($125) |
| Jetson Xavier NX | 50 | 3.33 | 0.294 | 0.125 ($399) |

### Composite score (SAR-weighted: Throughput 40 / Endurance 25 / Accuracy@altitude 20 / Cost 15)

| Board | Throughput | Endurance | Accuracy@altitude | Cost/avail | **Composite** |
|---|---|---|---|---|---|
| Jetson Xavier NX | 9 | 6 | 9 | 4 | **7.5** |
| Jetson Nano | 5 | 7 | 6 | 6 | **5.85** |
| RPi4 + NCS2 | 2 | 8 | 5 | 5 | **4.55** |
| Coral Dev Board | 2 | 8 | 3 | 5 | **4.15** |

**Recommendation ordering (these four):** Xavier NX > Jetson Nano > RPi4+NCS2 > Coral Dev Board.

## Small-Object Detection at 40–50 m: 640 vs 1280 px and SAHI
At 40–50 m a standing adult subtends only a handful of pixels; **input resolution is the dominant accuracy lever**. 640 → 1280 px is ~4× the compute; SAHI/tiled inference (Akyon et al., arXiv:2202.06934, +5–7 AP) multiplies it 4–9×. Only the Xavier NX (or a modern successor) can absorb this near real-time. This *amplifies* the throughput gap in the Nano's disfavor for high-altitude small-object work.

## 2026 Reality Check: EOL and Successors
- **Jetson Orin Nano Super ($249):** 67 INT8 TOPS, 102 GB/s, 8 GB LPDDR5, native INT8 + sparsity, CUDA/TensorRT-compatible (ports from Nano), supported to 2032. Runs YOLOv8n well past 100 FPS. **Best modern port.**
- **Raspberry Pi 5 + AI HAT+ (Hailo-8L/8, 13–26 TOPS, $70–110; Hailo-10H 40 TOPS ~$130, Jan 2026):** Seeed benchmark YOLOv8s @640 = 127.85 FPS @~12 W. Best low-power/low-mass alternative; less mature toolchain than TensorRT.
- **Avoid** Coral and NCS2 for new YOLOv8-class designs.

## Caveats
- Nano YOLOv8n (~15–19 FPS) and Coral (~2.6–4 FPS @640) are community-measured; **Xavier NX (~45–55 FPS) and RPi4+NCS2 (~5–8 FPS) are extrapolations** from YOLOv5s/YOLOv3-tiny proxies. Inference-only figures exclude capture/pre/post, which can roughly halve end-to-end FPS.
- Thermal throttling in an enclosed payload bay is a real risk for the Xavier NX (active airflow) and the 10 W Nano.
- Flight-time model gives robust *percentages* but assumption-dependent *absolute* minutes; recompute k for the real airframe.
- Coral mass (~130 g) is an estimate; INT8 mAP deltas are calibration-dependent (YOLOv8n: ~1.5% mAP50 careful PTQ up to ~6.5 pts naive).
