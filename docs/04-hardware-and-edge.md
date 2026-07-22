# On-board compute & edge deployment

## Why on-board inference

Offloading inference to a ground station or cloud endpoint is not viable for aerial SAR. The three structural reasons:

**Latency.** A round-trip over a drone telemetry link — even a low-latency 5.8 GHz video link — adds tens to hundreds of milliseconds of network transit on top of encode/decode overhead. At 40–50 m altitude and a drone moving at 5–10 m/s, a 100 ms latency spike translates to 0.5–1.0 m of positional drift between the frame captured and the alert raised. For a person-sized target (~0.5 × 0.3 m apparent at that altitude), that slip degrades detection reliability and bounding-box accuracy enough to matter operationally.

**Connectivity dead-zones.** Wilderness SAR missions — forests, mountain terrain, flood plains — are precisely the environments where cellular and Wi-Fi coverage is absent or intermittent. A ground-station pipeline that depends on a reliable uplink fails exactly where the system is most needed. On-board compute eliminates the dependency entirely.

**Autonomy.** Strix is designed to operate with minimal operator interaction: the drone follows a predefined search grid, flags detections, and returns coordinates. That autonomy loop — perceive → decide → act — must close on the aircraft. Externalising the perceive step breaks the loop.

The compute target follows directly: run YOLOv8-nano (3.0 M parameters, 6.3 MB weights, ~8.6 ms/image on a Tesla P100 at 1280 px input) at ≥10 FPS on the edge board, without drawing enough power to gut flight time below mission-useful endurance.

## Candidate boards

Four boards were evaluated against this target. Each takes a distinct approach to neural inference on constrained hardware:

- **NVIDIA Jetson Nano (4 GB)** — 128-core Maxwell GPU; inference via CUDA + TensorRT. FP16 only — Maxwell has no INT8 acceleration. The project's final hardware choice.
- **Google Coral Dev Board** — dedicated Edge TPU co-processor; requires INT8-only models compiled with the Edge TPU compiler. No floating-point inference path on the TPU itself.
- **Raspberry Pi 4 (4 GB) + Intel Neural Compute Stick 2 (Myriad X VPU)** — host CPU (Cortex-A72) offloads inference to the NCS2 via OpenVINO; model must be converted to OpenVINO IR format.
- **NVIDIA Jetson Xavier NX** — larger Volta GPU (384 CUDA cores) with 48 Tensor Cores; CUDA + TensorRT, higher theoretical throughput and higher peak power draw.

## System pipeline

The on-board processing pipeline — from camera capture through detection to telemetry output — is documented in the system diagrams notebook. See [`../notebooks/04-system-pipeline.ipynb`](../notebooks/04-system-pipeline.ipynb) for the drone control system diagram, the engine ↔ on-board-controller interaction scheme, and the general end-to-end interaction diagram.

## Quantitative comparison

Full spec, throughput, power, and price data (with sources) are in the returned
research report: [`superpowers/research/hardware-deep-research-report.md`](superpowers/research/hardware-deep-research-report.md).
The condensed picture:

| Board | AI compute | YOLOv8n FPS @640 | YOLOv8n FPS @1280 | Peak power | Flight mass | 2023 price | Verdict |
|---|---|---|---|---|---|---|---|
| **Jetson Nano (4 GB)** | ~0.5 TOPS (472 GFLOPS FP16) | ~15–19 | ~3–5 | ~10 W | ~140 g | $149 | **Chosen** — cheapest, maturest stack, adequate at 640 px |
| Coral Dev Board | 4 TOPS INT8 | ~2.6–4 | fails (≤320 px) | ~5 W | ~130 g | ~$130–150 | Reject — SiLU crashes Edge TPU, CPU fallback |
| RPi 4 + Intel NCS2 | ~1–4 TOPS FP16 | ~5–8 | ~1–2 | ~6.5 W | ~134 g | ~$125 | Reject — sub-real-time, NCS2 discontinued |
| Jetson Xavier NX | 21 TOPS INT8 | ~45–55 | ~15–20 | ~15 W | ~170 g | ~$399 | Throughput winner — only board real-time at 1280 px |

> FPS for Nano and Coral are community-measured; Xavier NX and NCS2 are extrapolated
> from YOLOv5s / YOLOv3-tiny proxies (flagged in the report). Inference-only figures
> exclude capture/pre/post-processing, which can roughly halve end-to-end throughput.

## Flight-time impact — the SWaP lever is weak

We modelled endurance on a representative airframe (5000 mAh 4S LiPo → 59.2 Wh usable
at 80% DoD; base all-up mass 1500 g; hover power 350 W; momentum-theory scaling
`P_hover ∝ m^1.5`). Each board adds both mass (raising hover power) and its own
electrical draw:

| Config | Δmass | Board draw | Total draw | Endurance | Loss |
|---|---|---|---|---|---|
| No compute payload | 0 | 0 W | 350.0 W | 10.15 min | — |
| Coral Dev Board | 130 g | 5.0 W | 401.4 W | 8.85 min | 12.8% |
| RPi 4 + NCS2 | 134 g | 6.5 W | 404.4 W | 8.78 min | 13.5% |
| **Jetson Nano** | 140 g | 10.0 W | 410.1 W | 8.66 min | 14.7% |
| Jetson Xavier NX | 170 g | 15.0 W | 426.2 W | 8.33 min | 17.9% |

The endurance spread across all four boards is **~31 seconds** — well inside the
uncertainty of the hover-power assumptions. Endurance is therefore a *weak
discriminator*: every board costs 13–18% of hover time, and the airframe, not the
compute module, dominates. This flips the decision away from raw SWaP and onto
**throughput and accuracy at the input resolution the mission actually needs.**

## Trade-off and the honest verdict

Scoring the four on a SAR-weighted composite (throughput 40%, endurance 25%,
accuracy-at-altitude 20%, cost/availability 15%) ranks them
**Xavier NX (7.5) > Jetson Nano (5.85) > RPi 4 + NCS2 (4.55) > Coral (4.15)**.

So why did Strix ship on the **Jetson Nano**? The choice is defensible on the axes
that bound a volunteer, budget-constrained 2023 build:

- **Cost & availability** — at $149 it was a third of the Xavier NX's price, the
  decisive factor for a non-commercial SAR project.
- **Ecosystem maturity** — CUDA + TensorRT is the most documented edge-inference
  stack, with the largest body of published drone-SAR precedent running YOLOv5/v8
  on exactly this board. Lowest integration risk.
- **Model fit** — YOLOv8n's 6.3 MB / 3.0 M-parameter footprint sits comfortably in
  the Nano's 4 GB with headroom for the capture and telemetry pipeline. The nano
  model and the Nano board were co-selected: choosing the smallest capable detector
  is what *made* the cheapest board sufficient.
- **Adequate at the operating point** — at 640 px the Nano sustains ~15 FPS end-to-end,
  clearing the ≥10 FPS target for the altitudes the prototype flew.

What the analysis honestly surfaces — and a mature write-up should not hide — is the
**resolution ceiling**. Detecting a person who subtends only a handful of pixels at
40–50 m really wants 1280 px input or SAHI tiling, and there the Nano drops to ~3–5 FPS,
below real-time. The **Jetson Xavier NX** is the only one of the four that holds
real-time at 1280 px, for a sub-minute endurance penalty — the throughput-optimal pick
if budget allowed. The Nano was the right *pragmatic* call under the project's
constraints; the Xavier NX was the right *performance* call.

## What a 2026 rebuild would choose

All four boards are now EOL or dormant (Nano modules extended only to Jan 2027,
JetPack 4 EOL; Xavier NX devkit EOL; Coral dormant since 2021; NCS2 discontinued).
A greenfield build today would move to:

- **Jetson Orin Nano Super ($249)** — 67 INT8 TOPS, native INT8 + sparsity,
  CUDA/TensorRT-compatible so existing Nano pipelines port directly; runs YOLOv8n
  past 100 FPS; supported to 2032. The smoothest upgrade path.
- **Raspberry Pi 5 + AI HAT+ (Hailo-8L/8)** — best FPS/W and FPS/$ at low mass
  (YOLOv8s ~128 FPS @640, ~12 W), at the cost of a less mature toolchain than TensorRT.
