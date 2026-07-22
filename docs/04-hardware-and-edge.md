# On-board compute & edge deployment

## Why on-board inference

Offloading inference to a ground station or cloud endpoint is not viable for aerial SAR. The three structural reasons:

**Latency.** A round-trip over a drone telemetry link — even a low-latency 5.8 GHz video link — adds tens to hundreds of milliseconds of network transit on top of encode/decode overhead. At 40–50 m altitude and a drone moving at 5–10 m/s, a 100 ms latency spike translates to 0.5–1.0 m of positional drift between the frame captured and the alert raised. For a person-sized target (~0.5 × 0.3 m apparent at that altitude), that slip degrades detection reliability and bounding-box accuracy enough to matter operationally.

**Connectivity dead-zones.** Wilderness SAR missions — forests, mountain terrain, flood plains — are precisely the environments where cellular and Wi-Fi coverage is absent or intermittent. A ground-station pipeline that depends on a reliable uplink fails exactly where the system is most needed. On-board compute eliminates the dependency entirely.

**Autonomy.** Strix is designed to operate with minimal operator interaction: the drone follows a predefined search grid, flags detections, and returns coordinates. That autonomy loop — perceive → decide → act — must close on the aircraft. Externalising the perceive step breaks the loop.

The compute target follows directly: run YOLOv8-nano (3.0 M parameters, 6.3 MB weights, ~8.6 ms/image on a Tesla P100 at 1280 px input) at ≥10 FPS on the edge board, without drawing enough power to gut flight time below mission-useful endurance.

## Candidate boards

Four boards were evaluated against this target. Each takes a distinct approach to neural inference on constrained hardware:

- **NVIDIA Jetson Nano (4 GB)** — 128-core Maxwell GPU; inference via CUDA + TensorRT. FP16 and INT8 quantisation available. The project's final hardware choice.
- **Google Coral Dev Board** — dedicated Edge TPU co-processor; requires INT8-only models compiled with the Edge TPU compiler. No floating-point inference path on the TPU itself.
- **Raspberry Pi 4 (4 GB) + Intel Neural Compute Stick 2 (Myriad X VPU)** — host CPU (Cortex-A72) offloads inference to the NCS2 via OpenVINO; model must be converted to OpenVINO IR format.
- **NVIDIA Jetson Xavier NX** — larger Volta GPU (384 CUDA cores) with 48 Tensor Cores; CUDA + TensorRT, higher theoretical throughput and higher peak power draw.

## System pipeline

The on-board processing pipeline — from camera capture through detection to telemetry output — is documented in the system diagrams notebook. See [`../notebooks/04-system-pipeline.ipynb`](../notebooks/04-system-pipeline.ipynb) for the drone control system diagram, the engine ↔ on-board-controller interaction scheme, and the general end-to-end interaction diagram.

## Quantitative comparison

> **Status:** The quantitative board comparison, power/flight-time model, and final Jetson Nano justification are sourced from a dedicated Deep Research pass — see `docs/superpowers/research/hardware-deep-research-prompt.md`. This section is filled in once that report returns.

| Board | TOPS | YOLOv8n FPS @1280 | Peak power (W) | Mass (g) | Price (USD) | Verdict |
|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |
