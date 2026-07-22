# Deep Research Prompt — Edge AI boards for on-board drone person-detection

**Title:** On-board inference boards for a real-time aerial search-and-rescue drone: Jetson Nano vs Coral Dev Board vs Raspberry Pi 4 + Intel NCS2 vs Jetson Xavier NX

**Context:** We run YOLOv8-nano (3.0 M params, 6.3 MB weights, ~8.6 ms/img on a Tesla P100 at 1280 px) to detect people from a drone at 40–50 m altitude. We need to pick an on-board compute module. The drone is battery-powered, so weight and power draw directly reduce flight time. We want real-time inference (target ≥10 FPS at the model's input size) with the smallest hit to endurance. The final project chose Jetson Nano; we want the numbers that justify (or challenge) that.

**Boards to compare (exactly these four):**
1. NVIDIA Jetson Nano (4 GB)
2. Google Coral Dev Board (Edge TPU)
3. Raspberry Pi 4 (4 GB) + Intel Neural Compute Stick 2 (Myriad X)
4. NVIDIA Jetson Xavier NX

**Open questions / required data points (per board):**
- AI throughput (TOPS / GFLOPS) and realistic YOLOv8n FPS at 640 and 1280 px.
- Idle and peak power draw (W), and typical draw under continuous inference.
- Board mass (grams) and dimensions.
- Price (USD, 2023 and current).
- Supported inference stacks (TensorRT / Edge TPU compiler / OpenVINO) and whether YOLOv8n runs natively or needs conversion/quantization (INT8) — and the accuracy cost of that quantization.
- RAM, thermal/cooling needs, typical operating temperature range.

**Analysis asked for:**
- A flight-time impact model: given a representative drone (state assumptions: e.g. ~5000 mAh 4S LiPo, ~350 W hover draw), estimate how each board's added power shortens endurance.
- A weighted trade-off (FPS/W, FPS/gram, FPS/$) and a recommendation, checked against the real-world choice of Jetson Nano.
- Note any 2023-vs-now changes (e.g. Jetson Nano EOL, Orin Nano successor).

**Deliverable:** a markdown report with per-board spec tables, the flight-time model with stated assumptions, and a ranked recommendation. Cite sources.
