# 05 — Results and Conclusions

## Headline Result

The primary objective of the Strix project was to demonstrate real-time person detection from drone altitude using hardware that could realistically be carried onboard an aerial platform. That objective is met. YOLOv8n achieves **mAP@50 of 0.698** with a Precision of 0.852, running inference at **8.6 ms per image** with a weight file of **6.3 MB** — well within the memory and compute envelope of a Jetson Nano. This confirms that a sub-10-ms, single-stage aerial detector is achievable without purpose-built hardware.

---

## What Worked

**Instance-level detection over semantic segmentation.** Treating each person as a bounding-box target rather than a pixel mask proved decisive for aerial imagery. At altitude, person silhouettes occupy a small fraction of the frame and share pixel statistics with surrounding terrain; mask-based approaches accumulate noise at these scales, while detection heads remain discriminative even for single-digit pixel heights.

**Smaller model, better fit.** Counter-intuitively, YOLOv8n outperformed the larger s, m, and l variants on this dataset. The LADD aerial dataset exhibits low visual variance — controlled altitudes, consistent illumination during collection — meaning the capacity headroom of larger models offered no accuracy gain but added latency and weight. Model selection should track dataset complexity, not default to the largest available variant.

**Heavy augmentation on a constrained dataset.** The aerial imagery pool is inherently limited compared to ground-level benchmarks. Geometric and photometric augmentation — flipping, scaling, HSV shifts, mosaic composition — extended the effective training distribution and suppressed overfitting without requiring additional collection flights.

**Deterministic seeded splits.** Fixing the random seed for train/validation/test partitioning throughout both phases ensured that reported metrics are reproducible from the stored checkpoints and are not artifacts of a favorable draw.

---

## What Did Not Work

**Semantic segmentation at altitude** produced a person-class IoU of approximately 0.34 — too low for reliable localization in SAR conditions. The core failure is mask noise: person boundaries blur into background at low pixel resolution, and the DeepLabV3+ head cannot recover fine-grained edges without substantially higher input resolution. Coupled with the energy and latency demands of a full encoder-decoder pipeline, segmentation is not a viable on-board SAR approach at current drone payload constraints.

**Oversized detection models.** Scaling from YOLOv8n to m or l returned negligible mAP improvement at meaningfully higher inference cost. On constrained datasets, larger capacity amplifies overfitting risk rather than precision.

---

## Results at a Glance

| Phase | Model | Key metric | Value |
|---|---|---|---|
| Phase 1 — Segmentation | DeepLabV3+ (resnet34) | pixel accuracy / mean IoU / person IoU | 0.85 / 0.58 / 0.34 |
| Phase 2 — Detection | YOLOv8n | mAP@50 / Precision / Recall | 0.698 / 0.852 / 0.598 |

---

## Limitations

The Recall of 0.598 is the most operationally significant weakness: roughly four in ten people are missed at inference time. The failures concentrate on the smallest targets — persons partially occluded by vegetation, lying prone, or at the edge of the altitude range — exactly the cases that dominate difficult SAR scenarios.

Additional constraints: the detector is single-class (person only), which prevents distinguishing rescuers from victims or detecting relevant objects such as vehicles or shelters. The training distribution is narrow — a limited number of flight sessions under similar conditions — which limits generalization to new terrains, seasons, and lighting regimes. Finally, the Phase 1 segmentation IoU figures are reconstructed from surviving accuracy/loss checkpoints rather than logged directly; the detection metrics are exact.

---

## Future Work

Several directions follow directly from the identified gaps.

**Recall improvement.** High-resolution input tiling (slicing large frames into overlapping sub-images before inference) and test-time augmentation are the most tractable near-term levers. Both can be applied to the existing model without retraining.

**Thermal and IR fusion.** Low-light and dense-canopy conditions render RGB cameras unreliable. A lightweight sensor-fusion pipeline combining thermal IR with RGB detection would address the dominant failure mode in night or overcast SAR operations.

**On-board multi-frame tracking.** Frame-by-frame detection produces spurious false negatives on individual frames. A lightweight tracker (e.g., ByteTrack) integrated on-board would smooth detections across time and substantially reduce the effective miss rate.

**Hardware refresh.** The Jetson Orin Nano offers roughly 40× the INT8 throughput of the original Nano at a comparable form factor and power envelope. Retraining and profiling on this target would likely enable higher-resolution inference within the same latency budget.

**Dataset expansion.** Retraining LADD-derived models across multiple seasons, altitudes, and geographic regions would reduce distribution shift and improve robustness in operational deployments. Collaboration with SAR agencies to collect annotated mission footage remains the highest-leverage data investment.
