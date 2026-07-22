<h1 align="center">Strix — Drone Search &amp; Rescue</h1>
<p align="center"><em>Finding missing people from the air with on-board neural detection.</em></p>
<p align="center">Presented at <strong>TIBO 2023</strong> — international forum &amp; exhibition, Minsk.</p>

---

Every year thousands of people go missing in hard-to-reach terrain, where ground
search is slow and manpower-limited. **Strix** is a research effort toward an
autonomous drone that sweeps large areas from altitude and detects people in
**real time, on-board** — turning hours of walking into minutes of flight. This
repository documents the full research arc: what was tried, what worked, what did
not, and why each decision was made.

## Headline result

> **YOLOv8-nano**, trained on the Lacmus Drone Dataset (LADD), detects people from
> 40–50 m altitude at **mAP@50 = 0.698**, **Precision = 0.852**, in **8.6 ms/image**,
> with **6.3 MB** of weights — small enough to run on a **Jetson Nano**.

![Sample detection — winter scene, three people found from above](docs/figures/detection-sample-winter-3people.png)

## Research arc

1. **Phase 1 — Semantic segmentation.** Per-pixel person masks (Mask R-CNN → UNet →
   UNet++ → DeepLabV3+). Precise in principle, but masks go **noisy at altitude**
   (person IoU ≈ 0.34) and dense prediction is **too slow/power-hungry** for real time.
2. **Phase 2 — Detection (YOLOv8).** The pivot to instance-level bounding boxes.
   A size sweep showed **nano wins** — larger variants add only ~2 mAP points for
   ~14× the parameters. This is the winning approach.
3. **Hardware.** On-board compute trade-off across four edge boards
   (Jetson Nano, Coral Dev Board, Raspberry Pi 4 + Intel NCS2, Jetson Xavier NX),
   weighing throughput vs. power vs. flight time — ending on **Jetson Nano**.

## Repository map

| Document | What's inside |
|---|---|
| [`docs/00-motivation-and-context.md`](docs/00-motivation-and-context.md) | The problem, LizaAlert, project framing, TIBO 2023 |
| [`docs/01-data.md`](docs/01-data.md) | LADD v4.0 + Semantic Drone Dataset, splits, annotation formats |
| [`docs/02-phase1-segmentation.md`](docs/02-phase1-segmentation.md) | Segmentation experiments and the two walls that ended the phase |
| [`docs/03-phase2-detection.md`](docs/03-phase2-detection.md) | YOLOv8 training, full metrics, size sweep — the core results |
| [`docs/04-hardware-and-edge.md`](docs/04-hardware-and-edge.md) | On-board board trade-off and edge deployment |
| [`docs/05-results-and-conclusions.md`](docs/05-results-and-conclusions.md) | Consolidated results, limitations, future work |
| [`notebooks/`](notebooks/) | The four source notebooks (data prep, segmentation, detection, system pipeline) |
| [`data/README.md`](data/README.md) | How to obtain and prepare the datasets |

## Tech stack

PyTorch Lightning · `segmentation_models_pytorch` · Ultralytics **YOLOv8** ·
albumentations · torchmetrics · ClearML · Weights &amp; Biases.

## Results at a glance

| Phase | Model | Key metrics |
|---|---|---|
| Phase 1 — Segmentation | DeepLabV3+ (resnet34) | pixel acc 0.85 · mean IoU 0.58 · person IoU 0.34 |
| Phase 2 — Detection | YOLOv8n | mAP@50 **0.698** · Precision **0.852** · Recall 0.598 |
