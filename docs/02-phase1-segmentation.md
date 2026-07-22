# Phase 1 — Semantic Segmentation

## Why Segmentation First

The opening hypothesis was straightforward: if a model can assign a per-pixel class label to every location in an aerial image, the output is the most geometrically precise localisation signal possible. For a drone surveying a search zone, a tight pixel mask around a person is in theory better than a bounding box — it gives exact outline, supports size estimation, and avoids the padding that bounding boxes carry. The dataset (≈ 400 labelled images, three classes: background, person, vehicle) was annotated at pixel level, which made semantic segmentation the natural first target.

---

## Engineering

The segmentation stack was built on **PyTorch Lightning**. Core modules:

- **`PhotopicVisionDataset`** — wraps the annotated aerial imagery, handles mask loading and class remapping.
- **`PhotopicVisionDataModule`** — Lightning data module; splits, batches, and feeds both train and validation loaders.
- **`LightningPhotopicVisionModule`** — training/validation step logic, metric logging, and checkpoint callbacks.
- **`SegmentationPostProcessing`** — applies CRF refinement (via `pydensecrf`) to smooth raw logit maps before evaluation.

Model architectures were sourced from **`segmentation_models_pytorch`**. Metrics tracked per epoch: pixel Accuracy, JaccardIndex / IoU, and FBetaScore, all via **torchmetrics**. Loss: CrossEntropyLoss with class weights to counteract background dominance.

The hardest constraint was the **400-image ceiling**. To compensate, a custom **`TransformPipelineModule`** was written implementing a weather-augmentation pipeline with the following stochastic passes: rain streaks, snow overlay, fog/haze simulation, sun glare, defocus blur, random pixel dropout, and a suite of spatial transforms (flip, rotate, scale, crop). This pipeline was the primary defence against overfitting and a deliberate attempt to make models robust to real-world atmospheric variation during aerial SAR operations.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 704 × 1056 px |
| Batch size | 2 |
| Epochs | 35 – 80 (model-dependent) |
| EarlyStopping patience | 8 epochs |
| ModelCheckpoint | save_top_k = 10, monitor = val_loss |

---

## Model Comparison

Architectures were evaluated in the following chronological order, each informed by limitations observed in the prior run:

| Model | Backbone | mean IoU | person IoU | Params | Notes |
|-------|----------|----------|------------|--------|-------|
| Mask R-CNN | R50-FPN | 0.54 | 0.31 | ~44 M | Instance masks; heavy, ~5 FPS on P100; noisy at altitude |
| UNet | resnet34 | 0.52 | 0.27 | ~24 M | Fast to train; weak on tiny people |
| UNet++ | resnet34 | 0.56 | 0.31 | ~26 M | Nested skips help edges; still noisy on small objects |
| DeepLabV3+ | resnet34 | **0.58** | **0.34** | ~22 M | Best; ASPP gives global context; val_acc 0.85 |

**Best checkpoint — DeepLabV3+ (resnet34):** val_accuracy 0.85, val_loss 0.48 at epoch 62. Mean IoU ≈ 0.58, person IoU ≈ 0.34.

The ASPP (Atrous Spatial Pyramid Pooling) module in DeepLabV3+ samples feature maps at multiple dilation rates simultaneously, giving the network a wide receptive field without losing spatial resolution — a direct structural advantage for detecting context around small objects scattered across large aerial scenes.

---

## Results

![Segmentation triptych: Image | Ground Truth | Prediction](figures/segmentation-image-gt-pred.png)

*Figure: representative output — original aerial frame (left), ground-truth pixel mask (centre), model prediction (right).*

The prediction recovers large, well-defined regions (ground surface, vehicles) with acceptable fidelity. However, on small and geometrically thin classes — people in particular, but also vertical structures such as poles — the prediction mask is visibly fragmented and noisy. This is not a training artifact; it reflects a fundamental resolution mismatch: when an entire person occupies fewer than 10 × 10 pixels, the spatial features available to the decoder are insufficient to form a clean, contiguous mask. The CRF post-processing step reduces isolated noise pixels but cannot recover structure that was not encoded in the first place.

---

## Two Walls That Ended the Phase

### (a) Altitude Noise

As drone altitude increases — a necessity for covering large search areas in reasonable time — human targets shrink rapidly. Below approximately 10 pixels in height, the per-pixel IoU for the person class collapses. The best person IoU achieved across all architectures was **0.34** (DeepLabV3+), already a low number; at operational altitude this figure degrades further. The segmentation signal becomes too noisy to be acted on reliably by an operator or an automated alert system. A metric designed to reward precise boundary delineation penalises exactly the regime where aerial SAR needs to operate.

### (b) Latency and Energy Budget

Dense prediction at 704 × 1056 on a P100 yields approximately 5 FPS for the lightest model tested (DeepLabV3+/resnet34, ~22 M parameters). On an embedded edge board representative of drone-deployable compute, throughput drops further. Real-time coverage of a multi-hectare search zone requires continuous, low-latency inference; the power draw of running full-resolution segmentation on an edge GPU competes directly with flight time. This is not a software optimisation problem — it is a structural consequence of the dense-prediction formulation.

### Conclusion

Both walls point in the same direction. The task does not require precise pixel boundaries; it requires reliable binary answers — **person present / not present** — at sufficient speed and energy efficiency to run aboard a drone in flight. That reformulation is the definition of object detection. The approach, models, and metrics used in Chapter 03 follow directly from this conclusion.
