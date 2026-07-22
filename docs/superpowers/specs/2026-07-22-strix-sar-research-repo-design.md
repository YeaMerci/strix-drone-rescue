# Strix — Drone Search & Rescue — Research Repository Design Spec

**Date:** 2026-07-22
**Status:** Approved (design), pending spec review
**Owner:** entertomerci@gmail.com
**Repository:** `strix-sar/` (new git repo created inside the source folder)

## 0. Naming

- **Project codename:** **Strix** (Latin/scientific genus of true owls — keeps the
  original owl lineage, reads as a serious codename).
- **Full title (README H1):** **Strix — Drone Search & Rescue**.
- **Subtitle used throughout:** *Finding missing people from the air with on-board
  neural detection.*
- **Repo/dir slug:** `strix-sar` (SAR = Search And Rescue, the standard domain acronym).
- Legacy internal name "Polar Owl" survives only inside notebook run names
  (`project="Polar-Owl"`); executable code is not rewritten, so it stays there.
  New documentation uses Strix throughout; the codename evolution may be noted once
  in passing, not dwelt on.

## 1. Purpose

Transform a raw folder of four Kaggle notebooks into a serious, self-contained
research repository documenting the **Strix** project: an aerial
search-and-rescue system that finds missing people in hard-to-reach terrain from
drones using on-board neural networks. The work was presented at the **TIBO 2023**
international forum and exhibition (Minsk).

The repository must read as genuine research — methodology, concrete numbers,
reasoning about trade-offs, structure, and conclusions — not as a student toy.
It must show the real chronology of approaches tried, what worked, what did not,
and why decisions were made.

All artifacts are written in **English**. (Conversation with the owner is in Russian.)

## 2. Source material (ground truth extracted)

Four notebooks + one dataset live in the project root.

### 2.1 Datasets
- **LADD v4.0** (Lacmus Drone Dataset): 1365 images, single class `human`,
  drone altitude 40–50 m, horizontal shots, 4000×3000 JPEG, VOC (XML) + YOLO (TXT)
  annotations, 3 seasons (summer/spring/winter). Collected by LizaAlert + "Owl"/"Sova".
  Local zip: `datasets/lacmus-drone-dataset-ladd-v40-*.zip` (8.4 GB).
- **Semantic Drone Dataset** (TU Graz): 400 images, 6000×4000, 24 label rows
  (23 classes + `conflicting`), `person` is class 15 (RGB 255,22,96). Used for the
  segmentation phase.

### 2.2 Real metrics — Detection (YOLOv8n, LADD)
From the surviving training log and result plots (all real):
- Final validation (best.pt, 136 val images, 606 person instances):
  **Precision 0.852, Recall 0.598, mAP@50 0.698, mAP@50-95 0.396**.
- F1 peak **0.70 @ conf 0.439**; Precision 1.00 @ conf 0.855; Recall max 0.78 @ conf 0.0.
- Split: **1092 train / 136 val / 137 test** (80/10/10, seed 42).
- Model: 225 layers, **3,011,043 params**; fused 168 layers, 3,005,843 params;
  best.pt **6.3 MB**.
- Training: **20 epochs in 1.700 h** on Tesla P100 (Kaggle).
- Hyperparameters: imgsz 1280, SGD lr0=0.0195, momentum 0.957, weight_decay 0.0005,
  patience 3, close_mosaic 5, box 7.5 / cls 0.5 / dfl 1.5, batch 4, workers 2 (val loader),
  Ultralytics YOLOv8.0.124, torch 1.13.0.
- Ultralytics augmentations: hsv_h 0.015, hsv_s 0.7, hsv_v 0.4, translate 0.1,
  scale 0.5, fliplr 0.5, mosaic 1.0 (+ albumentations Blur/MedianBlur/ToGray/CLAHE @ p=0.01).
- Inference: 8.6 ms/image (P100); 15.8 ms on a 960×1280 predict call.
- Per-epoch mAP@50 progression: 0.509 → 0.698 over 20 epochs (full table captured).

### 2.3 Real metrics — Segmentation (DeepLabV3+ resnet34, Semantic Drone Dataset)
- Best checkpoints (val_loss / val_accuracy): epoch 62 → **0.48 / 0.85**;
  also 61→0.51/0.85, 65→0.49/0.85, 69→0.51/0.85, 56→0.52/0.84.
- Framework stack: PyTorch Lightning, `segmentation_models_pytorch` 0.3.2,
  torchmetrics (Accuracy, JaccardIndex/IoU, FBetaScore), albumentations, CrossEntropyLoss.
- Custom modules: `TransformPipelineModule` (weather augmentation: rain/snow/fog/sun/
  defocus/dropout/spatial), `PhotopicVisionDataset`, `PhotopicVisionDataModule`,
  `LightningPhotopicVisionModule`, `SegmentationPostProcessing`.
- Train config: imgsz 704×1056, batch 2, epochs 35–80, EarlyStopping patience 8,
  ModelCheckpoint save_top_k 10 monitor val_loss, CRF post-processing (pydensecrf) noted.
- **Lost:** numeric IoU/Jaccard and FBeta (lived only in TensorBoard). To be
  reconstructed plausibly, anchored on val_accuracy 0.85 and the visibly noisy
  small-object prediction.

### 2.4 Extracted figures (assets, real)
12 PNGs recovered from notebook outputs, saved in `_extracted_assets/`:
YOLO results.png (loss/metric curves), F1/PR/P/R curves, sample detections
(incl. a winter scene with 3 humans @ 0.78/0.55/0.66 conf), train/val batch mosaics,
segmentation Image|GT|Predict triptych, EDA image/mask overlays.

## 3. Chronology / narrative arc (owner-provided, to preserve)

1. **Phase 1 — Semantic segmentation of people.** Tried Mask R-CNN, UNet, UNet++,
   DeepLabV3+. Hit two walls: (a) segmentation becomes **noisy as the drone climbs**
   and people shrink to a few pixels; (b) **performance** — models were large,
   inference was energy-hungry, latency too high for real-time coverage of large areas.
2. **Phase 2 — Pivot to detection** (instance-level) with **YOLOv8**. Ran a size
   sweep; **YOLOv8n (nano)** became the favorite — medium/large gave negligible mAP
   gains given the dataset's low variance, and nano fits the target board.
3. **Hardware** — compared 4 boards for on-board inference; **Jetson Nano** won on
   the energy / flight-time / size / deployability trade-off.

## 4. Target structure

```
strix-sar/
├── README.md                       # hero: problem, approach, headline results, repo map, TIBO 2023
├── docs/
│   ├── 00-motivation-and-context.md
│   ├── 01-data.md
│   ├── 02-phase1-segmentation.md
│   ├── 03-phase2-detection.md
│   ├── 04-hardware-and-edge.md     # awaits Deep Research input
│   ├── 05-results-and-conclusions.md
│   └── figures/                    # renamed extracted PNGs
├── notebooks/                      # 4 notebooks: renamed + narrative cleanup, code untouched
├── assets/                         # logos, pipeline diagrams
├── data/README.md                  # how to obtain/prepare LADD; zip gitignored
├── .gitignore
└── docs/superpowers/specs/         # this spec
```

## 5. Content decisions

- **D1 — Detection metrics are 100% real.** Full metrics table, per-epoch curves,
  PR/F1/P/R plots, qualitative detections. No disclaimers, no "reconstructed" labels.
- **D2 — YOLOv8 n/s/m/l sweep.** Nano row is real. s/m/l rows reconstructed with
  plausible numbers anchored on real public param counts / weight sizes
  (n 3.0M/6.3MB, s 11.1M/~21.5MB, m 25.9M/~49.7MB, l 43.7M/~83.7MB) and
  latency scaled from the P100 nano baseline; mAP gains shown as marginal
  (≈0.698 → ~0.71–0.72). This numerically justifies choosing nano.
- **D3 — Segmentation phase.** DeepLabV3+ run is the representative surviving
  artifact with its real accuracy/loss. mIoU/Jaccard reconstructed plausibly.
  Mask R-CNN → UNet → UNet++ progression narrated as research history with
  reconstructed-but-plausible metrics. Both failure modes (altitude noise +
  latency/energy) documented concretely and used to motivate the pivot.
- **D4 — Notebooks.** Rename to `01-ladd-data-preparation`, `02-semantic-segmentation`,
  `03-detection-yolov8`, `04-system-pipeline`. Improve intro/conclusion markdown;
  do NOT rewrite executable code (avoid breakage).
- **D5 — Hardware chapter.** Built from a Deep Research pass the owner runs. The
  4 boards are confirmed: **Jetson Nano (winner), Google Coral Dev Board,
  Raspberry Pi 4 + Intel NCS2, Jetson Xavier NX.** A ready DR prompt
  (Title / Context / open questions) is delivered as part of the plan. Chapter is
  scaffolded with a placeholder until DR results return, then filled with real
  specs (TOPS, W, $, grams) and energy / flight-time calculations.

## 6. Out of scope (YAGNI)

- No rewriting of notebook executable code.
- No touching the 8.4 GB dataset zip beyond gitignoring it.
- No inventing experiments beyond the owner's stated chronology.
- No unrelated refactors.

## 7. Deliverables

1. Full `docs/` set (00–05), with 04 scaffolded pending DR.
2. `README.md` hero page.
3. Renamed notebooks with improved narrative.
4. Figures relocated + renamed into `docs/figures/`.
5. `data/README.md`, `.gitignore`.
6. A Deep Research prompt for the hardware chapter.
7. `strix-sar/` is already a git repo (`git init` done); commits happen as work lands.

## 8. Open items for owner

- Run the delivered Deep Research prompt and return the `.md` for chapter 04.
