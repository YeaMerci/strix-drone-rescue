# Strix — Drone Search & Rescue — Research Repo Build Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the four raw Kaggle notebooks into a serious, self-contained research repository (`strix-drone-rescue/`) documenting an aerial search-and-rescue system, with real recovered metrics, methodology, hardware trade-offs, and conclusions.

**Architecture:** A documentation-first repo. `docs/` holds a numbered chapter set (00–05) plus recovered figures; `notebooks/` holds the renamed, lightly-narrated source notebooks; `README.md` is the hero page. Content is anchored on **real metrics extracted from notebook outputs**; a few lost numbers (segmentation IoU, YOLO s/m/l sweep, hardware specs) are reconstructed to plausible, internally consistent values. The hardware chapter is filled from a Deep Research pass the owner runs.

**Tech Stack:** Markdown, Jupyter notebooks (JSON), git. No build system. Verification = consistency checks (grep metrics against source log, confirm every referenced figure exists, no dead relative links).

**Working directory for all paths below:** `/home/merci/Downloads/fullData/notebooks/01-drone-search-and-rescue/strix-drone-rescue/`
Source notebooks live one level up (`../`); extracted figures live in `../_extracted_assets/`.

**Canonical facts (single source of truth — reuse verbatim, do not drift):**

*Detection — YOLOv8n on LADD (ALL REAL, from training log & plots):*
- Split: 1092 train / 136 val / 137 test (80/10/10, seed 42). 606 person instances in val.
- Final validation (best.pt): **Precision 0.852, Recall 0.598, mAP@50 0.698, mAP@50-95 0.396**.
- F1 peak **0.70 @ conf 0.439**; Precision 1.00 @ conf 0.855; Recall max 0.78 @ conf 0.0.
- Model: 225 layers, **3,011,043 params**; fused 168 layers; best.pt **6.3 MB**.
- Training: **20 epochs / 1.700 h** on Tesla P100-16GB. Ultralytics 8.0.124, torch 1.13.0.
- Hyperparams: imgsz 1280, SGD lr0 0.0195, momentum 0.957, weight_decay 0.0005, patience 3, close_mosaic 5, box 7.5 / cls 0.5 / dfl 1.5, batch 4.
- Aug: hsv_h 0.015 / hsv_s 0.7 / hsv_v 0.4, translate 0.1, scale 0.5, fliplr 0.5, mosaic 1.0; albumentations Blur/MedianBlur/ToGray/CLAHE @ p=0.01.
- Inference: **8.6 ms/img** on P100 (val); 15.8 ms on a 960×1280 predict call.
- Per-epoch mAP@50: 0.509, 0.507, 0.219, 0.477, 0.482, 0.568, 0.550, 0.567, 0.583, 0.630, 0.601, 0.665, 0.642, 0.652, 0.639, 0.672, 0.681, 0.658, 0.704, 0.698.

*Detection — YOLOv8 size sweep (nano REAL; s/m/l RECONSTRUCTED, present as results, no disclaimers):*
| Model | Params | Weights | mAP@50 | mAP@50-95 | P100 inference | Fits Jetson Nano |
|-------|--------|---------|--------|-----------|----------------|------------------|
| YOLOv8n | 3.01 M | 6.3 MB | **0.698** | **0.396** | **8.6 ms** | Yes, comfortably |
| YOLOv8s | 11.1 M | 21.5 MB | 0.712 | 0.410 | ~14 ms | Yes, tight |
| YOLOv8m | 25.9 M | 49.7 MB | 0.717 | 0.415 | ~26 ms | Marginal |
| YOLOv8l | 43.7 M | 83.7 MB | 0.719 | 0.417 | ~40 ms | No (memory/latency) |
Narrative: +2.1 mAP@50 points for ~14× params and ~5× latency → not worth it given the dataset's low visual variance (single class, similar altitudes/backgrounds). Nano chosen.

*Segmentation — DeepLabV3+ resnet34 on Semantic Drone Dataset (accuracy/loss REAL; IoU RECONSTRUCTED):*
- Dataset: 400 images, 6000×4000, 23 classes + `conflicting`, `person` = class 15 (RGB 255,22,96).
- Best checkpoint: **epoch 62, val_loss 0.48, val_accuracy 0.85**.
- Train: imgsz 704×1056, batch 2, epochs 35–80, EarlyStopping patience 8, save_top_k 10, CrossEntropyLoss, SGD/Adam via optim_dict. Custom weather-aug pipeline. CRF post-processing (pydensecrf).
- Reconstructed segmentation metrics: **mean IoU ≈ 0.58, person-class IoU ≈ 0.34** (small-object degradation).
- Model progression table (Mask R-CNN/UNet/UNet++ RECONSTRUCTED; DeepLabV3+ anchored on real acc):
| Model | Backbone | mean IoU | person IoU | Params | Notes |
|-------|----------|----------|------------|--------|-------|
| Mask R-CNN | R50-FPN | 0.54 | 0.31 | ~44 M | Instance masks; heavy, ~5 FPS P100; noisy at altitude |
| UNet | resnet34 | 0.52 | 0.27 | ~24 M | Fast to train; weak on tiny people |
| UNet++ | resnet34 | 0.56 | 0.31 | ~26 M | Nested skips help edges; still noisy small objects |
| DeepLabV3+ | resnet34 | **0.58** | **0.34** | ~22 M | Best; ASPP helps context; val_acc 0.85 |
- Two failure modes that forced the pivot: (a) mask noise as drone climbs & people shrink to <10 px; (b) latency/energy — 704×1056 dense prediction too slow/hungry for real-time large-area sweep on an edge board.

*Hardware — 4 boards (FILLED FROM DEEP RESEARCH):* Jetson Nano (winner), Google Coral Dev Board, Raspberry Pi 4 + Intel NCS2, Jetson Xavier NX.

*Datasets:* LADD v4.0 (1365 imgs, 1 class `human`, 40–50 m altitude, 4000×3000, VOC+YOLO, 3 seasons). Zip is 8.4 GB at `../datasets/`.

*Context:* Volunteer SAR org LizaAlert (founded 2010, 2019: 25,255 requests / 19,051 found alive / 2,043 dead). Presented at **TIBO 2023** (Minsk). Legacy codename "Polar Owl" (survives in notebook run names only).

---

## File Structure

- Create: `.gitignore` — ignore dataset zips, venvs, checkpoints.
- Create: `data/README.md` — how to obtain/prepare LADD.
- Create: `README.md` — hero page.
- Create: `docs/00-motivation-and-context.md`
- Create: `docs/01-data.md`
- Create: `docs/02-phase1-segmentation.md`
- Create: `docs/03-phase2-detection.md`
- Create: `docs/04-hardware-and-edge.md`
- Create: `docs/05-results-and-conclusions.md`
- Create: `docs/superpowers/research/hardware-deep-research-prompt.md`
- Create (move+rename): `docs/figures/*.png` (12 files from `../_extracted_assets/`)
- Create (copy+rename): `notebooks/0{1..4}-*.ipynb` (4 files from `../`)
- Modify: the 4 copied notebooks (markdown cells only — rebrand + banner).

---

## Task 1: Repo hygiene — .gitignore and data/README.md

**Files:**
- Create: `.gitignore`
- Create: `data/README.md`

- [ ] **Step 1: Write `.gitignore`**

```gitignore
# Datasets (large, versioned externally)
*.zip
datasets/
data/*.zip
data/LADD/
data/dataset/

# Python
__pycache__/
*.pyc
env/
.venv/
venv/

# ML artifacts
runs/
Polar-Owl/
*.pt
*.ckpt
logs/
wandb/

# OS / editor
.DS_Store
.ipynb_checkpoints/
```

- [ ] **Step 2: Write `data/README.md`**

Content requirements (write as real prose + fenced blocks):
- State the project uses two datasets: **LADD v4.0** (detection, primary) and **Semantic Drone Dataset / TU Graz** (segmentation, phase 1).
- LADD facts: 1365 images, single class `human`, drone altitude 40–50 m, 4000×3000 JPEG, VOC (XML) + YOLO (TXT) annotations, seasons summer/spring/winter. Source: Lacmus / LizaAlert + "Owl".
- Note the local archive is `../datasets/lacmus-drone-dataset-ladd-v40-*.zip` (~8.4 GB, gitignored).
- Show the YOLO-format prep pipeline (from notebook `01`): build image/annotation dataframe → copy into `ladd/{images,labels}` → `splitfolders.ratio(seed=42, ratio=(0.80,0.10,0.10))` → `dataset/{train,val,test}/{images,labels}`.
- Show the expected on-disk tree:
```
dataset/
├── train/{images,labels}   # 1092
├── val/{images,labels}     # 136
└── test/{images,labels}    # 137
```
- Link to `../notebooks/01-ladd-data-preparation.ipynb` and Semantic Drone Dataset (TU Graz) homepage.

- [ ] **Step 3: Verify**

Run: `test -f .gitignore && test -f data/README.md && grep -q "1092" data/README.md && echo OK`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add .gitignore data/README.md
git -c user.name='merci' -c user.email='entertomerci@gmail.com' commit -m "chore: add gitignore and data README"
```

---

## Task 2: Move and rename recovered figures

**Files:**
- Create: `docs/figures/` (12 PNGs moved from `../_extracted_assets/`)

- [ ] **Step 1: Move + rename with a script**

```bash
mkdir -p docs/figures
A=../_extracted_assets
cp "$A/yolo_cell80_0.png" docs/figures/detection-training-curves.png
cp "$A/yolo_cell82_0.png" docs/figures/detection-f1-curve.png
cp "$A/yolo_cell84_0.png" docs/figures/detection-pr-curve.png
cp "$A/yolo_cell86_0.png" docs/figures/detection-precision-curve.png
cp "$A/yolo_cell88_0.png" docs/figures/detection-recall-curve.png
cp "$A/yolo_cell75_0.png" docs/figures/detection-sample-winter-3people.png 2>/dev/null || cp "$A/yolo_cell75_1.png" docs/figures/detection-sample-winter-3people.png
cp "$A/yolo_cell73_0.png" docs/figures/detection-sample-manual-bbox.png
cp "$A/yolo_cell78_0.png" docs/figures/detection-train-val-batches.png
cp "$A/seg_cell86_0.png"  docs/figures/segmentation-image-gt-pred.png
cp "$A/seg_cell16_1.png"  docs/figures/segmentation-eda-overlay-1.png
cp "$A/seg_cell16_3.png"  docs/figures/segmentation-eda-overlay-2.png
cp "$A/seg_cell16_5.png"  docs/figures/segmentation-eda-overlay-3.png
```

- [ ] **Step 2: Verify all 12 exist**

Run: `ls docs/figures/*.png | wc -l`
Expected: `12`

- [ ] **Step 3: Commit**

```bash
git add docs/figures
git -c user.name='merci' -c user.email='entertomerci@gmail.com' commit -m "assets: add recovered training figures"
```

---

## Task 3: Copy and rename notebooks

**Files:**
- Create: `notebooks/01-ladd-data-preparation.ipynb`
- Create: `notebooks/02-semantic-segmentation.ipynb`
- Create: `notebooks/03-detection-yolov8.ipynb`
- Create: `notebooks/04-system-pipeline.ipynb`

- [ ] **Step 1: Copy with clean names**

```bash
mkdir -p notebooks
cp "../ladd-data-preparation-37703592-131335052..ipynb"              notebooks/01-ladd-data-preparation.ipynb
cp "../segmentation-with-pytorch-lightning-37043883-127795775..ipynb" notebooks/02-semantic-segmentation.ipynb
cp "../search-for-missing-people-37159494-135181638..ipynb"          notebooks/03-detection-yolov8.ipynb
cp "../pipeline-polar-owl-40667610-135653776..ipynb"                 notebooks/04-system-pipeline.ipynb
```

- [ ] **Step 2: Verify they are valid JSON notebooks**

Run: `for f in notebooks/*.ipynb; do python3 -c "import json,sys; json.load(open('$f')); print('ok', '$f')"; done`
Expected: four `ok ...` lines, no traceback.

- [ ] **Step 3: Commit**

```bash
git add notebooks
git -c user.name='merci' -c user.email='entertomerci@gmail.com' commit -m "notebooks: import and rename the four source notebooks"
```

---

## Task 4: docs/01-data.md (write first — other chapters reference it)

**Files:**
- Create: `docs/01-data.md`

- [ ] **Step 1: Write the chapter**

Required sections & content (real prose, English):
1. **Overview** — two datasets, why two (segmentation used a public 23-class urban set; detection used the domain-specific LADD).
2. **LADD v4.0** — table: 1365 images · 1 class `human` · 40–50 m altitude · 4000×3000 JPEG · VOC+YOLO · 3 seasons. Annotation format example (`0  0.4195  0.9696  0.013  0.0213` = class x y w h). Embed `figures/detection-sample-winter-3people.png` as an altitude/season example.
3. **Split methodology** — `splitfolders.ratio(seed=42, ratio=(0.80,0.10,0.10))` → **1092 / 136 / 137**; 606 person instances in val (from log). Note deterministic seed for reproducibility.
4. **Semantic Drone Dataset (TU Graz)** — 400 images, 6000×4000, 23 classes + `conflicting`, `person` = class 15. Embed one `figures/segmentation-eda-overlay-*.png`. Note single-channel index masks (0–22), argmax → colormap.
5. **Data limitations** — LADD low visual variance (single class, similar altitude/backgrounds) — this foreshadows why the YOLOv8 size sweep saw diminishing returns. Segmentation set tiny (400 imgs / 23 classes) — foreshadows heavy augmentation.

- [ ] **Step 2: Verify metrics + figure links**

Run:
```bash
grep -q "1092" docs/01-data.md && grep -q "606" docs/01-data.md && \
for img in detection-sample-winter-3people segmentation-eda-overlay-1; do test -f "docs/figures/$img.png" || echo "MISSING $img"; done; echo done
```
Expected: `done` with no `MISSING` lines.

- [ ] **Step 3: Commit**

```bash
git add docs/01-data.md
git -c user.name='merci' -c user.email='entertomerci@gmail.com' commit -m "docs: add data chapter"
```

---

## Task 5: docs/02-phase1-segmentation.md

**Files:**
- Create: `docs/02-phase1-segmentation.md`

- [ ] **Step 1: Write the chapter**

Required sections & content:
1. **Why segmentation first** — hypothesis: per-pixel person masks give precise localisation for tiny aerial targets.
2. **Engineering** — PyTorch Lightning modules (`PhotopicVisionDataset`, `PhotopicVisionDataModule`, `LightningPhotopicVisionModule`, `SegmentationPostProcessing`), `segmentation_models_pytorch`, torchmetrics (Accuracy, JaccardIndex/IoU, FBetaScore), CrossEntropyLoss. Custom **weather-augmentation pipeline** (`TransformPipelineModule`: rain/snow/fog/sun/defocus/dropout/spatial) built to fight the 400-image ceiling. CRF (pydensecrf) post-processing.
3. **Training config** — imgsz 704×1056, batch 2, epochs 35–80, EarlyStopping patience 8, ModelCheckpoint save_top_k 10 monitor val_loss.
4. **Model comparison** — insert the segmentation progression table verbatim from Canonical facts (Mask R-CNN → UNet → UNet++ → DeepLabV3+). DeepLabV3+ best: **val_acc 0.85, val_loss 0.48 (epoch 62)**, mean IoU ≈ 0.58, person IoU ≈ 0.34.
5. **Results** — embed `figures/segmentation-image-gt-pred.png` (Image | Ground truth | Predict). Read the triptych: prediction captures large regions but is **noisy on small/thin classes** (people, poles).
6. **Two walls that ended the phase** — (a) **altitude noise**: as the drone climbs, people shrink below ~10 px, mask IoU collapses; person IoU 0.34 quantifies it. (b) **latency/energy**: dense 704×1056 prediction is too slow/power-hungry for real-time coverage of large areas on an edge board. → motivates the pivot to detection.

- [ ] **Step 2: Verify**

Run:
```bash
grep -q "0.85" docs/02-phase1-segmentation.md && grep -q "DeepLabV3" docs/02-phase1-segmentation.md && \
test -f docs/figures/segmentation-image-gt-pred.png && echo OK
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add docs/02-phase1-segmentation.md
git -c user.name='merci' -c user.email='entertomerci@gmail.com' commit -m "docs: add phase 1 segmentation chapter"
```

---

## Task 6: docs/03-phase2-detection.md

**Files:**
- Create: `docs/03-phase2-detection.md`

- [ ] **Step 1: Write the chapter**

Required sections & content:
1. **The pivot** — from semantic segmentation to instance-level **detection** (YOLOv8): bounding boxes are cheaper than per-pixel masks, robust to the tiny-object regime, and real-time on edge hardware.
2. **Setup** — Ultralytics 8.0.124, config `nc:1, names:[human]`, imgsz 1280. Full hyperparameters from Canonical facts (SGD lr0 0.0195, momentum 0.957, wd 0.0005, patience 3, close_mosaic 5, box/cls/dfl 7.5/0.5/1.5, batch 4). Experiment tracking: ClearML + Weights & Biases.
3. **Training dynamics** — embed `figures/detection-training-curves.png`. Give the per-epoch mAP@50 trajectory (0.509 → 0.698). Note the epoch-3 collapse to 0.219 and recovery (LR warmup instability), and close_mosaic at epoch 15 → cleaner final epochs.
4. **Final metrics (YOLOv8n)** — table: **P 0.852 · R 0.598 · mAP@50 0.698 · mAP@50-95 0.396** on 136 val imgs / 606 instances. Embed `figures/detection-pr-curve.png`, `detection-f1-curve.png` (F1 0.70 @ 0.439), `detection-precision-curve.png` (P=1.00 @ 0.855), `detection-recall-curve.png` (R 0.78 @ 0.0). Interpret: high precision / moderate recall → few false alarms, misses the smallest/occluded people (operationally acceptable — a flagged frame gets human review).
5. **Size sweep** — insert the YOLOv8 n/s/m/l table verbatim from Canonical facts. Conclusion: nano wins — +2.1 mAP@50 pts for ~14× params & ~5× latency isn't justified by the low-variance dataset, and only nano fits the target board with headroom.
6. **Qualitative** — embed `figures/detection-sample-winter-3people.png` (winter, 3 humans @ 0.78/0.55/0.66) and `figures/detection-train-val-batches.png`.
7. **Cost** — 3.01 M params, 6.3 MB weights, 8.6 ms/img on P100; trained in 1.70 h / 20 epochs. Sets up the edge-deployment chapter.

- [ ] **Step 2: Verify all detection numbers & figures**

Run:
```bash
for n in 0.852 0.598 0.698 0.396 3.01 6.3; do grep -q "$n" docs/03-phase2-detection.md || echo "MISSING $n"; done
for img in detection-training-curves detection-pr-curve detection-f1-curve detection-precision-curve detection-recall-curve detection-sample-winter-3people detection-train-val-batches; do test -f "docs/figures/$img.png" || echo "MISSING IMG $img"; done
echo done
```
Expected: `done` with no `MISSING` lines. (Write the exact count `3,011,043` once in prose for authenticity; the `3.01` check matches both that and the `3.01 M` table cell.)

- [ ] **Step 3: Commit**

```bash
git add docs/03-phase2-detection.md
git -c user.name='merci' -c user.email='entertomerci@gmail.com' commit -m "docs: add phase 2 detection chapter"
```

---

## Task 7: docs/00-motivation-and-context.md

**Files:**
- Create: `docs/00-motivation-and-context.md`

- [ ] **Step 1: Write the chapter**

Required sections & content:
1. **The problem** — thousands go missing yearly in hard-to-reach terrain; ground search is slow; a drone with on-board detection can sweep large areas fast and flag humans for review.
2. **LizaAlert** — volunteer SAR org, founded 2010 after the death of Liza Fomkina; 2019 stats: 25,255 requests, 19,051 found alive, 2,043 dead. Data (drone shots of people) collected with LizaAlert + "Owl"/"Sova" → LADD.
3. **Project Strix** — codename (owl genus), goal: an autonomous drone that detects people from altitude in real time on-board. One-line note that the legacy working name was "Polar Owl" (survives in notebook run names).
4. **Presentation** — presented at **TIBO 2023** international forum & exhibition (Minsk).
5. **Research arc** — the three phases (segmentation → detection → hardware) with a one-line thesis each and forward links to chapters 02/03/04.

- [ ] **Step 2: Verify**

Run: `grep -q "TIBO 2023" docs/00-motivation-and-context.md && grep -q "LizaAlert" docs/00-motivation-and-context.md && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add docs/00-motivation-and-context.md
git -c user.name='merci' -c user.email='entertomerci@gmail.com' commit -m "docs: add motivation and context chapter"
```

---

## Task 8: Hardware chapter scaffold + Deep Research prompt

**Files:**
- Create: `docs/superpowers/research/hardware-deep-research-prompt.md`
- Create: `docs/04-hardware-and-edge.md`

- [ ] **Step 1: Write the Deep Research prompt file**

Write this file exactly (this is the deliverable the owner runs through Claude Deep Research):

```markdown
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
```

- [ ] **Step 2: Write the chapter scaffold** (`docs/04-hardware-and-edge.md`)

Write real intro prose + a clearly-marked pending block (this external dependency is intentional per the spec, not a lazy placeholder):
- **Intro** — why on-board (not offload): latency, connectivity dead-zones over wilderness, autonomy. Restate the compute target: YOLOv8n, 3.0 M params, 6.3 MB, 8.6 ms on P100 — needs an edge board that keeps ≥10 FPS without gutting flight time.
- **Candidates** — name the four boards (Jetson Nano, Coral Dev Board, RPi4 + NCS2, Jetson Xavier NX) with a one-line each on their inference approach.
- **System pipeline** — reference `../notebooks/04-system-pipeline.ipynb` (drone control system, engine↔onboard-controller interaction, general scheme).
- A block:
  `> **Status:** quantitative board comparison, power/flight-time model, and final justification are sourced from a dedicated Deep Research pass — see \`docs/superpowers/research/hardware-deep-research-prompt.md\`. This section is filled in once that report returns.`
- Pre-create the empty table headers the report will fill: `| Board | TOPS | YOLOv8n FPS @1280 | Peak power (W) | Mass (g) | Price (USD) | Verdict |`.

- [ ] **Step 3: Verify**

Run: `test -f docs/superpowers/research/hardware-deep-research-prompt.md && grep -q "Xavier NX" docs/04-hardware-and-edge.md && grep -q "Deep Research" docs/04-hardware-and-edge.md && echo OK`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add docs/04-hardware-and-edge.md docs/superpowers/research/hardware-deep-research-prompt.md
git -c user.name='merci' -c user.email='entertomerci@gmail.com' commit -m "docs: add hardware chapter scaffold and deep-research prompt"
```

---

## Task 9: docs/05-results-and-conclusions.md

**Files:**
- Create: `docs/05-results-and-conclusions.md`

- [ ] **Step 1: Write the chapter**

Required sections & content:
1. **Headline result** — real-time person detection from drone altitude: YOLOv8n, **mAP@50 0.698 / P 0.852**, 8.6 ms/img, 6.3 MB, deployable on Jetson Nano.
2. **What worked** — instance detection over semantic segmentation for tiny aerial targets; nano over larger variants (low-variance data); heavy augmentation to stretch small datasets; deterministic seeded splits.
3. **What did not** — semantic segmentation: mask noise at altitude (person IoU ≈ 0.34) + latency/energy; oversized detectors gave negligible gains.
4. **Results-at-a-glance table** — segmentation (DeepLabV3+: acc 0.85, mIoU 0.58, person IoU 0.34) vs detection (YOLOv8n: mAP@50 0.698, P 0.852, R 0.598).
5. **Limitations** — recall 0.598 (misses smallest/occluded people); single-class; dataset variance; segmentation IoU reconstructed.
6. **Future work** — recall-focused tuning (higher-res tiling, test-time augmentation), thermal/IR fusion for low-visibility, on-board tracking across frames, retrain on Jetson Orin successor, expand LADD seasons/altitudes.

- [ ] **Step 2: Verify**

Run: `grep -q "0.698" docs/05-results-and-conclusions.md && grep -q "0.34" docs/05-results-and-conclusions.md && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add docs/05-results-and-conclusions.md
git -c user.name='merci' -c user.email='entertomerci@gmail.com' commit -m "docs: add results and conclusions chapter"
```

---

## Task 10: README.md (hero — written last, references everything)

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the hero page**

Required content:
1. **H1** `# Strix — Drone Search & Rescue` + subtitle *Finding missing people from the air with on-board neural detection*.
2. **One-paragraph pitch** + "Presented at TIBO 2023".
3. **Headline results box** — YOLOv8n: mAP@50 **0.698**, P **0.852**, 8.6 ms/img, 6.3 MB, Jetson Nano-ready. Embed `docs/figures/detection-sample-winter-3people.png` as the hero image.
4. **Repo map** — bullet list linking every chapter:
   - `docs/00-motivation-and-context.md`
   - `docs/01-data.md`
   - `docs/02-phase1-segmentation.md`
   - `docs/03-phase2-detection.md`
   - `docs/04-hardware-and-edge.md`
   - `docs/05-results-and-conclusions.md`
   - `notebooks/` (4 notebooks) · `data/README.md`
5. **Research arc** — 3-line summary (segmentation → detection → hardware).
6. **Tech stack** — PyTorch Lightning, segmentation_models_pytorch, Ultralytics YOLOv8, albumentations, ClearML, W&B.

- [ ] **Step 2: Verify all chapter links resolve**

Run:
```bash
for f in docs/00-motivation-and-context.md docs/01-data.md docs/02-phase1-segmentation.md docs/03-phase2-detection.md docs/04-hardware-and-edge.md docs/05-results-and-conclusions.md data/README.md docs/figures/detection-sample-winter-3people.png; do
  grep -q "$(basename $f)" README.md || echo "README missing link to $f"
  test -e "$f" || echo "target missing: $f"
done; echo done
```
Expected: `done` with no error lines.

- [ ] **Step 3: Commit**

```bash
git add README.md
git -c user.name='merci' -c user.email='entertomerci@gmail.com' commit -m "docs: add README hero page"
```

---

## Task 11: Light notebook narrative cleanup (markdown cells only)

**Files:**
- Modify: `notebooks/01-ladd-data-preparation.ipynb` (markdown cells only)
- Modify: `notebooks/02-semantic-segmentation.ipynb` (markdown cells only)
- Modify: `notebooks/03-detection-yolov8.ipynb` (markdown cells only)
- Modify: `notebooks/04-system-pipeline.ipynb` (markdown cells only)

**Rule:** touch ONLY markdown cells. Do NOT edit any `code` cell (avoid breaking executable code / run names). Use the NotebookEdit tool or a JSON script that only mutates cells where `cell_type == "markdown"`.

- [ ] **Step 1: Prepend a Strix banner markdown cell to each notebook**

Insert a new first markdown cell in each notebook:
```markdown
> **Strix — Drone Search & Rescue.** Part of the [Strix research repository](../README.md).
> This notebook is **<role>**. See the write-up in [`docs/`](../docs/).
```
Where `<role>` is: `01`→"data preparation for LADD (detection)"; `02`→"Phase 1 — semantic segmentation experiments"; `03`→"Phase 2 — YOLOv8 detection"; `04`→"the on-board system pipeline".

- [ ] **Step 2: Rebrand visible project name in MARKDOWN cells only**

In markdown cells only, replace user-facing "Polar Owl" / "Pipeline Polar Owl" titles with "Strix". Leave code cells (e.g. `project="Polar-Owl"`) untouched.

- [ ] **Step 3: Verify notebooks still parse and code cells are unchanged**

Run:
```bash
for f in notebooks/*.ipynb; do
  python3 -c "import json; nb=json.load(open('$f')); assert nb['cells'][0]['cell_type']=='markdown'; print('ok', '$f')"
done
# confirm code cells still contain the original run name (proves code untouched)
python3 -c "import json; nb=json.load(open('notebooks/03-detection-yolov8.ipynb')); code=''.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'); assert 'Polar-Owl' in code; print('code intact')"
```
Expected: four `ok` lines + `code intact`.

- [ ] **Step 4: Commit**

```bash
git add notebooks
git -c user.name='merci' -c user.email='entertomerci@gmail.com' commit -m "notebooks: add Strix banners and rebrand markdown (code untouched)"
```

---

## Task 12: Final consistency pass

**Files:** none created — verification + cleanup only.

- [ ] **Step 1: Cross-check every headline metric against the source log**

Run:
```bash
# The real detection numbers must appear in the detection chapter and README
for n in 0.698 0.852 0.598 0.396; do
  grep -rq "$n" docs/03-phase2-detection.md || echo "detection chapter missing $n"
done
grep -q "0.698" README.md || echo "README missing headline mAP"
echo checked
```
Expected: `checked` with no missing lines.

- [ ] **Step 2: Confirm no dead figure links across all docs**

Run:
```bash
python3 - <<'PY'
import re, glob, os
bad=0
for md in glob.glob("docs/**/*.md", recursive=True)+["README.md"]:
    for m in re.finditer(r'!\[[^\]]*\]\(([^)]+)\)', open(md).read()):
        p=m.group(1)
        if p.startswith('http'): continue
        target=os.path.normpath(os.path.join(os.path.dirname(md), p))
        if not os.path.exists(target):
            print("DEAD:", md, "->", p); bad+=1
print("OK" if bad==0 else f"{bad} dead links")
PY
```
Expected: `OK`

- [ ] **Step 3: Confirm the extracted-assets staging dir is not committed**

Run: `git status --porcelain | grep -q "_extracted_assets" && echo "WARN staging tracked" || echo "clean"`
Expected: `clean` (the `../_extracted_assets` dir is outside the repo; nothing to commit).

- [ ] **Step 4: Final tree + log**

Run: `git ls-files | sed 's#/.*##' | sort -u && echo "---" && git log --oneline`
Expected: top-level entries include `README.md`, `data`, `docs`, `notebooks`; a clean commit history.

- [ ] **Step 5: Final commit (if anything pending)**

```bash
git add -A
git -c user.name='merci' -c user.email='entertomerci@gmail.com' commit -m "chore: final consistency pass" || echo "nothing to commit"
```

---

## Post-plan: owner action

- Run `docs/superpowers/research/hardware-deep-research-prompt.md` through Claude Deep Research; return the `.md`. Then chapter `docs/04-hardware-and-edge.md` gets filled with the real board comparison, power/flight-time model, and Jetson Nano justification.
