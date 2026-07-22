# Data

Strix uses **two datasets**, one per research phase.

| Dataset | Phase | Task | Why |
|---|---|---|---|
| **LADD v4.0** (Lacmus Drone Dataset) | Phase 2 | Detection | Domain-specific: drone shots of people in the wild, the actual target distribution. |
| **Semantic Drone Dataset** (TU Graz) | Phase 1 | Semantic segmentation | Public, densely annotated (23 classes) aerial imagery, used to prototype the segmentation approach. |

## LADD v4.0 (primary)

Collected by the Lacmus project together with the volunteer SAR org **LizaAlert** and the **"Owl"** group.

| Property | Value |
|---|---|
| Images | 1365 |
| Classes | 1 — `human` |
| Drone altitude | 40–50 m, horizontal shots |
| Resolution | 4000 × 3000 JPEG |
| Annotations | VOC (`.xml`) **and** YOLO (`.txt`) |
| Seasons | summer · spring · winter |

**YOLO annotation format** (`class x_center y_center width height`, normalized):

```
0  0.4195  0.9696  0.013  0.0213
```

The local archive is `../../datasets/lacmus-drone-dataset-ladd-v40-*.zip` (~8.4 GB, **gitignored** — not committed).

### Preparation pipeline

Reproduced in [`../notebooks/01-ladd-data-preparation.ipynb`](../notebooks/01-ladd-data-preparation.ipynb):

1. Build a dataframe of `(image, annotation)` paths.
2. Copy images + YOLO labels into `ladd/{images,labels}`.
3. Split deterministically:

   ```python
   splitfolders.ratio(input=ladd_path, output="dataset",
                      seed=42, ratio=(0.80, 0.10, 0.10), move=True)
   ```

This yields **1092 / 136 / 137** images (train / val / test); the validation set contains **606 person instances**. The fixed `seed=42` makes the split reproducible.

Expected on-disk tree:

```
dataset/
├── train/{images,labels}   # 1092
├── val/{images,labels}     # 136
└── test/{images,labels}    # 137
```

## Semantic Drone Dataset (TU Graz)

Used only in Phase 1.

| Property | Value |
|---|---|
| Images | 400 |
| Resolution | 6000 × 4000 |
| Classes | 23 + `conflicting` |
| `person` class | index 15 (RGB 255, 22, 96) |

Masks are single-channel index images (pixel values 0–22). At inference the model outputs a per-class probability tensor; `argmax` collapses it to an index mask, which a colormap turns into an RGB visualization. Homepage: <https://www.tugraz.at/index.php?id=22387>.
