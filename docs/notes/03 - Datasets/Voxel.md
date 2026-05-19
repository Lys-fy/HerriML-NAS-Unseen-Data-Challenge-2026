# Voxel

> **Status:** drafting | **Topic:** 3-D shape classification on a 20×20×20 voxel grid. Channels axis is *spatial-z*, not RGB.
> **Related:** [[Datasets Overview]], [[Failure Modes]]

![[Voxel.png]]
> *Example samples + the classification target — raw `.npy` arrays rendered for viewing. Regenerate: `python3 dev/plot_dataset_examples.py`.*


3-D voxelised shapes, where the shape is constructed by intersecting a generated 3-D primitive with a 20×20×20 grid of points. The "channel" dimension here is repurposed as a spatial-z axis — there is no colour. 7 classes: Sphere, Cube, Cone, Cylinder, Ellipsoid, Cuboid, Pyramid.

| Field | Value |
|---|---|
| Codename | Volga |
| Era | 2024 (one of three hidden datasets in the 2024 competition) |
| Shape | 20×20×20 — `metadata` lists `[50000, 20, 20, 20]`, **no explicit channel dim** |
| Train / val / test | 50,000 / 10,000 / 10,000 |
| Classes | 7 (Sphere, Cube, Cone, Cylinder, Ellipsoid, Cuboid, Pyramid) |
| Benchmark (`metadata`) | 71.35% (ResNet-18, presumably treating one of the 20-axes as channel) |
| License | inferred CC BY 4.0 |
| DOI | https://data.ncl.ac.uk/articles/dataset/Voxel_Dataset/26970223 |

## The shape pathology — read this first

The `input_shape` metadata is `[50000, 20, 20, 20]`, **without a channel axis**. This is unique among the 14 datasets — every other one is `(N, C, H, W)`. Ours is plausibly `(N, D, H, W)` interpreted as a 4-D volume, *or* `(N, 20, 20, 20)` to be reshaped as `(N, 1, 20, 20, 20)` for a 3-D conv, *or* `(N, 20, 20, 20)` treating axis-1 as a 20-channel input to a 2-D conv.

Geada's paper does not cover this dataset (it's post-CVPR-2024). The 71.35% ResNet-18 benchmark in metadata almost certainly comes from **interpreting axis-1 as the channel dimension** for a 2-D conv (since ResNet-18 expects 4-D input). That's the path of least resistance and the organisers' likely choice.

Implication: **our [[Submission Contract]] must not blindly assume `metadata['input_shape'][1]` is a small channel count**. A 20-channel input is unusual but legal (similar to [[Chesseract]]'s 12 channels). A NAS pipeline tuned for 3-channel images will need explicit code for this.

## Type and difficulty

Effectively a **Type-1 with a representational twist**: a human can identify a sphere from a voxel grid. The interesting research question is whether a *2-D* CNN run over axis-1-as-channels can learn the 3-D shape semantics — or whether full 3-D convolution (`Conv3d`) is required.

ResNet-18's 71% suggests 2-D treatment captures plenty of signal, but leaves substantial headroom.

## Implications for our pipeline

- Dataset size on disk: ~2.2 GB — moderately large among the 13 public.
- Score slope at benchmark 71.35% is ~0.35 / 1%. A jump from 71% → 95% earns ~+8.4 pts.
- A 3-D conv backbone (`nn.Conv3d`) is plausibly a winning architecture choice. If our [[Submission Contract]] search space supports 3-D ops, this dataset is a place to use them. If it doesn't, default to "axis-1 = channels in 2-D conv" and accept the 71%-baseline ceiling.
- Memory: 50000 × 20³ × 4-byte float = ~1.5 GB raw — fits easily but watch when materialised through 3-D conv layers.
