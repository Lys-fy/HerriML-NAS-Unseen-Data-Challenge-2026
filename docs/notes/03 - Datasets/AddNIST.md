# AddNIST

> **Status:** drafting | **Topic:** Three MNIST images stacked as RGB; label = (r + g + b) − 1, 20 classes.
> **Related:** [[Datasets Overview]], [[MultNIST]], [[Score-Efficient Targeting]]

![[AddNIST.png]]
> *Example samples + the classification target — raw `.npy` arrays rendered for viewing. Regenerate: `python3 dev/plot_dataset_examples.py`.*


Three MNIST digits stacked as the R/G/B channels of a 3×28×28 image. Label is `l = (r + g + b) − 1`, where `r,g,b` are the underlying digits.

| Field | Value |
|---|---|
| Codename | Adaline |
| Era | CVPR (2021–23) |
| Shape | 3×28×28 (channels-first) |
| Train / val / test | 45,000 / 15,000 / 10,000 |
| Classes | 20 (labels 0–19; sum constrained to 1–20) |
| Benchmark (ResNet-18) | 89.85% (`metadata`) — Geada Tab. 1 reports 92.08% from a fuller training run |
| Best off-the-shelf CNN (Geada) | AlexNet 94.87% |
| Best NAS method (Geada Tab. 2) | Bonsai-Net 97.91% |
| Best 2023-comp submission | 95.06% |
| License | CC BY 4.0 |
| DOI | https://data.ncl.ac.uk/articles/dataset/AddNIST_Dataset/24574354 |

## Type and difficulty

**Type-1** (Geada): a human with prior knowledge can solve it perfectly — split the channels, run a digit classifier on each, then add and subtract one. The dataset's research question is whether NAS can "figure out" that a calculation is required and whether the searched architecture has the inference capacity to do so.

ResNet-18 already hits ~90%, so absolute headroom is small (~10%). The [[Scoring]] slope at this benchmark is high (~+1.0 pt per 1% raw accuracy), so each percent matters disproportionately for the total.

## Pathologies and design notes

- The label distribution is **not uniform** over 0–19: there are more ways to get a middle sum (e.g. 9 = 0+0+9, 0+1+8, …) than an extreme one (0 = 0+0+1; 19 = 9+9+9 with permutations). Geada notes this skew but doesn't quantify it.
- Class-balance asymmetry: AlexNet reportedly does well here partly because it captures MNIST-style spatial hierarchies cleanly.
- ConvNext catastrophically underperforms (38.06% in Geada Tab. 1) — design intuition: the dataset's spatial structure is too small / too simple for ConvNext's blocks. Worth treating as a sanity-check signal — if a NAS run lands a ConvNext-shaped architecture on AddNIST, suspect search misbehaviour.

## Implications for our pipeline

- Probably **not** a high-leverage search target — ResNet-18 + reasonable training already gets ~92%. Expected marginal gain from search is ~3–5%, capped by the +10 ceiling at 100%.
- Good candidate for the "fast lane" in [[Budget Allocation]]: train a small CNN with augmentation and move on; don't burn search budget here.
- A reliable +9–10 pt floor here is achievable. See [[Reliability vs Peak]].
