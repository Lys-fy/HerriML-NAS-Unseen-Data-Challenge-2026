# TCN Backbone Excursion
> Formerly "Tier 2a - TCN Excursion" (Tier-era naming). Naming map: [[README - Map and Lineage]].

> **Status:** ✅ probe COMPLETE (26/26, seed 42, full 2h). Verdict: the dilated-conv backbone (TCN-2d) beats the searched conv backbone by **+10.9 total adj** as a fixed architecture. Follow-ups in flight: v9 (searched dilated backbone, 3 seeds) + a random-search arm on the 3 stuck datasets.
> **Related:** [[Pipeline v1-v7 - Aug Probe]], [[Language Deep Dive]], [[Run State + Handoff]], [[Key Learnings]], [[Probe-Driven Self-Configuration]]

## The idea

A colleague suggested Temporal Convolutional Networks (TCNs, dilated convs with an exponential receptive field) for the sequential-looking datasets. We tested two readings of "TCN on a C×H×W tensor", as fixed architectures across all 13 datasets, vs the v8 conv backbone:

- **TCN-1d**: true sequence TCN. Reshape the input to a 1-D signal (C channels, length H·W), then 1-D dilated convs. Unlike the row-BiLSTM it trains under plain SGD, so it is searchable.
- **TCN-2d**: 2-D dilated-conv net. Keeps spatial structure, exponential 2-D dilation + residual blocks.

Both: 1×1 stem, N dilated residual blocks (two dilated convs + residual each), global-avg-pool, linear. Code in `baselines/tcn_probe/helpers.py` (`TCN1D`, `TCN2D`, `build_tcn`), run via `PROBE_ARCH=tcn1d|tcn2d`. Sweep `dev/run_tcn_probe.sh`, full 2h, train-to-budget, seed 42.

## Result: TCN-2d is the broad winner

Total adj over the 13 datasets (seed 42, all fixed-arch):

| Backbone | total adj |
|---|---:|
| conv (v8, **searched**) | +48.2 |
| TCN-1d (fixed) | +44.7 |
| **TCN-2d (fixed)** | **+59.2** |
| oracle (best per dataset) | +63.2 |

TCN-2d, a fixed architecture, beats the searched conv backbone by **+10.9**.

Per dataset, TCN-2d vs conv: **win 6, tie 4, lose 3**.
- Big wins: CIFARTile +4.5 (dilation cracks the tiling task, conv 48% to TCN-2d 72%), Language +3.1, Myofibre +3.0. Smaller: AddNIST +0.8, MultNIST +0.8, Voxel.
- Ties: Cryptic, GameOfLife, GeoClassing, Windspeed.
- Losses: Gutenberg −1.1, Sudoku −0.6, Chesseract −0.5. All three are small, odd-shaped, logic-style inputs.

**TCN-1d** is a real specialist (best on Cryptic, Language, Voxel) but weak on the image and logic sets, so it finishes below conv overall. It stays a documented arm, not a backbone.

Plots: `images/architecture/lang_tcn_curves.png` (Language, epoch + wall-clock), `images/architecture/tcn_curves_grid.png` (all 13, per-panel curves). Scripts `dev/plot_lang_tcn.py`, `dev/plot_tcn_grid.py`.

## Two follow-up questions, and what the data already says

**1. Can a search pick the backbone per dataset?** (conv vs TCN-1d vs TCN-2d bake-off, then search within the winner). Tested offline from the curves (`dev/tcn_bakeoff_reliability.py`): the proxy wall shows up even at the coarse backbone level.

| bake-off budget | top-1 correct | total | vs TCN-2d-fixed |
|---|---|---|---|
| 3 min each | 6/13 | +53.8 | worse by 5.4 |
| 5 min each | 8/13 | +61.3 | better by 2.1 |
| 10 min each | 8/13 | +58.7 | about the same |

A naive early bake-off is budget-sensitive: at 3 min it loses to just fixing TCN-2d, at 5 min it wins, at 10 min some curves cross (Myofibre flips). The right bake-off budget is not knowable blind. So a backbone bake-off needs verification (re-check the top one or two at a longer horizon), the same mechanism v8 uses one level up. (Analysis is optimistic: same seed for early and final, so live reliability is a bit worse.)

**2. The 3 losses are random-search territory.** Gutenberg / Sudoku / Chesseract are the weird/odd-shaped inputs where directed search loses to random search in the literature (Chesseract: random-Bonsai 68.8% vs our ~59%). See [[#SOTA context]].

## SOTA context

Per-dataset champions (offline, unlimited compute, from the Geada 2024 paper):

| Dataset | best method | acc | our best |
|---|---|---|---|
| AddNIST | Bonsai | 97.9 | 96.6 |
| MultNIST | DrNAS | 98.1 | 94.8 (conv) |
| GeoClassing | DrNAS | 96.0 | 95.5 |
| Language | PC-DARTS | 90.1 | 89.3 |
| Gutenberg | PC-DARTS | 49.1 | 47.1 (conv) |
| CIFARTile | PC-DARTS | 92.3 | 72.2 |
| Chesseract | random-Bonsai | 68.8 | 59.1 (conv) |

We are within ~1 to 3 points of the offline SOTA on most. The CIFARTile 20-point gap is offline-only (the competition-best was 73, about our 72). The one genuine, budget-fair gap is **Chesseract** (~10 points), which is the random-search opportunity. DrNAS = Dirichlet NAS (a DARTS-family differentiable method); it and PC-DARTS are directed, so they share the bias that loses Chesseract to random search.

### vs the all-time COMPETITION best (the budget-fair frame, from the organisers' repo)

Best result by ANY submission across all competition years, vs our best (seed 42, 2h, blind pipeline):

| Dataset | comp best | ours (config) | delta |
|---|---|---|---|
| AddNIST | 95.06 | **96.62** (tcn2d fixed) | **+1.56 (beat)** |
| MultNIST | 95.45 | **95.52** (tcn2d fixed) | **+0.07 (beat)** |
| Language | 89.71 | 89.34 (tcn1d) | -0.37 |
| CIFARTile | 73.08 | 72.20 (tcn2d fixed) | -0.88 |
| GeoClassing | 96.08 | 95.51 (v9 searched) | -0.57 |
| Chesseract | 62.98 | 61.27 (random arm; best seed 62.95) | -1.71 (tie at best seed) |
| Gutenberg | 50.85 | 47.08 (conv v8) | **-3.77 (our one real gap)** |
| Isabella | 61.42 | not testable (no download) | n/a |

Read: at or beyond the all-time competition frontier on 6 of 7 testable datasets, from ONE coherent blind pipeline (each comp-best row is potentially a different team). Caveats: their hardware/budgets per year unpublished, ours mostly single-seed. Gutenberg is the one real gap; its comp best even beats the organisers' offline PC-DARTS (49.12), so a 2023 team found something non-architectural there (likely training policy or aug). Score-slope perspective: those 3.8 raw points are only ~+0.6 adj, low priority.

## Next steps (in flight)

1. **v9** = the dilated TCN-2d block put INSIDE the v8 search (depth/width tuned per dataset), 3 seeds. `baselines/v9/`, sweep `dev/run_v9.sh`. Confirms whether searching the dilated backbone holds or beats the +59.2 fixed result, multi-seed (the fixed number is single-seed). Micro off, EGNAS_DMAX=8 (the dilated net keeps full resolution).
2. **Random-search arm** = `RAND_SEARCH=1` in `baselines/tcn_probe/nas.py` (samples K random archs from the wide op vocab, short-trains, keeps best). Tested on the 3 stuck datasets (Chesseract, Sudoku, Gutenberg), 3 seeds. `dev/run_random_search.sh`. Tests whether a budget-limited random arm beats conv where every fixed backbone is stuck.

Design rule kept clean: v9 and the random arm are separate experiments, not folded together, so each result is unconfounded (the same reason the fixed-arch probe was separated from the search).
