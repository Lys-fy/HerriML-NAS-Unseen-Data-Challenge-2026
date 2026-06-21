# Dataset Ideas - Backlog

> **Status:** living backlog | **Topic:** candidate custom datasets not yet built, with the design call and an honest fit assessment for each. Built ones graduate to [[Datasets Overview]] + their own note in `additional/`.
> **Related:** [[Datasets Overview]], [[Quincy]], [[Magnus]], [[Cornelius]], [[Pavel]], [[Dominik]], [[Pipeline v10 - Design + Idea Ledger]]

The custom sets exist for the RQ2 generalisation test: built blind to the pipeline, an honest out-of-design-distribution check. The most useful new ones are HARD ENCODED tasks (spatial prior misleads), because that is where v10 has headroom and where the v11 macro/attention lever pays off. Chesseract (chess outcome, 57.66%) is the current hardest and the only place macro wins (+5.94). The goal of new sets is more siblings of that, not more easy image data.

## 1. Go outcome  (codename: Sedol)  -- BUILT 2026-06-17, see [[Sedol]]

> Built: 50k/10k/10k, **2x19x19 (black/white)**, balanced binary, from 228,849 KGS games. Routes **encoded** (corr1 +0.15) with the minimal board-state encoding -- the Chesseract sibling we wanted. (A first 4-plane version routed image-like at +0.42, an artifact of a constant side-to-move plane; see [[Sedol]].) Details in [[Sedol]].


The Chesseract sibling. Predict the eventual game winner from a single mid-to-late board position. Direct test of whether the macro lever that helps chess generalises to a second encoded board game.

- **Source:** free dan-level SGF (KGS via u-go.net/gamerecords, 6d+/7d+ archives; OGS as backup). We redistribute only derived board tensors, not the SGF. Licensing clean.
- **Example:** one board position per game (no game spans the train/test split), label = final result from `RE[...]` (Black win / White win), binary. Drop no-result/jigo.
- **Difficulty:** HARD, Chesseract-like. Sample mid-to-late (tune the move window to a ~60-70% ResNet benchmark). Too early = chance, too late = trivial.
- **Encoding:** 19x19xC uint8 planes: black stones, white stones, empty (+ optional to-move / normalised move-number plane). Matches the encoded-board family.
- **Parse:** `sgfmill` (MIT) replays to the sampled move and reads the result.
- **Interesting unknown:** Go has more genuine 2D locality than chess (stones interact with neighbours), so the fingerprint `corr1` may route it differently than Chesseract. That routing call is itself a test.
- **Codename rationale:** Lee Sedol (AlphaGo), matching the domain-celebrity convention (Magnus = Carlsen for chess).
- **Compute:** generation is light CPU (parse ~100k SGF, single-digit minutes, parallel). Runs on Spark CPU. Training (ResNet zero-point + v10) on Spark GPU as usual. No dedicated box / LXC needed.

## 2. Convergence of an iterative solver  (codename: Cauchy)  -- BUILT 2026-06-17, see [[Cauchy]]

> Built: 30k/7k/7k, 2x16x16 (real/imag of Hermitian A), balanced binary. Routes **encoded** (corr1 -0.03) as intended -- the one clean encoded probe of the three. Details in [[Cauchy]].


"Convex functions in complex space, predict convergence." Pinned-down framing (the encoded one, framing 2):

- **Input:** the coefficients of the problem packed as a small CxHxW tensor -- e.g. a quadratic form's matrix / Hessian, or complex-polynomial coefficients ( C = R^2). NO spatial prior -> genuinely encoded, hard.
- **Label:** converges-in-K-steps (binary) or convergence-rate bucket of an iterative method (gradient descent / Newton) on that operator.
- **Why this framing:** the rejected framing 1 (render the landscape as a heatmap) is smooth/image-like (high corr1 -> routes to CNN, easy). The coefficient-tensor framing is the encoded, hard sibling we want. Infinite cheap labelled data, fully controllable difficulty (condition number, dimension).
- **Open math:** "convex in complex space" is non-standard (convex functions are real-valued); treat C^n = R^2n. Define the solver + target precision before generating.
- **Codename rationale:** Cauchy (complex analysis + convergence).

## 3. Photometric source classification (from Kaggle S6E6)  (codename: Sloan)  -- BUILT 2026-06-17, see [[Sloan]]

> Built: 60k/15k/15k, **2x5x5 (colour + sum matrices)**, balanced 3-way. Giveaways (redshift + leaked label columns) dropped, and sky position dropped too (weak + a constant-plane artifact). Routes **image-like** (corr1 +0.83, smooth colours). Details in [[Sloan]].


Kaggle Playground S6E6 = SDSS star/galaxy/quasar classification (577k rows). Tabular: alpha, delta, u/g/r/i/z bands, redshift, spectral_type, galaxy_population -> class in {GALAXY, QSO, STAR}.

- **As-shipped: do NOT use.** Nearly trivial -- `redshift` alone separates most classes, `spectral_type`/`galaxy_population` are leaked labels. Logistic regression ~95%. Also only ~8 numeric features, too few for a meaningful image.
- **The fix that makes it worthwhile:** (1) DROP the giveaways (redshift, spectral_type, galaxy_population), keep only raw photometry u/g/r/i/z + position. Predicting class from photometry alone is the real, hard historical astronomy problem. (2) Encode by physics: a color-index matrix (all pairwise band differences u-g, g-r, ...) as a small structured 2D tensor, or the 5-band SED as a 1D sequence (feeds the 1D-TCN family).
- **Priority:** below Go. A fine 3rd set IF we strip giveaways + encode by color structure; pointless as a raw reshape.
- **Codename rationale:** Sloan (SDSS = Sloan Digital Sky Survey).

## Assessed and rejected

- **Foundation models / pretrained encoders:** almost certainly disallowed (the unseen-data benchmark is from-scratch by design), and useless-to-harmful on the encoded half where our headroom is. Only the meta-learned router variant is interesting, and that hits the proxy-reliability wall. See [[Pipeline v10 - Design + Idea Ledger]].
