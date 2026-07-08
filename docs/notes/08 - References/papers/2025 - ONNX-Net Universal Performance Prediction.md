# 2025 - ONNX-Net: Universal Representations and Instant Performance Prediction

> **Status:** reference | **Topic:** a modality-agnostic architecture encoder (ONNX-graph -> text -> LM) that predicts accuracy zero-shot across search spaces and datasets | **Found by:** research scout July 2026 (proxy-wall replication angle)
> **Source:** https://arxiv.org/abs/2510.04938 | **PDF:** `docs/papers/Qin-2025_ONNX-Net-Universal-Perf-Prediction_arxiv-preprint.pdf` | **Authors:** Shiwen Qin, Alexander Auras, Elliot J. Crowley et al. (Edinburgh; same group as [[2025 - Transferrable Surrogates in Expressive NAS Spaces]])
> **Related:** [[Papers]], [[2025 - Transferrable Surrogates in Expressive NAS Spaces]], [[2024 - einspace]], [[Probe-Driven Self-Configuration]], [[Key Learnings]]

## TL;DR
Represent ANY architecture by exporting it to ONNX, serialising the graph to text, and encoding that text with a fine-tuned LM (ModernBERT). Train the predictor with a pairwise hinge (ranking) loss, then predict architecture accuracy **zero-shot** on unseen search spaces AND unseen datasets. Tested transferring from CIFAR-10 onto the same eight Unseen-NAS datasets. The pitch is a single "universal" predictor that needs no per-space retraining because ONNX is a common denominator across conv/attention/hybrid graphs.

## Key idea
Prior surrogates (GRAF, cell encodings) are tied to one search space's node/edge vocabulary. ONNX is a universal computational-graph IR, so an ONNX-text encoder gives one representation that spans ConvNets, Transformers and hybrids -> "instant" (millisecond) performance prediction for a brand-new architecture without training it. It is the more ambitious, cross-space sibling of the same group's [[2025 - Transferrable Surrogates in Expressive NAS Spaces]].

## Results that matter for us
- Zero-shot rank correlation across the eight challenge datasets is **very uneven: Kendall tau roughly 0.095 (GeoClassing) up to 0.58**, i.e. near-zero on some tasks. Even a trained, non-trivial cross-architecture LM predictor has no universal validity across this heterogeneous dataset family.
- This is a **second independent replication of our proxy-reliability wall on the exact competition datasets**, and a stronger one than [[2025 - Transferrable Surrogates in Expressive NAS Spaces]]: it fails even with a universal graph encoding, not just within one search space.
- Confirms that "misaligned-modality" datasets are the ones where prediction collapses — the same tasks (GeoClassing, Isabella-like) that break the Transferrable-Surrogates predictor.

## Relevance to our pipeline
Medium. NOT usable at runtime (needs a pretrained cross-architecture predictor + an offline meta-corpus we cannot build in 1-2h/dataset on unknown hardware). Its VALUE to us is evidential: cite alongside [[2025 - Transferrable Surrogates in Expressive NAS Spaces]] as external, dataset-matched corroboration that no cheap-or-even-learned predictor reliably ranks architectures across these datasets -> informed-defaults + a MEASURED probe is the right design, not a predictor. Reinforces the "never let a predictor override a measured probe" rule.

## Concrete idea to steal
Only the framing: if we ever want a coarse pre-order of which probe to run first, an ONNX-text encoding is a clean modality-agnostic key. But given tau can be ~0.1, use it (if ever) as a tie-breaker on probe ORDER, never as a selection signal. Skeptic's note: treat any "universal/instant predictor" claim as unproven on our benchmark - this paper is itself the evidence it does not hold uniformly.
