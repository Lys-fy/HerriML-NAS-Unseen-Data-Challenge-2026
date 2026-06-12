# Canonical naming map (2026-06-12 reorg)

Scheme: four themes. pipeline = the single version line v1..v10 (probe-era v1-v7
continued into pipeline-era v8-v10). baselines = reference methods the pipeline
is measured against. architecture = backbone investigations. reliability =
variance/ensembling investigations. phaseA artifacts belong to pipeline.
EXCLUSIONS (active sweep on spark, names already consistent, DO NOT TOUCH):
baselines/v10/, dev/run_v10_sweep.sh, dev/logs/v10_sweep/ (contents incl. inner dirs).

## Notes (docs/notes/09 - Implementation/implementation/)
history/Tier 2a - Results.md            -> pipeline/Pipeline v1-v7 - Aug Probe.md
Pipeline v10 - Design + Idea Ledger.md  -> pipeline/Pipeline v10 - Design + Idea Ledger.md (move only)
history/Tier 0 - Results.md             -> baselines/Fixed CNN.md
history/Tier 1 - Results.md             -> baselines/Aug + Op Search.md
history/Tier 2b - Results.md            -> baselines/EGNAS Full.md
history/Tier 2b - Macro Diagnostic.md   -> baselines/EGNAS Macro Diagnostic.md
history/Tier 2a - TCN Excursion.md      -> architecture/TCN Backbone Excursion.md
history/Tier 2a - Language Deep Dive.md -> architecture/Language Deep Dive.md
history/Multi-Seed - Results.md         -> reliability/Multi-Seed Variance.md
history/E1 - Results.md                 -> reliability/Ensemble E1.md
(history/ dissolves; research/, root live notes, ../plan/ unchanged)

## Code dirs (baselines/)
tier0_fixed_cnn           -> fixed_cnn
tier1_aug_op_search       -> aug_op_search
tier1e1_ensemble          -> ensemble_e1
tier2b_egnas_macro        -> egnas_macro
tier2b_egnas_full         -> egnas_full
tier2a_language_excursion -> language_excursion
fixed_arch_probe          -> tcn_probe
tier2a_p1_augsafety       -> v2   (v1 failed smoke; the dir's final state is v2)
tier2a_p1_augsafety_v3    -> v3   (... same through ...)
tier2a_p1_augsafety_v9    -> v9
v10                       -> (keep)

## Run scripts (dev/)
run_tier2a_p1.sh        -> run_v2.sh
run_tier2a_p1_v3..v9.sh -> run_v3.sh .. run_v9.sh
run_tier2b_full.sh      -> run_egnas_full.sh
run_tier2b_full_gated.sh-> run_egnas_full_gated.sh
run_tier2b_newdatasets.sh -> run_egnas_newdatasets.sh
run_egnas_diagnostic.sh -> run_egnas_macro.sh
run_tier1_ablations.sh  -> run_aug_op_ablations.sh
run_e1_validation.sh    -> run_ensemble_e1_validation.sh
(all other dev/ script names stay)

## Aggregators / plotters (dev/) renames
aggregate_tier2a_p1.py    -> aggregate_v2.py
aggregate_tier2a_p1_v4.py -> aggregate_v4.py
aggregate_tier2a_p1_v6.py -> aggregate_v6.py
aggregate_tier2a_p1_v7.py -> aggregate_v7.py
aggregate_tier2bfull.py   -> aggregate_egnas_full.py
(plot_tier_results.py keeps its name but its internal tier configs get the new
log paths; aggregate_egnas.py, aggregate_e1.py, aggregate_multiseed.py keep names)

## Log dirs / files (dev/logs/)
tier2a_p1        -> v2     (inner seed dirs p1_sNN -> v2_sNN)
tier2a_p1_v3..v9 -> v3..v9 (inner p1v8_s42 style -> v8_s42 style; rename inner
                            seed dirs of every vN to vN_sNN pattern)
tier2bfull       -> egnas_full
egnasdiag        -> egnas_macro
e1val            -> ensemble_e1
training-2h      -> fixed_cnn_2h
training-tier1-2h-> aug_op_search_2h
tier0-sweep-2h.log  -> fixed_cnn-sweep-2h.log
tier1-sweep-2h.log  -> aug_op_search-sweep-2h.log
tier2bfull-all.log  -> egnas_full-all.log
egnasdiag-all.log   -> egnas_macro-all.log
e1val-all.log       -> ensemble_e1-all.log
(keep: multiseed*, lang_excursion, lstm_lang_probe, tcn_probe, tcnblock_ab,
fingerprint, phaseA, ablation-*, v10_sweep)
Individual CSV/JSON artifact files inside log dirs keep their names.

## Images (docs/notes/09 - Implementation/implementation/images/)
New subdir set: pipeline/ baselines/ architecture/ reliability/
tier0/tier0_curves.png   -> baselines/fixed_cnn_curves.png
tier0/tier0_scores.png   -> baselines/fixed_cnn_scores.png
tier1/tier1_curves.png   -> baselines/aug_op_search_curves.png
tier1/tier1_scores.png   -> baselines/aug_op_search_scores.png
tier1/tier1_ablation.png -> baselines/aug_op_search_ablation.png
tier2b/tier2bfull.png        -> baselines/egnas_full.png
tier2b/tier2bfull_curves.png -> baselines/egnas_full_curves.png
tier2b/tier2b_sNN_curves.png -> baselines/egnas_full_sNN_curves.png (s42/s1337/s2024)
tier2b/tier2b_sNN_scores.png -> baselines/egnas_full_sNN_scores.png
tier2a/tier2a_p1_vN*.png     -> pipeline/vN*.png  (strip the tier2a_p1_ prefix,
                                e.g. tier2a_p1_v6_s42_curves.png -> v6_s42_curves.png)
tier2a/tier2a_v9_curves_grid.png      -> pipeline/v9_curves_grid.png
tier2a/tier2a_v10_curves_grid_sNN.png -> pipeline/v10_curves_grid_sNN.png
tier2a/tier2a_tcn_curves_grid.png     -> architecture/tcn_curves_grid.png
language/*.png            -> architecture/ (names unchanged)
phaseA/*.png              -> pipeline/ (names unchanged)
v10_pipeline.png          -> pipeline/v10_pipeline.png
e1/e1.png                 -> reliability/ensemble_e1.png
multiseed/multiseed.png   -> reliability/multiseed.png
(old subdirs tier0/ tier1/ tier2a/ tier2b/ e1/ language/ multiseed/ phaseA/ removed
once empty)
PNG renames require updating every ![[...png]] embed in the notes AND every
plot script that writes these files.
