#!/usr/bin/env bash
# Run the full EGNAS-faithful + evolution sweep (search + validate, all 16
# datasets) for the two remaining seeds, giving the 3-seed picture (s42 already
# done). Seed-isolated dirs: ~/local/{egnas_faithful,evo}_s<seed>{,_validated}.
# Self-cd, detach:
#   setsid bash dev/queue_search_seeds.sh > ~/local/search_seeds.log 2>&1 < /dev/null &
set -u
cd "$HOME/projects/HerriML-NAS-Unseen-Data-Challenge-2026" || exit 1

# s42 is already complete (v11.1 in ~/local/v11_1/, egnas+evo validated) -> skip.
# s1337, s2024: full three-way (v11.1 + egnas + evo, searched + validated)
for SEED in 1337 2024; do
    echo "########## SEED $SEED (full)  $(date) ##########"
    SEED="$SEED" bash dev/queue_search_tests.sh
    echo "########## SEED $SEED DONE  $(date) ##########"
done
echo "[seeds] ALL SEEDS DONE  $(date)"
