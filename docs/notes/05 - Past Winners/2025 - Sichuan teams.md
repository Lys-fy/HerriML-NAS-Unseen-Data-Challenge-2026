# 2025 - Sichuan teams

> **Status:** current | **Topic:** 2nd (Jiawen Deng) and 3rd (Aojun Lu, Yanan Sun) at AutoML 2025. Both Sichuan University, [[Researchers and Groups|Yanan Sun's NAS group]].
> **Related:** [[Predictor-Guided Evolutionary]], [[Researchers and Groups]], [[2025 - EVAI]], [[Cryptic]], [[GameOfLife]], [[Windspeed]], [[Papers]]

## Two teams, one group

Both 2nd and 3rd place at AutoML 2025 came from **Yanan Sun's NAS group** at Sichuan University. The competition allows multiple teams from the same institution if member-disjoint and methodologically distinct (see [[Rules]]).

| Rank | Team lead | Total | GameOfLife | Windspeed | Cryptic |
|-----:|-----------|------:|-----------:|----------:|--------:|
| 2 | Jiawen Deng | 10.885 | 99.66% / +9.942 | 15.278% / +0.206 | 73.235% / +0.737 |
| 3 | Aojun Lu (with Yanan Sun) | 10.430 | 99.97% / +9.995 | 13.776% / +0.033 | 72.265% / +0.402 |

Combined gap to [[2025 - EVAI]]: ~1.3 to ~1.8 points, almost entirely on [[Cryptic]].

## Team members

### 2nd — Jiawen Deng

- PhD student in Sun's group. Has work accepted at PRICAI 2025 on a **performance predictor** for NAS, per the [group news page](https://yn-sun.github.io/).
- Submission methodology not published; inferred from the group's research direction (see below).

### 3rd — Aojun Lu, Yanan Sun

- **Aojun Lu** — PhD student. Multiple competition wins on the group's record:
  - 1st place at the **AutoML 2024** Neural Architecture Search Challenge (with Xiaotian Song and Yuqi Feng — credited together on the group page).
  - 1st place at the **CVPR 2023** NAS competition (likely the previous "unseen data" iteration).
  - 3rd at AutoML 2025 (this entry).
  - Author of "Revisiting Neural Networks for Continual Learning: An Architectural Perspective" — [arXiv:2404.14829](https://arxiv.org/abs/2404.14829), 2024.
- **Yanan Sun** — Professor, Sichuan University. Group lead. Editorial roles at IEEE TEVC and others; "2025 Outstanding Associate Editor" award at TEVC.

## Inferred submission approach: [[Predictor-Guided Evolutionary]]

No write-up of the 2025 submissions exists publicly. The approach is inferred from the group's published work in 2025–2026:

| Paper                                                                      | Venue                        | Topic                              |
| -------------------------------------------------------------------------- | ---------------------------- | ---------------------------------- |
| DAP: Domain Adaptive Performance Predictor for Efficient NAS — Song et al. | IEEE Trans. Computers (2026) | Cross-dataset predictor adaptation |
| Online Performance Predictor for Evolutionary NAS — Song et al.            | IEEE TSMC (2025)             | Surrogate trained during search    |
| Listwise Ranking Predictor for Evolutionary NAS — Li et al.                | Swarm and Evol. Comp. (2025) | Rank-based surrogate loss          |
| Accurate and Robust NAS via a Flexible Supernet — Feng et al.              | IEEE TNNLS (2025)            | One-shot supernet variant          |

**Predictor-guided evolutionary NAS** trains a surrogate accuracy predictor on cheap architecture encodings, runs the evolutionary search using the predictor, then fully trains only the top-k. See [[Predictor-Guided Evolutionary]] for the method note.

The two Sichuan submissions likely differ in predictor variant or evolutionary operator — that's the team-disjointness justification.

## Why they finished behind EGNAS

- **Predictor priors don't transfer to weird data.** The Cryptic gap (3–4% absolute behind [[2025 - EVAI]]) suggests the predictor's cross-dataset prior — trained on CIFAR-style benchmarks — mis-ranked architectures on Cryptic.
- **Both still beat the −10 cliff on every dataset** — the methodology is competitive, just second-best on Cryptic specifically.
- **GameOfLife saturated the same.** No discrimination there.

The DAP paper (IEEE Trans. Computers 2026) explicitly targets domain adaptation for predictors — likely the group's own response to this competition's signal.

## Implications for HerriML

- **Predictor-guided is a viable runner-up architecture.** If we go this route, [[Predictor-Guided Evolutionary]] tracks how to build it.
- **The Cryptic-gap lesson is the load-bearing one.** Whatever method we ship must not over-rely on CIFAR-pretrained priors. Reinforces [[Threats to Validity]].
- **Sichuan group will likely compete again in 2026.** Expect their predictor methods to be a step better than 2025.
