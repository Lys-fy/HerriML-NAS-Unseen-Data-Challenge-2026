# Cryptic

> **Status:** drafting | **Topic:** 2025 competition dataset. Source DOI broken — content unknown. Was the **decisive dataset** in the 2025 leaderboard.
> **Related:** [[Datasets Overview]], [[Windspeed]], [[2025 - EVAI]], [[2025 - Sichuan teams]]

One of the three hidden datasets used in the **2025 NAS Unseen-Data Challenge at AutoML**. As of April 2026, the dataset is **not yet publicly available** through any verified channel.

## Source — broken DOI

The starter-kit / organiser docs reference DOI `10.7488/ds/8054` for Cryptic, but that handle resolves to **unrelated content** (Tower Blocks UK photos in the Edinburgh DataShare archive). This is documented in [[PRIOR_FINDINGS]].

`scripts/download_datasets.sh` in this repo deliberately omits Cryptic — there is no working figshare article to point to:
> Cryptic and Windspeed are not included — the starter-kit DOIs (10.7488/ds/8053 and ds/8054) resolve to unrelated content; ask the organisers for correct links.

WebSearch (April 2026) for "Cryptic NAS Unseen Data Competition Newcastle" returns the unrelated CRyPTIC tuberculosis-genomics project, the Edinburgh DataShare CrypticDataset (also unrelated), and various search-engine misroutings. **No correct source has been located**. Action: contact `nas-competition-contact@newcastle.ac.uk` (see [[Official Pages]]) or the organisers' GitHub.

| Field | Value |
|---|---|
| Codename | unknown |
| Era | 2025 |
| Shape | unknown |
| Train / val / test | unknown |
| Classes | unknown |
| Benchmark | unknown |
| License | unknown |
| DOI | broken (`10.7488/ds/8054`) — placeholder until corrected |

## What the 2025 leaderboard tells us

The 2025 competition used three datasets — [[GameOfLife]], [[Windspeed]], **Cryptic**. From [[PRIOR_FINDINGS]] §3:

| Team | Cryptic accuracy | adj_score |
|---|---:|---:|
| EVAI (1st) | 76.145% | +1.745 |
| Jiawen Deng (2nd) | 73.235% | +0.737 |
| Aojun Lu / Y. Sun (3rd) | 72.265% | +0.402 |

A few observations:

- **Cryptic decided the entire 2025 leaderboard.** GameOfLife was max-ed by all three teams (+9.99); Windspeed was near-impossible (+0.03 to +0.45). The 4-percentage-point gap between 1st and 3rd on Cryptic translates to a 1.3-point swing — large enough to flip the ranking.
- The accuracy range (72–76%) sits in a regime where NAS / training tuning has real differential value. Not at the noise floor (so models *can* learn it), not saturated (so optimisation matters).
- The benchmark must be in the 70%+ range to give those modest adj_scores — implying Cryptic is a **medium-benchmark dataset** where the [[Scoring]] slope is ~0.33–0.50 per 1%.

## Working hypothesis on content

Pure speculation, listed for falsification:

- The name suggests an encoding / cipher / cryptography flavour ("cryptic crossword"–style or substitution-cipher classification). Plausible Type-2 in Geada's taxonomy.
- The 76% ceiling and 72% benchmark-style gap is similar in shape to [[Language]] or [[Gutenberg]] — encoded-non-image datasets where letter-frequency / pattern-matching matters.

## Implications for our pipeline

- **Cannot tune for it locally** until the source is found. Pipeline must be generic.
- Likely included in 2026 Phase 2/3 (organisers tend to reuse hidden datasets across years where licenses allow).
- Reading the metadata at runtime is critical — our pipeline must adapt to whatever shape arrives.
- If we're competitive on the rest of the 2025 set, **Cryptic is where we win or lose**. Worth pre-budgeting a "long-tail dataset" branch in the [[Submission Contract]] strategy that does heavier search when the metadata signature is unfamiliar.

## Open actions

- [ ] Email `nas-competition-contact@newcastle.ac.uk` for the correct Cryptic DOI / link.
- [ ] Watch `github.com/Towers-D/NAS-Unseen-Datasets` for additions.
- [ ] If we obtain Cryptic, append the real metadata to this note.
