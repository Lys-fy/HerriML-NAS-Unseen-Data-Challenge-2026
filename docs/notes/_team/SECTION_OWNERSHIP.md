# Section ownership and note inventory

> Single source of truth for which teammate owns which notes. Use this list to validate wikilinks (only link to notes that exist here).

## Teammate → section mapping

| Teammate name | Owns sections |
|---------------|---------------|
| `competition-framework` | 01 - Competition + 02 - Framework |
| `datasets-researcher` | 03 - Datasets |
| `approaches-researcher` | 04 - Approaches |
| `winners-references` | 05 - Past Winners + 08 - References |
| `strategy-workflow` | 06 - Strategy + 07 - Workflow |

## Topic boundaries

To avoid duplication and competing claims, each section owns specific topics:

| Section | Owns |
|---------|------|
| 01 - Competition | Rules, phases, dates, scoring formula, submission limits |
| 02 - Framework | Pipeline contract, clock semantics, failure modes, mock harness |
| 03 - Datasets | Per-dataset facts (size, format, characteristics, public benchmarks) |
| 04 - Approaches | NAS method descriptions (search algorithms, search spaces, when each applies) |
| 05 - Past Winners | Who-did-what, 2025 leaderboard, 2023 finalists, team affiliations |
| 06 - Strategy | "Why X over Y" reasoning, expected-score analysis, threats to validity |
| 07 - Workflow | Dev tooling, packaging, testing, Phase 2 submission cadence |
| 08 - References | External pointers (papers, repos, pages, people) |

When you mention a concept owned by another section, **wikilink it**, don't restate.

## Full note inventory (by section)

### 01 - Competition (4 notes)
- `Overview.md`
- `Phases and Timeline.md`
- `Scoring.md`
- `Rules.md`

### 02 - Framework (5 notes)
- `Submission Contract.md`
- `Pipeline Flow.md`
- `Clock and Time Budget.md`
- `Failure Modes.md`
- `Mock Harness.md`

### 03 - Datasets (15 notes)
- `Overview.md`
- `AddNIST.md`
- `Language.md`
- `MultNIST.md`
- `CIFARTile.md`
- `Gutenberg.md`
- `Isabella.md`
- `GeoClassing.md`
- `Chesseract.md`
- `Sudoku.md`
- `Voxel.md`
- `Myofibre.md`
- `GameOfLife.md`
- `Cryptic.md`
- `Windspeed.md`

### 04 - Approaches (7 notes)
- `Overview.md`
- `EGNAS.md`
- `Predictor-Guided Evolutionary.md`
- `DARTS Family.md`
- `Zero-Cost Proxies.md`
- `Multi-Fidelity.md`
- `Weight-Sharing One-Shot.md`

### 05 - Past Winners (3 notes)
- `2025 - EVAI.md`
- `2025 - Sichuan teams.md`
- `2023 Finalists.md`

### 06 - Strategy (5 notes)
- `Reliability vs Peak.md`
- `Budget Allocation.md`
- `Score-Efficient Targeting.md`
- `Search vs Train Split.md`
- `Threats to Validity.md`

### 07 - Workflow (4 notes)
- `Dev Loop.md`
- `Submission Packaging.md`
- `Testing Strategy.md`
- `Phase 2 Submission Strategy.md`

### 08 - References (4 notes)
- `Papers.md`
- `Repositories.md`
- `Official Pages.md`
- `Researchers and Groups.md`
