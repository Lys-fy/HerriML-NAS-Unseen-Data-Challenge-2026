# Team Writing Conventions

> All teammates writing notes in `docs/notes/` must follow these rules. Read this first.

## Hard rules

- **No AI / Claude / Anthropic / Co-Authored-By mentions, ever.** This rule is absolute. It applies to note content, commit messages, PR descriptions — everywhere.
- **Don't commit, don't push.** Only the team lead does that, after consolidation.
- **Don't modify notes outside your assigned section.** If you need to add a wikilink to a note in another section, just write `[[Note Name]]` — the consolidator verifies links.

## File format

- Markdown, Obsidian-flavoured. `[[Wikilinks]]` resolve by basename across folders.
- Each note opens with this frontmatter blockquote, then the body:

  ```
  > **Status:** [drafting | current | stale] | **Topic:** [one-line]
  > **Related:** [[Note A]], [[Note B]], ...
  ```

- After the frontmatter, no second `# Title` — the filename is the title.
- Use tables for structured facts (sizes, dates, comparisons).
- Use code blocks for command examples or file structures.

## Style

- Terse but informative. **200–800 words** per note typical; deep dives can go longer when justified (e.g. EGNAS method note).
- Lead with the answer, then justify. No long preamble.
- Cite sources inline as `[label](url)` for non-obvious or external claims.
- Avoid filler ("In this note, we'll discuss…").

## Wikilinks

- Use `[[Note Name]]` (case-sensitive basename, no folder path).
- Only link to notes that exist. The full note list is in `SECTION_OWNERSHIP.md`.
- If a concept doesn't have a note, write the term plainly. Don't create new files outside your section.

## Cross-section coordination

When you mention a concept owned by another section, **wikilink rather than restate**. Each section owns specific topics — see `SECTION_OWNERSHIP.md` for the canonical list.

Example: a Strategy note discussing CIFARTile-shaped problems should say "for CIFARTile-style problems (see [[CIFARTile]])" — not redescribe the dataset.

## Output deliverables

For each teammate:

1. **Fill assigned `.md` files** with full content. Replace the stub body, keep the frontmatter pattern.
2. **Write a section dossier** to `docs/notes/_team/findings/<your-folder-id>.md` (≤25 bullet points). Cover:
   - Key facts established beyond `PRIOR_FINDINGS.md`
   - Open questions
   - Any conflicts with `PRIOR_FINDINGS.md` or sibling sections
   - Wikilinks to other sections you used (so the consolidator can verify they resolve)
3. **Coordinate via SendMessage** when you need facts from another section, or notice a duplicate / conflict.

## When in doubt

- Check `PRIOR_FINDINGS.md` first — most facts are already established.
- Use WebSearch / WebFetch to fill gaps.
- Message a sibling teammate if you suspect overlap.
- Don't invent claims. Mark uncertainty with "(unconfirmed: …)".
