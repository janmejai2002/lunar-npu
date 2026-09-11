# Archived documentation — do not treat as a source of truth

These 26 files (~705 KB of prose against ~9,000 lines of code) were archived on
2026-09-11 after an audit found that they mix verified fact with fabricated
measurement in a way that gives a reader no way to tell them apart.

**Superseded by:**
- `docs/HARDWARE_AND_RUNTIME_REFERENCE.md` — the facts, each tagged by how it
  was obtained.
- `docs/IDEAS_WORTH_BUILDING.md` — the ideas worth keeping, and why the rest
  were dropped.

**Specific reasons these are not trustworthy:**

- `RESEARCH_VS_REALITY_GAP_ANALYSIS.md` claims **88.6% "REAL & WORKING IN
  PRODUCTION"** and cites the then-166 passing tests as verification. The audit
  found the safety classifier emits a constant, HDC recall returns noise, the
  "zero-copy USM bridge" calls no Level Zero function, and a large share of
  those assertions were tautologies that could not fail. The document written to
  enforce anti-hallucination is itself the largest single overstatement here.
- `research/11_unexplored_frontiers_and_novel_research_ideas.md` presents twelve
  "world-first" paradigms claimed to be undocumented in the literature. All
  twelve are active published fields, none are implemented, and every
  performance figure in the chapter is fabricated.
- The `*_50_PAGES`, `*_TREATISE`, `*_CONSTITUTION` and `*_COMPENDIUM` documents
  are specification prose for systems that were never built.
- Machine facts are wrong in several places: the CPU is a Core Ultra **256V**,
  not 258V, and OpenVINO is **2026.2.1**, not 2025.0.0.

**What is still worth reading here:** `research/01` (architecture background,
broadly consistent with the driver) and `research/03` (quantisation theory,
textbook-correct). Both have been distilled into the reference doc with
provenance tags. Go there instead.
