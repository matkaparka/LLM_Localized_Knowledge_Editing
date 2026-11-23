# Licenses of the non-code material

| Material | License |
|---|---|
| Code (`*.py`) | MIT, see [LICENSE](LICENSE) |
| Our own material: configurations, run metadata, per-item outcome records, analysis outputs and checks, the span annotations and adjudication decisions of FP-12, and the documentation files of this repository | CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ |

## Third-party material

Third-party material keeps its own terms; the CC BY 4.0 grant above does not cover it.

- **WikiBigEdit** (`lukasthede/WikiBigEdit`; Thede et al., "Understanding the Limits of Lifelong Knowledge Editing in
  LLMs", 2025): Apache License 2.0, as stated on the dataset page. The records of KT-10, FP-12, EC-11 and BR-14, RC-15's
  WikiBigEdit records, the RV-09 records and FP-12's annotation files contain its questions, rephrasings, subjects and
  answers. A copy of the license
  is in [third_party_licenses/Apache-2.0.txt](third_party_licenses/Apache-2.0.txt).
- **Wikidata** labels, aliases and descriptions in `experiments/field_prefix_ablation_012/annotation/wikidata_subjects.json`
  (fetched through the Wikidata API on 2026-11-21) and in the audit sheets of `annotation/human_audit/`: CC0 1.0.
- **Model outputs** in the records remain subject to the license of the model that produced them: Qwen3-0.6B-Base
  (the answers in every study except BR-14), OLMo-2-1124-7B (the answers in BR-14), Qwen2.5-7B-Instruct (FP-12
  annotator B's spans) and OLMo-2-1124-7B-Instruct (XR-13's extracted anchor spans), all Apache License 2.0.
- **Registry metadata** (XR-13, and RC-15's registry records): the package identifiers and licence labels from npm, PyPI, crates.io, Hackage,
  Homebrew, NuGet, Packagist and CRAN remain subject to the terms of those registries. The registry-derived question
  texts and the raw snapshots are not included (see [REDACTIONS.md](REDACTIONS.md)).
