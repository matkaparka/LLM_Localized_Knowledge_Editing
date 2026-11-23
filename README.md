# LLM_Localized_Knowledge_Editing

Code, frozen configurations and per-item outcome records for the manuscript

> **When memory keys fire on the wrong question: Diagnosing collateral damage in continual knowledge editing**
> Lihan Xu, 2026

The manuscript studies a routed residual memory editor of the GRACE family on Qwen3-0.6B-Base and repeats the main
analysis on OLMo-2-1124-7B. Each stored edit is a key at one decoder block (block 8 of Qwen3 in the main runs) with its
own firing threshold, calibrated on unlabelled background questions. Permuting the thresholds within write batches
tests whether it matters which key receives which threshold; replays with single keys enabled or disabled link damaged
answers to the entries that cause them. The manuscript also applies the protocol, prospectively, to SHARD, a published
slot editor, on new CounterFact streams with Qwen2.5-1.5B-Instruct, and tests whether background similarity can tell an
inspector in advance which entries will damage answers (SP-16). The manuscript itself is not part of this repository.

## Contents

| Study (label in the manuscript) | What it tests | Folder under `experiments/` |
|---|---|---|
| Assignment test (KT-10) | Matched against permuted and uniform thresholds on WikiBigEdit streams D4–D7 (D0–D3 for tuning) | `key_thresholds_010/` |
| Factorial attribution (FP-12) | Entity anchor (original or corrected) crossed with routing on shared demonstrations (allowed or blocked), on the same streams | `field_prefix_ablation_012/` |
| New-source test (XR-13) | The threshold test on questions about packages in eight software registries | `external_key_replication_013/` |
| Systems comparison (EC-11) | Probe-calibrated keys against a GRACE reimplementation, radius rules, subject cleaning with a uniform threshold, and retrieval, on streams C0–C7 | `editor_compare_011/` |
| Robustness checks (RC-15) | Entries disabled by their thresholds held in place or disabled under a uniform threshold; write seeds 1 and 2; routing blocks 6 and 12; raw cosines without centring; on FP-12's streams and XR-13's registries | `robustness_checks_015/` |
| Second backbone (BR-14) | The threshold test on OLMo-2-1124-7B with RC-15's comparators, on streams D4–D7 (D0–D3 for tuning), and an exploratory run with the similarity floor lowered to 0.80 | `backbone_replication_014/` |
| Published slot editor (SP-16) | SHARD (the authors' single-slot code) on Qwen2.5-1.5B-Instruct and new CounterFact streams T01–T08 (D01–D02 for tuning): matched, permuted, uniform and suppressed read-outs of each frozen table, and a prospective screen that ranks slots by maximum background similarity or by background wins; with the author's blind audit of the scoring | `shard_prospective_016/` |

Two earlier folders contribute modules that the studies import unchanged. Neither earlier study is reported in
the manuscript, and only what the studies need is included:

- `self_observation_008/` (SO-08): WikiBigEdit stream construction, the model wrapper and evaluation, and
  `so_fetch.py`, the hashed download of the pinned data and model;
- `reversal_pilot_009/` (RV-09): the key-routed residual editor (`rv_editor.py`) and its writing policy;
  `calibration/` holds the scripts that chose the editor's settings on a pilot stream, and `runs/run_002/a1/` the four
  records that KT-10's analysis compares against.

The folder names are the internal ones, because the code imports its sibling folders by these names. In each study
folder:

- `*.py`: experiment and analysis code, with CPU unit tests in `test_*.py`;
- `config_*.json`: the frozen configurations (XR-13's test configuration is `data/host/config_test.json`);
- `runs/<run>/` (XR-13: `data/host/runs/<run>/`): `config.json` (a copy of the configuration), `run.json` (SHA-256 of the
  configuration and of the code files the run imported, and the environment), `manifest.json` (SHA-256 of the
  outputs), one gzipped JSON record per stream (`stream_<name>.json.gz`) and `analysis.json` (RC-15 and BR-14 also keep
  the recount there, as `recount.json`);
- `checks/`: an independent recount of the main numbers (`recount.py`, which shares no code with the analysis), its
  output, and other descriptive checks. BR-14 runs RC-15's code (`rc_run.py`, `checks/recount.py`) with its own
  configurations; its own code is limited to the download, calibration and design scripts.

The stream records hold, for every item (an edit's original wording and rephrasings, earlier edits, and base-known
locality facts) and every threshold arm, whether the answer was right, which memory entries fired, the model's wrong
answers, and the single-key interventions. The docstrings of the analysis modules (`kt_analysis.py`, `ec_analysis.py`,
`fp_analysis.py`, `xr_analysis.py`, `rc_analysis.py`, `sp_analysis.py`) define the measures.

SP-16's records differ in form: each `stream_<name>.json.gz` holds, for the protected facts, the edits' original
prompts and their paraphrases, every arm's generated text and the slots that fired, the native trace and the single-slot
replays; `seal_<name>.json` holds the rankings, per-slot thresholds and permutations, which were written and hashed
before any protected fact was read. `runs/smoke_001/smoke.json` is the acceptance test against the unmodified SHARD
code.

Every file is stored byte for byte (`.gitattributes` turns off line-ending conversion), so the hashes in `run.json`
and `manifest.json` can be checked with `sha256sum`. Records changed for the public release are listed, with their
hashes before and after, in [REDACTIONS.md](REDACTIONS.md). `SHA256SUMS.txt` covers every file.

### Where the main numbers are

Paths are under `experiments/`. The files store differences as fractions; the manuscript gives percentage points.

| Number in the manuscript | File | Field |
|---|---|---|
| Assignment test: damage reduction of matched over permuted thresholds, 11.65 pp | `key_thresholds_010/runs/run_002/analysis.json` | `test.h1_matched_vs_shuffled` |
| Factorial attribution: 3.83 pp with corrected anchors and blocked demonstrations; 12.67 pp remeasured baseline | `field_prefix_ablation_012/runs/test_001/analysis.json` | `headline.delta_corr_off`, `headline.delta_orig_on` |
| Registry test: 34.00 pp | `external_key_replication_013/data/host/runs/test_001/analysis.json` | `delta` |
| Registry test without npm and Hackage (post hoc): 11.37 pp | `external_key_replication_013/checks/posthoc_test_001.json` | `subsets["without hackage+npm"].delta` |
| Systems comparison: retrieval ahead of probe-calibrated keys by 6.86 pp | `editor_compare_011/runs/test_001/analysis.json` | `primary.kt_vs_rag` |
| Disabled entries held in place: 3.24 pp (corrected anchors, blocked demonstrations) | `robustness_checks_015/runs/fp_test_001/analysis.json` | `headline["corr/off"].delta_active` |
| The same on the registries: 8.56 pp | `robustness_checks_015/runs/xr_test_001/analysis.json` | `headline.xr.delta_active` |
| Write seeds 1 and 2, blocks 6 and 12, no centring: 3.97, 3.72, 4.80, 3.49 and 19.52 pp | `robustness_checks_015/runs/<seed1, seed2, layer6, layer12, raw>_test_001/analysis.json` | `headline["corr/off"].delta` |
| Second backbone: 0.27 pp at the floor 0.90; 1.46 pp at the exploratory floor 0.80 | `backbone_replication_014/runs/test_001/analysis.json`, `runs/explore_f080_001/analysis.json` | `headline["corr/off"].delta` |
| Human audit of the corrected spans: same final character in 94 of 100 sampled cases and in 36 of 50 sampled adjudicated disagreements | `field_prefix_ablation_012/annotation/human_audit/audit_result.json` | `final_char`, `scored`, `final_char_ci` per sheet |
| SHARD screen with 13 slots (5%): 35 of 824 damaged facts located (4.25%) against 5.43% for random inspection and 15.78% for background wins; differences −1.18 and −11.53 pp | `shard_prospective_016/runs/test_001/analysis.json` | `main.budgets["13"]` (`hits`, `rate_risk`, `rate_random`, `rate_freq`, `diff_random`, `diff_freq`, `ci`), `main.decision` |
| SHARD protection: damage 87.75% native, 6.07% matched, 24.56% permuted (mean of eight), 19.17% uniform 0.95 | the same file | `main.arms.<arm>.pooled.damage`, `main.matched_vs_shuffled` |
| Background wins against random inspection (post hoc): 10.35 pp | `shard_prospective_016/runs/test_001/freq_vs_random_post_hoc.json` | `results.main["13"]` |
| Post-hoc descriptions of the screen (similarity spread, firing, sufficient slots, leakage of the written object) | `shard_prospective_016/runs/test_001/describe_post_hoc.json` | `pooled` |
| Blind audit of SP-16's scoring: 36 of 40 damage pairs confirmed, the other four ambiguous | `shard_prospective_016/annotation/blind_audit/audit_result.json` | `damage_pairs`, `matched`, `split`, `agreement` |

In the RC-15 and BR-14 analyses each `headline` entry is `[mean, [lower, upper], positive units, units]`, and
`cells_result.<cell>` holds the factorial of disabled entries and graded thresholds (`factorial`), the practical
comparisons (`practical`) and the culprit groups (`culprits`).

## Re-running the analyses from the records

The analyses run on a CPU in under a minute each. The command-line scripts import torch and transformers as well as
numpy (versions in `requirements.txt`; Python 3.12). From the repository root:

```
cd experiments/key_thresholds_010
python kt_run.py analyze runs/run_002 --extra runs/run_003_D7 --rv09-a1 ../reversal_pilot_009/runs/run_002/a1
python checks/recount.py runs/run_002 runs/run_003_D7

cd ../editor_compare_011
python ec_run.py analyze runs/test_001 --dev runs/dev_001
python checks/recount.py runs/test_001 runs/dev_001 checks/recount_test_001.json

cd ../field_prefix_ablation_012
python fp_run.py analyze runs/test_001 --dev runs/dev_001
python checks/recount.py runs/test_001 runs/dev_001 checks/recount_test_001.json

cd ../external_key_replication_013
python xr_run.py analyze data/host/runs/test_001 --dev data/host/runs/dev_001
python checks/recount.py data/host/runs/test_001 data/host/runs/dev_001 checks/recount_test_001.json

cd ../robustness_checks_015
python rc_run.py analyze runs/fp_test_001 --dev runs/fp_dev_001 --ref ../field_prefix_ablation_012/runs/test_001
python checks/recount.py runs/fp_test_001 runs/fp_dev_001 runs/fp_test_001/recount.json
python rc_run.py analyze runs/xr_test_001 --dev runs/xr_dev_001 --ref ../external_key_replication_013/data/host/runs/test_001
python checks/recount.py runs/xr_test_001 runs/xr_dev_001 runs/xr_test_001/recount.json
python rc_run.py analyze runs/seed1_test_001            # likewise seed2, layer6, layer12 and raw
python checks/recount.py runs/seed1_test_001 runs/seed1_test_001/recount.json

cd ../backbone_replication_014
python ../robustness_checks_015/rc_run.py analyze runs/test_001 --dev runs/dev_001
python ../robustness_checks_015/checks/recount.py runs/test_001 runs/dev_001 runs/test_001/recount.json
python ../robustness_checks_015/rc_run.py analyze runs/explore_f080_001
python ../robustness_checks_015/checks/recount.py runs/explore_f080_001 runs/explore_f080_001/recount.json

cd ../field_prefix_ablation_012/annotation/human_audit
python score_audit.py audit_sheet.csv audit_sheet_disagreements.csv audit_result.json

cd ../../../shard_prospective_016          # sp_run.py needs the SHARD code; fetch and verify it first (see below)
python verify_upstream.py
python sp_run.py analyze runs/dev_001
python sp_run.py analyze runs/test_001 --dev runs/dev_001
python checks/recount.py runs/test_001 runs/test_001/recount.json      # standard library only; likewise runs/dev_001
python checks/describe_post_hoc.py runs/test_001 runs/test_001/describe_post_hoc.json
python checks/freq_vs_random_post_hoc.py runs/test_001 runs/test_001/freq_vs_random_post_hoc.json
cd annotation/blind_audit
python score_audit.py audit_sheet.csv audit_result.json
```

`analyze` overwrites the `analysis.json` of the run it reads. On 2026-11-22 (2026-11-23 for RC-15, BR-14 and SP-16)
these commands, run on a copy of this repository, regenerated every included `analysis.json` and recount output
exactly, with one exception: in XR-13 the culprit categories of the single-key interventions use question texts that
are not published, so that descriptive breakdown differs (see REDACTIONS.md). The manuscript does not report it. The
unit tests (`python -m pytest test_*.py` in each folder, 97 tests; SP-16's need the SHARD code) pass.

Notes on the runs:

- KT-10's stream D7 was rerun in `runs/run_003_D7` after the first cloud instance stopped; `--extra` adds it.
- FP-12's first analysis, written on the GPU host before the correction described below, is kept as
  `runs/analysis_host_v1_practical_gate_vs_shuffles.json`.
- In XR-13, CRAN has fewer than 30 base-known facts and is reported without a damage measurement, as the rule fixed
  before the test runs requires.
- RC-15 ran on a different GPU host from FP-12 and XR-13. With `--ref`, its analysis compares the rewritten tables and
  the arms shared with the earlier runs item by item (field `reproduction`). The analyses first written on the GPU
  host, without that field, are kept as `runs/analysis_host_fp_test_001.json` and `runs/analysis_host_xr_test_001.json`.
- In BR-14, development stream D1 ran out of GPU memory with 16 facts per optimisation chunk; `config_dev_full_c8.json`
  (8 facts per chunk, nothing else changed) reran development from D0. `br_design.py` chose among the frozen designs
  `config_dev_full.json` and `config_dev_r1/r2/r3.json` from timings alone (`runs/design.json`).
- BR-14's exploratory run `runs/explore_f080_001` (floor 0.80) was specified after development stream D0 and before any
  test stream; `make_explore_config.py` derived its configuration from `config_test.json`.

### Code changes after a run

`run.json` records 334 code hashes. For 316 the published file is identical, including every hash recorded by the
RC-15, BR-14 and SP-16 runs. 10 belong to the SHARD code that SP-16 imported, which is not published here; they match
the pin in `shard_prospective_016/upstream/UPSTREAM.json`. The 8 others belong to files changed after the run that
recorded them:

- `key_thresholds_010/kt_run.py` (recorded in `run_002`): the options `--streams` and `--extra` were added, to rerun
  stream D7 and to analyse both run directories together. `kt_core.py`, `kt_analysis.py` and the RV-09 and SO-08 code
  are unchanged.
- `external_key_replication_013/xr_run.py` (recorded in `dev_001`): before any test stream ran, the script's folder was
  put on the import path, which `python -I` leaves out.
- `field_prefix_ablation_012/fp_analysis.py` (recorded in the FP-12 and XR-13 runs) and
  `external_key_replication_013/xr_analysis.py` (recorded in XR-13's `dev_001` and `test_001`): after the FP-12 test
  run, the practical gate's quality comparison was corrected in both. It had compared the matched thresholds with the
  permutations instead of with the selected uniform threshold. The corrected code wrote the included analyses; in
  FP-12 the correction turned one practical gate (corrected anchors, blocked demonstrations) from pass to fail and
  changed no other number. Before that, between XR-13's dev and test runs, `xr_analysis.py` had gained the rule for
  families with too few base-known facts.

## Re-running the experiments

The runs used one NVIDIA GeForce RTX 4090 (24 GB) per run, with Python 3.12.3, torch 2.14.0+cu130 and transformers
5.20.2. The inputs are pinned on the Hugging Face Hub and are not redistributed here:

| Input | Repository | Revision | Used by |
|---|---|---|---|
| WikiBigEdit | `lukasthede/WikiBigEdit` (dataset) | `8ddfff0ada5496ce1f0a4989b7a49dd0546be010` | all |
| Base model | `Qwen/Qwen3-0.6B-Base` | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | all |
| Retriever | `sentence-transformers/all-mpnet-base-v2` | `e8c3b32edf5434bc2275fc9bab85f82640a19130` | EC-11 |
| Instruction model | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | FP-12 annotator B; XR-13 write questions, probes and licence questions |
| Instruction model | `allenai/OLMo-2-1124-7B-Instruct` | `470b1fba1ae01581f270116362ee4aa1b97f4c84` | XR-13 evaluation wordings and span extractor |
| Second backbone | `allenai/OLMo-2-1124-7B` | `7df9a82518afdecae4e8c026b27adccc8c1f0032` | BR-14 |
| SHARD's backbone | `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | SP-16 |
| CounterFact | `https://rome.baulab.info/data/dsets/counterfact.json` | SHA-256 `d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f` | SP-16 |
| SHARD code | `https://github.com/Minds-R-Lab/ComplexAttn` (directory `sfib`) | commit `96ede1c0bf23195492fe97c7b0ecc5339184512d` | SP-16 |

`so_fetch.py` downloads the files of a pinned listing and checks each one against the official Hub listing (size and
SHA-256); no token is used. The listings the runs used are in `reversal_pilot_009/runs/host/acq/` (WikiBigEdit and
Qwen3), `external_key_replication_013/listings/` (the two instruction models) and
`backbone_replication_014/acquisition/` (OLMo-2-1124-7B); `editor_compare_011/ec_fetch.py`,
`external_key_replication_013/xr_fetch.py`, `field_prefix_ablation_012/annotation/fetch_annotator.py` and
`backbone_replication_014/br_fetch.py` fetch the other models. For example:

```
mkdir -p rec/data rec/model
cp experiments/reversal_pilot_009/runs/host/acq/data/listing.json rec/data/
cp experiments/reversal_pilot_009/runs/host/acq/model/listing.json rec/model/
python experiments/self_observation_008/so_fetch.py fetch data  DATA_DIR  rec/data
python experiments/self_observation_008/so_fetch.py fetch model MODEL_DIR rec/model
```

The frozen configurations keep the paths of the original machine (`E:/research/pause2/...`). The run scripts take
`RV_DATA_DIR` and `RV_MODEL_PATH` (and `EC_RETRIEVER_PATH` in EC-11) instead and record the substitution in
`run.json`, so the configuration files stay byte-identical. The full command lines are in the docstrings of
`kt_run.py`, `ec_run.py`, `fp_run.py` and `xr_run.py`; for example, in `experiments/key_thresholds_010`:

```
RV_DATA_DIR=DATA_DIR RV_MODEL_PATH=MODEL_DIR python kt_run.py run config_run_001.json runs/new_run
```

FP-12 reads the final entity spans from `annotation/spans_final.json`. `annotation/` also holds the rule-based
annotator A, the model-based annotator B, their spans, the disagreements, the adjudication brief and the decisions.
`annotation/human_audit/` holds the author's blind audit of the final spans, done after all experiments: with a fixed
seed, `make_audit.py` drew 100 of the 8,412 D0–D7 cases and, separately, 50 of the 510 adjudicated disagreements
(`audit_sample.json`); the sheets show only the question and the entity's Wikidata labels, aliases and description.
The author filled in `your_mention`, `status` and `note`, and `score_audit.py` compared them with the final spans
(`audit_result.json`). Row 100 of `audit_sheet.csv` was first left blank; its status, `ambiguous`, was entered on the
author's instruction without the final span being shown.
SP-16 runs the SHARD authors' code unchanged, imported from `shard_prospective_016/upstream/sfib`. That code is not
redistributed here (see Not included); fetch the pinned commit and check it with `verify_upstream.py`, whose docstring
gives the commands. `sp_fetch.py` downloads the model (listing in `acquisition/listing.json`) and CounterFact (checked
against the SHA-256 above), and `build_split.py` rebuilds `counterfact_roles.csv` byte for byte from CounterFact and
the WikiBigEdit snapshots. The run script takes `SP_MODEL_PATH` and `SP_COUNTERFACT` and records them in `run.json`;
`sp_run.py smoke`, `run`, `select` and `analyze` are described in its docstring. `annotation/blind_audit/` holds the
author's blind audit of SP-16's scoring, done after the test analysis: `make_audit.py` drew 140 continuations with a
fixed seed (strata in its docstring), the sheet shows only the prompt, the reference answer and the continuation, and
`audit_key.json` holds each row's arm, stream and automatic verdicts; `score_audit.py` wrote `audit_result.json`.
Three rows judged wrong carry no error type; they are left as the author wrote them.
RC-15 and BR-14 runs use `rc_run.py run CONFIG OUT --phase P` (see its docstring). For BR-14, `br_calibrate.py`
measured the Qwen3 reference norm (`calibration/qwen_norms.json`) by which `rc_core.py` scales the high-norm cut-off. XR-13, and RC-15's registry runs that reuse its streams, cannot be rerun
from this repository, because the streams are built from registry snapshots that are not redistributed. `xr_collect.py`, `xr_events.py`, `xr_questions.py`, `xr_extract.py` and `xr_build.py` are the code that
collected and built them.

## Not included

- WikiBigEdit and the model weights (public; see above).
- XR-13's registry snapshots, event files, generated questions and streams, and the registry-derived question texts
  inside its per-item records, because their redistribution terms have not been established
  ([REDACTIONS.md](REDACTIONS.md)).
- The study plans, reports and tables, which were written in Chinese for internal use and record run logistics, the
  cloud scripts and host logs, and runs the manuscript does not use (smoke tests, an aborted first KT-10 run, an XR-13
  debug stream, RC-15's smoke and debug runs, and BR-14's first development record, which ran out of memory). The
  report scripts (`*_report.py`, and RC-15's `rc_summary.py`) are included; they write Chinese-language tables. The
  Chinese instructions of SP-16's blind audit are not included either.
- The SHARD code that SP-16 imports. The pinned commit, the head of the repository's `addressable-memory` branch,
  carries no licence file. GitHub reports an MIT licence for the repository from its default branch, which holds a
  different project, and `upstream/UPSTREAM.json` records that report. Fetch the code from the original repository
  instead (see above).
- SP-16's written key-value tables (`runs/*/table_*.pt`, about 3 MB per stream). The seal and manifest records give
  their SHA-256, and `sp_run.py run` can write them again; whether other hardware reproduces them bit for bit has not
  been checked.

## How the code was written

The experiment code, analysis code and run scripts were written by Claude Opus 5.5 (Anthropic), working as a coding
agent at the author's request and under the author's direction, as the manuscript describes. In FP-12, two fresh
Claude Opus 5.5 sessions adjudicated the span disagreements; each saw only `annotation/ADJUDICATION_BRIEF.md` and its
items. The audit of the final spans in `annotation/human_audit/` was done by the author. SP-16's plan, including its
decision rules, was drafted with GPT-6.2 Astra (OpenAI) and fixed before its first model run, and the blind audit of
its scoring (`shard_prospective_016/annotation/blind_audit/`) was done by the author.

## License

Code: MIT ([LICENSE](LICENSE)). Our own records and documentation: CC BY 4.0. Third-party material keeps its own terms;
see [LICENSE-DATA.md](LICENSE-DATA.md).

## Citation

Please cite the manuscript. A full reference will be added here once it is published.
