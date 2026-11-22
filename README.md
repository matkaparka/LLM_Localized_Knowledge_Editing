# LLM_Localized_Knowledge_Editing

Code, frozen configurations and per-item outcome records for the manuscript

> **When memory keys fire on the wrong question: Diagnosing collateral damage in continual knowledge editing**
> Lihan Xu, 2026

The manuscript studies a routed residual memory editor of the GRACE family on Qwen3-0.6B-Base. Each stored edit is a
key at layer 8 with its own firing threshold, calibrated on unlabelled background questions. Permuting the thresholds
within write batches tests whether it matters which key receives which threshold; replays with single keys enabled or
disabled link damaged answers to the entries that cause them. The manuscript itself is not part of this repository.

## Contents

| Study (label in the manuscript) | What it tests | Folder under `experiments/` |
|---|---|---|
| Assignment test (KT-10) | Matched against permuted and uniform thresholds on WikiBigEdit streams D4–D7 (D0–D3 for tuning) | `key_thresholds_010/` |
| Factorial attribution (FP-12) | Entity anchor (original or corrected) crossed with routing on shared demonstrations (allowed or blocked), on the same streams | `field_prefix_ablation_012/` |
| New-source test (XR-13) | The threshold test on questions about packages in eight software registries | `external_key_replication_013/` |
| Systems comparison (EC-11) | Probe-calibrated keys against a GRACE reimplementation, radius rules, subject cleaning with a uniform threshold, and retrieval, on streams C0–C7 | `editor_compare_011/` |

Two earlier folders contribute modules that the four studies import unchanged. Neither earlier study is reported in
the manuscript, and only what the four studies need is included:

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
  outputs), one gzipped JSON record per stream (`stream_<name>.json.gz`) and `analysis.json`;
- `checks/`: an independent recount of the main numbers (`recount.py`, which shares no code with the analysis), its
  output, and other descriptive checks.

The stream records hold, for every item (an edit's original wording and rephrasings, earlier edits, and base-known
locality facts) and every threshold arm, whether the answer was right, which memory entries fired, the model's wrong
answers, and the single-key interventions. The docstrings of the analysis modules (`kt_analysis.py`, `ec_analysis.py`,
`fp_analysis.py`, `xr_analysis.py`) define the measures.

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
```

`analyze` overwrites the `analysis.json` of the run it reads. On 2026-11-22 these commands, run on a copy of this
repository, regenerated every included `analysis.json` and recount output exactly, with one exception: in XR-13 the
culprit categories of the single-key interventions use question texts that are not published, so that descriptive
breakdown differs (see REDACTIONS.md). The manuscript does not report it. The unit tests (`python -m pytest test_*.py`
in each folder, 80 tests) pass.

Notes on the runs:

- KT-10's stream D7 was rerun in `runs/run_003_D7` after the first cloud instance stopped; `--extra` adds it.
- FP-12's first analysis, written on the GPU host before the correction described below, is kept as
  `runs/analysis_host_v1_practical_gate_vs_shuffles.json`.
- In XR-13, CRAN has fewer than 30 base-known facts and is reported without a damage measurement, as the rule fixed
  before the test runs requires.

### Code changes after a run

`run.json` records 114 code hashes. For 106 the published file is identical; the 8 others belong to files changed
after the run that recorded them:

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

`so_fetch.py` downloads the files of a pinned listing and checks each one against the official Hub listing (size and
SHA-256); no token is used. The listings the runs used are in `reversal_pilot_009/runs/host/acq/` (WikiBigEdit and
Qwen3) and `external_key_replication_013/listings/` (the two instruction models); `editor_compare_011/ec_fetch.py`,
`external_key_replication_013/xr_fetch.py` and `field_prefix_ablation_012/annotation/fetch_annotator.py` fetch the
other models. For example:

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
XR-13 cannot be rerun from this repository, because its streams are built from registry snapshots that are not
redistributed. `xr_collect.py`, `xr_events.py`, `xr_questions.py`, `xr_extract.py` and `xr_build.py` are the code that
collected and built them.

## Not included

- WikiBigEdit and the model weights (public; see above).
- XR-13's registry snapshots, event files, generated questions and streams, and the registry-derived question texts
  inside its per-item records, because their redistribution terms have not been established
  ([REDACTIONS.md](REDACTIONS.md)).
- The study plans, reports and tables, which were written in Chinese for internal use and record run logistics, the
  cloud scripts and host logs, and runs the manuscript does not use (smoke tests, an aborted first KT-10 run, an XR-13
  debug stream). The report scripts (`*_report.py`) are included; they write Chinese-language tables.

## How the code was written

The experiment code, analysis code and run scripts were drafted by AI coding assistants (Anthropic Claude models in a
coding-agent interface) at the author's request and under the author's direction, as the manuscript describes. In
FP-12, two fresh Claude subagents adjudicated the span disagreements; each saw only
`annotation/ADJUDICATION_BRIEF.md` and its items.

## License

Code: MIT ([LICENSE](LICENSE)). Our own records and documentation: CC BY 4.0. Third-party material keeps its own terms;
see [LICENSE-DATA.md](LICENSE-DATA.md).

## Citation

Please cite the manuscript. A full reference will be added here once it is published.
