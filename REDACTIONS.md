# Redactions for the public release

Every file under `experiments/` is a byte-for-byte copy of the archived file, except the 13 files below. No number
in the manuscript depends on what was removed.

**Registry-derived question texts (XR-13).** The per-item records of the software-registry study held the locality
questions that a language model wrote from registry metadata, in two places: the field `loc_questions` and the second
element of every locality item in `items`. The manuscript states that the registry-derived question texts and raw API
snapshots are not redistributed, because their redistribution terms have not been established. So `loc_questions` is
now an empty object, the question text in each locality item of `items` is an empty string (the build checked that
each removed text equals the `loc_questions` entry of that item), and a field `public_release_note` says so. The records
were re-serialised exactly as `fp_run.write_gz` writes them (compact JSON, gzip with mtime 0); before the edit, each
record was checked to round-trip to its original bytes, so nothing else changed. Kept: the gold licence labels
(`loc_answers`), the package identifiers in the item IDs, the extracted anchor spans (`tables.entity`) and the base
model's answers (`wrong_loc` and the intervention answers), which sometimes repeat a package name.

Consequences, checked on 2026-11-22: `checks/recount.py` reproduces its archived output exactly, and `xr_run.py analyze`
reproduces the archived `analysis.json` except the culprit categories of the single-key interventions
(`interventions.<arm>.<family>.culprits` and `.per_item`, 96 values). The category `near name` compares a culprit's
anchor with the question text, so without the texts those culprits count as `other`, and items whose culprits were of
several kinds then count once under `other`. That breakdown is descriptive and is not reported in the manuscript.
`xr_posthoc.py` needs the registry streams and cannot be rerun; its output, `checks/posthoc_test_001.json`, is included.
The run manifests (`manifest.json`) keep the hashes of the original records; the table below gives both.

**Cloud account details (KT-10).** The free-text note in `manifest_local.json` named the rented instance and an
unpublished cloud script. The note now describes the same events without them; the hashes the file lists are unchanged.

| File | SHA-256 before | SHA-256 after | Change |
|---|---|---|---|
| `experiments/key_thresholds_010/runs/run_002/manifest_local.json` | `794010abc86c9bc08ee948d349cb9c7f3f880021b9c104eb849a062335895e05` | `75fd2aabb704869f0499737364d7c63d121739dcc7613ddc4a623bd8c73acafc` | instance number and the name of an unpublished cloud script removed from the free-text `note` |
| `experiments/external_key_replication_013/data/host/runs/dev_001/stream_cran_dev.json.gz` | `3f37e13a34facc3feb7e7aafef3fe7312d39ca4eecaa17f83794cf25ba195686` | `63c5990d9bd227c32e21783e01ac1e75fccb21efcb719059bbeada999cec59e8` | 1 registry-derived question texts removed (`loc_questions` emptied, the text in each locality item of `items` blanked); `public_release_note` added |
| `experiments/external_key_replication_013/data/host/runs/dev_001/stream_crates_dev.json.gz` | `4e9a75681962863330d679d7ce7de5c2733cca37db1a17f60b8996f76bf0776f` | `65c30b2fe5866f37f90bfc84d4f81f086d90439d6058706a14cf76fca0d429e7` | 515 registry-derived question texts removed (`loc_questions` emptied, the text in each locality item of `items` blanked); `public_release_note` added |
| `experiments/external_key_replication_013/data/host/runs/dev_001/stream_npm_dev.json.gz` | `790d417966b5573bdfde3d6f0634d89656ede5c24a6e2ec591c4363ef09e0c0d` | `73a3f11f80b32175012e48efb9d7f6b6d823f73fe3448e03e8617126a4379203` | 2417 registry-derived question texts removed (`loc_questions` emptied, the text in each locality item of `items` blanked); `public_release_note` added |
| `experiments/external_key_replication_013/data/host/runs/dev_001/stream_pypi_dev.json.gz` | `4c5c7dff7af0d7068681e68e3e0dd9c3f97ccab7771ee42115d3c8757c02b8d7` | `c1b48add5fe2054f27d909e1fc70985de3ae78badf8312b840ad09746d1f45da` | 683 registry-derived question texts removed (`loc_questions` emptied, the text in each locality item of `items` blanked); `public_release_note` added |
| `experiments/external_key_replication_013/data/host/runs/test_001/stream_cran.json.gz` | `6ae198953864bcafd4bd4bf259a8eca725c9a0fa68318c5cb28a0a8e06459f10` | `8fbefda9e0fcd81d3131a362518bef163fc67329776183d77228912c7c3f3c2f` | 0 registry-derived question texts removed (`loc_questions` emptied, the text in each locality item of `items` blanked); `public_release_note` added |
| `experiments/external_key_replication_013/data/host/runs/test_001/stream_crates.json.gz` | `0b2e358ca9bb78f43f18622d3e8f1132f5d2605864d5d00c1a1c2b4171bc5e1f` | `cb8ffdf996d5f2864c8dad32d36f8b6a5c75bb3b523722cca332c4446cf03c17` | 502 registry-derived question texts removed (`loc_questions` emptied, the text in each locality item of `items` blanked); `public_release_note` added |
| `experiments/external_key_replication_013/data/host/runs/test_001/stream_hackage.json.gz` | `6d99e924003ddf5a1d7aa3c348bfc34e010e77afc1aa257ccaf1f1a0f8fd3755` | `fed1b9968e4c4697edb2983529da8e027f6649d994d733f28d83175c98709829` | 346 registry-derived question texts removed (`loc_questions` emptied, the text in each locality item of `items` blanked); `public_release_note` added |
| `experiments/external_key_replication_013/data/host/runs/test_001/stream_homebrew.json.gz` | `5393bbaef0199671e36e2a510874ab6f0b8e7aacf4e0614a77b58acb58580048` | `b9054706e88c1042a7d5e9d46479059481fc6dbcdd126f3eeebbf5af5f6c4dfb` | 42 registry-derived question texts removed (`loc_questions` emptied, the text in each locality item of `items` blanked); `public_release_note` added |
| `experiments/external_key_replication_013/data/host/runs/test_001/stream_npm.json.gz` | `55f6b360f70f151e1d9cdb9a0265787b5486bb579169f019325ff774e83f2d24` | `235777a6ad483cabcb380500047171b380c3aef9c62c839dc90257796feb1c12` | 2361 registry-derived question texts removed (`loc_questions` emptied, the text in each locality item of `items` blanked); `public_release_note` added |
| `experiments/external_key_replication_013/data/host/runs/test_001/stream_nuget.json.gz` | `5a46fe246a1d23ae83d43608de34c575904b656778ae11853fa9ed58589023c7` | `1d8b5a1589ab1cabe43252134a59038dc3ab0bdb652f9ba3cc67eb6769822427` | 280 registry-derived question texts removed (`loc_questions` emptied, the text in each locality item of `items` blanked); `public_release_note` added |
| `experiments/external_key_replication_013/data/host/runs/test_001/stream_packagist.json.gz` | `47a96e2cbd23aee210069d4b791fa2d9670ca4c167aed65c336e2fda73a65274` | `fd8b1a123bafdbb5ca3b34fa8ff5fe617988b3dafbd9a13936d12e91e088a813` | 1957 registry-derived question texts removed (`loc_questions` emptied, the text in each locality item of `items` blanked); `public_release_note` added |
| `experiments/external_key_replication_013/data/host/runs/test_001/stream_pypi.json.gz` | `dd7349cf691c1db8c2573c086018b75a63cf0362c6f133d70639148c52c09f75` | `6c6bb88be167aafcc2ac560ac1c77da8280d4a53e2c42d41dffdee6ee284e36b` | 607 registry-derived question texts removed (`loc_questions` emptied, the text in each locality item of `items` blanked); `public_release_note` added |
