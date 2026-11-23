"""Build the CounterFact audit manifest from source metadata, without model outputs."""

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path


VERSION = "KBS-SHARD-CF-20261123-v1"
SOURCE_SHA = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
FIELDS = ["partition", "stream", "role", "ordinal", "case_id", "relation_id"]


def norm(value):
    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def priority(namespace, value):
    return hashlib.sha256(f"{VERSION}|{namespace}|{value}".encode()).hexdigest()


def past_exposure(directory):
    subjects, prompts = set(), set()
    paths = sorted(directory.glob("wiki_big_edit_*.json"))
    if len(paths) != 8:
        raise ValueError("Expected the eight archived WikiBigEdit snapshots")
    for path in paths:
        for item in json.loads(path.read_bytes()):
            subjects.add(norm(item["subject"]))
            prompts.update(norm(item.get(k)) for k in ("update", "rephrase", "personas", "loc", "mhop"))
    return subjects, prompts


def clean_cases(raw, old_subjects, old_prompts):
    # Stable choice of at most one case for any normalized subject.
    subjects = set(old_subjects)
    prompts = set(old_prompts)
    for row in sorted(raw, key=lambda r: priority("candidate", r["case_id"])):
        rewrite = row["requested_rewrite"]
        subject = norm(rewrite["subject"])
        texts = {norm(rewrite["prompt"].format(rewrite["subject"]))}
        texts.update(norm(p) for p in row["paraphrase_prompts"])
        if not subject or subject in subjects or len(texts) != 3 or "" in texts or texts & prompts:
            continue
        subjects.add(subject)
        prompts.update(texts)
        yield row


def make_manifest(cases):
    # Assign roles by an independent hash; no sorting on answers or correctness.
    cells = []
    for partition, initial, count in [("dev", "D", 2), ("test", "T", 8)]:
        cells.extend((partition, "shared", "probe", i + 1) for i in range(800))
        for stream in range(1, count + 1):
            for role, size in [("edit", 250), ("protect", 800)]:
                cells.extend((partition, f"{initial}{stream:02d}", role, i + 1) for i in range(size))
    cells.sort(key=lambda x: priority("cell", "|".join(map(str, x))))
    if len(cases) < len(cells):
        raise ValueError(f"Only {len(cases)} eligible cases for {len(cells)} slots")
    rows = []
    for (part, stream, role, ordinal), case in zip(cells, cases):
        rewrite = case["requested_rewrite"]
        rows.append(dict(partition=part, stream=stream, role=role, ordinal=ordinal,
                         case_id=case["case_id"], relation_id=rewrite["relation_id"]))
    return sorted(rows, key=lambda r: (r["partition"], r["stream"], r["role"], r["ordinal"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("counterfact", type=Path)
    parser.add_argument("wikibigedit_directory", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    data = args.counterfact.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA:
        raise ValueError("CounterFact checksum mismatch")
    raw = json.loads(data)
    old_subjects, old_prompts = past_exposure(args.wikibigedit_directory)
    cases = list(clean_cases(raw, old_subjects, old_prompts))
    rows = make_manifest(cases)
    with args.output_csv.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Source {len(raw)}; eligible distinct surface forms {len(cases)}; assigned {len(rows)}")
    print(dict(Counter((r["partition"], r["role"]) for r in rows)))
    print("Manifest SHA256", hashlib.sha256(args.output_csv.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
