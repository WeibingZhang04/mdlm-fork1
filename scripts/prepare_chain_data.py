#!/usr/bin/env python3
"""Prepare fresh hash-split OWT documents for the chain-CRF experiment.

Source can be an existing document-preserving HF cache, an identified JSONL,
or the pinned public OWT train split. All chunks of a source document have the
same split. Previously used evaluation documents are excluded before selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from chain_crf.backbone import file_sha256, load_tokenizer, TOKENIZER_REPOSITORY, TOKENIZER_REVISION
from chain_crf.data import atomic_json, atomic_torch_save, document_split, canonical_hash

OWT_REPOSITORY = "Skylion007/openwebtext"
OWT_REVISION = "79d93d786212f7344586290adb811d4ae6a1762c"


def prior_document_ids(paths):
    """Read only explicit identity fields; do not hash arbitrary output prose."""
    found = set()

    def visit(row):
        if isinstance(row, dict):
            for key, value in row.items():
                if key in {"document_id", "source_document_sha256"} and isinstance(value, str):
                    found.add(value)
                elif key in {"document_ids", "exclude_from_future_benchmark_document_ids"}:
                    if isinstance(value, list):
                        found.update(v for v in value if isinstance(v, str))
                elif isinstance(value, (dict, list)):
                    visit(value)
        elif isinstance(row, list):
            for value in row:
                visit(value)

    for path in paths:
        with Path(path).open() as f:
            if Path(path).suffix == ".jsonl":
                for line in f:
                    if line.strip():
                        visit(json.loads(line))
            else:
                visit(json.load(f))
    return found


def source_rows(source, cache_dir=None):
    if source:
        path = Path(source)
        if path.suffix == ".jsonl":
            def rows():
                with path.open() as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)
            return rows(), {"path": str(path.resolve()), "file_sha256": file_sha256(path),
                            "kind": "identified_jsonl"}
        from datasets import load_from_disk
        dataset = load_from_disk(str(path), keep_in_memory=False)
        sidecar = path.with_suffix(path.suffix + ".provenance.json")
        if not sidecar.is_file():
            raise ValueError("Cached OWT requires its provenance sidecar")
        provenance = json.loads(sidecar.read_text())
        spec = provenance.get("specification", {})
        import data_provenance
        data_provenance.validate_manifest(provenance, expected_specification=spec)
        if (spec.get("dataset_name_or_path") != OWT_REPOSITORY
                or spec.get("source_revision") != OWT_REVISION
                or spec.get("document_boundary_mode") != "source_document"
                or spec.get("tokenizer_revision") != TOKENIZER_REVISION):
            raise ValueError("Cache must be pinned OWT with original document boundaries/tokenizer")
        if not {"input_ids", "source_document_sha256"} <= set(dataset.column_names):
            raise ValueError("Cache lacks document identity")
        observed = provenance.get("observed", {})
        if (observed.get("processed_num_sequences") != len(dataset)
                or observed.get("processed_fingerprint") != dataset._fingerprint):
            raise ValueError("Cache contents/fingerprint differ from pinned provenance")
        return iter(dataset.with_format(None)), {
            "kind": "pinned_hf_cache", "path": str(path.resolve()),
            "provenance_sha256": file_sha256(sidecar), "source_provenance": provenance,
            "fingerprint": dataset._fingerprint, "rows": len(dataset),
        }
    from datasets import load_dataset
    dataset = load_dataset(OWT_REPOSITORY, "plain_text", revision=OWT_REVISION,
                           split="train", streaming=True, cache_dir=cache_dir)
    return iter(dataset), {"kind": "pinned_public_dataset", "repository": OWT_REPOSITORY,
                           "revision": OWT_REVISION, "split": "train"}


def prepare(rows, *, tokenizer, length, targets, exclusions=(), salt="chain-crf-20260925-v1",
            max_scan=10_000_000):
    if length < 2 or any(value < 1 for value in targets.values()):
        raise ValueError("Positive split counts and length >=2 required")
    selected = {split: [] for split in targets}
    documents = {split: [] for split in targets}
    seen = set()
    excluded = set(exclusions)
    scanned = 0
    for scanned, row in enumerate(rows, 1):
        if scanned > max_scan:
            break
        doc = row.get("document_id", row.get("source_document_sha256"))
        if "input_ids" in row:
            if not isinstance(doc, str) or not doc:
                raise ValueError("Tokenized input must retain the original document identity")
            ids = row["input_ids"]
            # Partition each existing document-preserving cache row without
            # joining rows or moving its BOS/EOS markers. Every output keeps
            # the original document ID and therefore the same hash split.
            chunks = [ids[i:i+length] for i in range(0, len(ids)-length+1, length)]
        else:
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError("Source row needs text or input_ids")
            # Same UTF-8 source-document hash as the existing pinned cache.
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if doc is not None and doc != content_hash:
                raise ValueError("Text document hash mismatch")
            doc = content_hash
            if doc in excluded:
                continue
            split = document_split(doc, salt)
            if len(selected[split]) >= targets[split]:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            payload_length = length - 2
            if payload_length < 1:
                raise ValueError("Wrapped text requires length >=3")
            chunks = [[tokenizer.bos_token_id] + ids[i:i+payload_length] + [tokenizer.eos_token_id]
                      for i in range(0, len(ids) - payload_length + 1, payload_length)]
        if doc in excluded:
            continue
        split = document_split(doc, salt)
        for chunk in chunks:
            if len(selected[split]) >= targets[split]:
                break
            if len(chunk) < length:
                continue
            chunk = chunk[:length]
            if any(type(token) is not int or not 0 <= token < tokenizer.vocab_size for token in chunk):
                raise ValueError("Clean source contains non-token integers or invalid token IDs")
            digest = canonical_hash(chunk)
            if digest in seen:
                continue
            seen.add(digest)
            selected[split].append(chunk)
            documents[split].append(doc)
        if all(len(selected[split]) >= count for split, count in targets.items()):
            break
    counts = {split: len(value) for split, value in selected.items()}
    if counts != targets:
        raise ValueError(f"Insufficient eligible documents: {counts}; requested {targets}; scanned {scanned}")
    for first in selected:
        for second in selected:
            if first != second and set(documents[first]) & set(documents[second]):
                raise AssertionError("Hash split document overlap")
    return selected, documents, {"source_rows_scanned": scanned, "split_examples": counts,
                                "excluded_document_count": len(excluded),
                                "split_documents": {s: len(set(d)) for s, d in documents.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Pinned HF document cache or identified JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--train-examples", type=int, default=40000)
    parser.add_argument("--dev-examples", type=int, default=128)
    parser.add_argument("--test-examples", type=int, default=256)
    parser.add_argument("--split-salt", default="chain-crf-20260925-v1")
    parser.add_argument("--max-scan", type=int, default=10000000)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    old = set(args.exclude)
    for basename in ("train.jsonl", "dev.jsonl", "test.jsonl", "manifest.json"):
        old.update((ROOT / "artifacts/paper/staged-debugging-v1").glob(f"**/{basename}"))
    exclusions = prior_document_ids(sorted(old))
    tokenizer = load_tokenizer(args.cache_dir)
    rows, source = source_rows(args.source, args.cache_dir)
    targets = {"train": args.train_examples, "dev": args.dev_examples, "test": args.test_examples}
    selected, documents, report = prepare(
        rows, tokenizer=tokenizer, length=args.length, targets=targets,
        exclusions=exclusions, salt=args.split_salt, max_scan=args.max_scan)
    provenance = {"schema": "chain_crf_data_v1", "source": source,
                  "tokenizer": TOKENIZER_REPOSITORY, "tokenizer_revision": TOKENIZER_REVISION,
                  "length": args.length, "split_salt": args.split_salt,
                  "split_rule": "sha256(salt:original_document_hash) mod 10000: dev<100,test<200,else train",
                  "excluded_ids_sha256": canonical_hash(sorted(exclusions)),
                  "excluded_sources": {str(p): file_sha256(p) for p in sorted(old)},
                  "selection": "source order after document hash assignment; no model scoring",
                  "tokenized_row_policy": "nonoverlapping length-sized blocks within each source row; no row joining; drop incomplete tail",
                  "document_disjoint": True, "report": report}
    args.output.mkdir(parents=True, exist_ok=False)
    for split in targets:
        atomic_torch_save({"tokens": torch.tensor(selected[split], dtype=torch.long),
                           "document_ids": documents[split],
                           "provenance": {**provenance, "split": split}}, args.output / f"{split}.pt")
        with (args.output / f"{split}.jsonl").open("x") as f:
            for ids, doc in zip(selected[split], documents[split]):
                f.write(json.dumps({"input_ids": ids, "document_id": doc, "split": split}) + "\n")
    provenance["files"] = {p.name: file_sha256(p) for p in args.output.iterdir() if p.is_file()}
    atomic_json(provenance, args.output / "manifest.json")
    print(json.dumps({"event": "prepared", "output": str(args.output), **report}), flush=True)


if __name__ == "__main__":
    main()
