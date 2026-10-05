#!/usr/bin/env python3
"""Prepare fresh, document-disjoint MDLM-style packed OWT subsets.

Protocol and rationale: notes/mdlm-crf-1024-decisions.md.
Select documents BEFORE packing. The published final-100k split rule is applied
to a pinned public source; that is not proof of the released checkpoint's exact
historical source bytes/order. Resolve that audit before making exclusion claims.

Document quotas are REQUIRED: no smoke-test budget is silently reused. This
script prepares data only; it never loads MDLM weights or starts training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from chain_crf.backbone import file_sha256, load_tokenizer, TOKENIZER_REPOSITORY, TOKENIZER_REVISION
from chain_crf.data import atomic_json, atomic_torch_save, canonical_hash, source_document_set
from scripts.prepare_chain_data import OWT_REPOSITORY, OWT_REVISION, prior_document_ids

SOURCE_ROWS = 8013769
HELDOUT_DOCUMENTS = 100000
SCHEMA = "chain_crf_packed_data_v2"
ROLES = ("dev", "train", "heldout_eval")


def text_identity(dataset, index):
    text = dataset[index]["text"]
    if not isinstance(text, str):
        raise ValueError(f"Source row {index} is not text")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def shuffled_indices(size, seed):
    """Lazy Fisher-Yates using SHA256 counter draws, independent of Python RNG.

    Selection is by source document index, not chunk length or model scores.
    Rejection avoids modulo bias. At most O(visited documents) indices are kept.
    """
    swaps, counter = {}, 0
    for remaining in range(size, 0, -1):
        limit = 2**256 - (2**256 % remaining)
        while True:
            raw = hashlib.sha256(f"{seed}:{counter}".encode("utf-8")).digest()
            counter += 1
            draw = int.from_bytes(raw, "big")
            if draw < limit:
                break
        position = draw % remaining
        result = swaps.get(position, position)
        swaps[position] = swaps.get(remaining - 1, remaining - 1)
        swaps.pop(remaining - 1, None)
        yield result


def select_documents(dataset, *, quotas, seed, heldout_documents=HELDOUT_DOCUMENTS,
                     exclusions=(), evaluation_population="sampled"):
    """Hold out the source tail; draw dev then train from the rest without replacement.

    Keeping the dev quota/seed fixed makes a larger train subset extend the same
    document order. Exact duplicates of ANY tail document are excluded from head
    train/dev, not only duplicates of the selected evaluation subset. Full-tail
    evaluation retains every original row, in source order, including duplicate
    and blank documents and prior-experiment overlaps; these are audited, not
    removed. Deduplication of head data uses exact UTF-8 content hashing.
    """
    if set(quotas) != set(ROLES) or any(type(n) is not int or n < 1 for n in quotas.values()):
        raise ValueError("Need positive document quotas for train, dev, heldout_eval")
    if not 0 < heldout_documents < len(dataset):
        raise ValueError("Invalid held-out source tail")
    if evaluation_population not in ("sampled", "full-tail"):
        raise ValueError("Unknown evaluation population")
    if evaluation_population == "full-tail" and quotas["heldout_eval"] != heldout_documents:
        raise ValueError("Full-tail evaluation requires the entire validation document quota")
    train_stop = len(dataset) - heldout_documents
    excluded = set(exclusions)
    heldout_hashes = set()
    tail_documents, tail_audit = [], Counter()
    for index in range(train_stop, len(dataset)):
        text, digest = text_identity(dataset, index)
        tail_audit["empty_documents"] += int(not text.strip())
        tail_audit["duplicate_content_documents"] += int(digest in heldout_hashes)
        tail_audit["prior_experiment_overlap_documents"] += int(digest in excluded)
        heldout_hashes.add(digest)
        if evaluation_population == "full-tail":
            tail_documents.append({"document_id": digest, "source_document_index": index,
                                   "split": "heldout_eval"})
    selected = {role: [] for role in ROLES}
    used, rejected = set(excluded), Counter()
    for index in shuffled_indices(train_stop, seed + ":head-documents"):
        text, digest = text_identity(dataset, index)
        if not text.strip():
            rejected["empty_training_document"] += 1
            continue
        if digest in used or digest in heldout_hashes:
            rejected["training_duplicate_excluded_or_in_validation"] += 1
            continue
        role = "dev" if len(selected["dev"]) < quotas["dev"] else "train"
        selected[role].append({"document_id": digest, "source_document_index": index,
                               "split": role})
        used.add(digest)
        if len(selected["train"]) == quotas["train"]:
            break
    if evaluation_population == "full-tail":
        selected["heldout_eval"] = tail_documents
    else:
        for offset in shuffled_indices(heldout_documents, seed + ":evaluation-documents"):
            index = train_stop + offset
            text, digest = text_identity(dataset, index)
            if not text.strip() or digest in used:
                rejected["evaluation_empty_duplicate_or_excluded"] += 1
                continue
            selected["heldout_eval"].append({"document_id": digest,
                                             "source_document_index": index,
                                             "split": "heldout_eval"})
            used.add(digest)
            if len(selected["heldout_eval"]) == quotas["heldout_eval"]:
                break
    counts = {role: len(rows) for role, rows in selected.items()}
    if counts != quotas:
        raise ValueError(f"Insufficient eligible source documents: {counts}; requested {quotas}")
    return selected, {"selected_documents": counts, "rejections": dict(rejected),
                      "validation_tail_unique_hashes": len(heldout_hashes),
                      "validation_tail_audit": dict(tail_audit),
                      "evaluation_population": evaluation_population,
                      "selection_sha256": canonical_hash(selected)}


def packed_rows(dataset, documents, tokenizer, *, length, report):
    """EOS-separated stream, L-2 payload, outer BOS/EOS, one final dropped tail.

    Unlike datasets.map's worker/batch-dependent remainder dropping, this stream
    crosses processing batches and drops a tail only at the end of each split.
    This matches MDLM's token/boundary convention, not its exact historical rows.
    Source spans include a document's appended EOS at offset len(raw_tokens).
    Outer BOS/EOS are synthetic row boundaries and have no source-document span.
    """
    if length < 3:
        raise ValueError("Packed rows need length >= 3")
    payload, spans = [], []
    report.update(selected_documents=len(documents), source_text_tokens=0,
                  source_eos_tokens=0, rows=0, row_boundary_tokens=0,
                  packed_text_tokens=0, packed_source_eos_tokens=0,
                  dropped_tail_tokens=0, dropped_tail_spans=[])
    for document in documents:
        text, digest = text_identity(dataset, document["source_document_index"])
        if digest != document["document_id"]:
            raise ValueError("Selected source document changed before tokenization")
        ids = tokenizer.encode(text, add_special_tokens=False)
        if any(type(token) is not int or not 0 <= token < tokenizer.vocab_size for token in ids):
            raise ValueError("Tokenizer returned invalid clean token IDs")
        raw_length = len(ids)
        ids = ids + [tokenizer.eos_token_id]
        report["source_text_tokens"] += raw_length
        report["source_eos_tokens"] += 1
        offset = 0
        while offset < len(ids):
            take = min(length - 2 - len(payload), len(ids) - offset)
            start = len(payload) + 1
            spans.append({**document, "row_token_start": start, "row_token_stop": start + take,
                          "source_token_start": offset, "source_token_stop": offset + take,
                          "source_text_length": raw_length})
            payload.extend(ids[offset:offset + take])
            offset += take
            if len(payload) == length - 2:
                source_eos = sum(span["source_token_stop"] > span["source_text_length"] for span in spans)
                report["packed_source_eos_tokens"] += source_eos
                report["packed_text_tokens"] += len(payload) - source_eos
                report["rows"] += 1
                report["row_boundary_tokens"] += 2
                yield {"input_ids": [tokenizer.bos_token_id] + payload + [tokenizer.eos_token_id],
                       "document_ids": list(dict.fromkeys(span["document_id"] for span in spans)),
                       "source_spans": spans}
                payload, spans = [], []
    report["dropped_tail_tokens"] = len(payload)
    report["dropped_tail_spans"] = spans
    report["packed_tokens"] = report["rows"] * length


def write_split(output, role, rows, *, length, provenance, row_roles):
    """Stream rows to JSONL and a disk-backed tensor; do not retain all token lists."""
    docs, digest, count = [], hashlib.sha256(), 0
    with tempfile.TemporaryDirectory(prefix=f".{role}-", dir=output) as temporary:
        binary = Path(temporary) / "tokens.bin"
        with (output / f"{role}.jsonl").open("x") as handle, binary.open("xb") as raw:
            for count, row in enumerate(rows, 1):
                encoded = struct.pack(f"<{length}q", *row["input_ids"])
                row_hash = hashlib.sha256(encoded).hexdigest()
                previous_role = row_roles.setdefault(row_hash, role)
                if previous_role != role:
                    raise ValueError(f"Identical packed token row in {previous_role} and {role}; "
                                     "inspect source duplication before accepting this dataset")
                digest.update(encoded)
                raw.write(encoded)
                docs.append(row["document_ids"])
                handle.write(json.dumps({"row_id": count - 1, "split": role, **row}) + "\n")
                if count % 10000 == 0:
                    print(json.dumps({"event": "packing_progress", "split": role,
                                      "rows": count}), flush=True)
        if not count:
            raise ValueError(f"{role}: selected documents yield no full packed rows")
        if sys.byteorder != "little":
            raise ValueError("Tensor export currently requires a little-endian host")
        tokens = torch.from_file(str(binary), size=count * length, dtype=torch.long).reshape(count, length)
        atomic_torch_save({"tokens": tokens, "document_ids": docs,
                           "provenance": {**provenance, "split": role}}, output / f"{role}.pt")
        del tokens
    return {"rows": count, "documents_with_retained_tokens": len(source_document_set(docs)),
            "tokens_sha256_le_int64": digest.hexdigest()}


def prepare_bundle(dataset, tokenizer, output, *, quotas, seed,
                   heldout_documents=HELDOUT_DOCUMENTS, length=1024,
                   exclusions=(), source=None, exclusion_sources=None,
                   evaluation_population="sampled"):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    selected, selection_report = select_documents(dataset, quotas=quotas, seed=seed,
        heldout_documents=heldout_documents, exclusions=exclusions,
        evaluation_population=evaluation_population)
    print(json.dumps({"event": "selected_documents", **selection_report}), flush=True)
    train_stop = len(dataset) - heldout_documents
    provenance = {"schema": SCHEMA, "length": length, "source": source or {"kind": "test_fixture"},
        "tokenizer": TOKENIZER_REPOSITORY, "tokenizer_revision": TOKENIZER_REVISION,
        "seed": seed, "document_quotas": quotas,
        "source_training_window": [0, train_stop], "source_validation_window": [train_stop, len(dataset)],
        "split_roles": {"train": "head optimization and count fitting", "dev": "head checkpoint and configuration selection",
                        "heldout_eval": "reserved evaluation from the published MDLM validation-tail rule"},
        "selection": "sha256 counter Fisher-Yates on training indices; dev first, then train",
        "evaluation_selection": ("complete validation tail in original source order; no filtering"
                                 if evaluation_population == "full-tail" else
                                 "separate seeded tail permutation; exclusions and exact deduplication"),
        "selection_report": selection_report,
        "exact_document_deduplication": {"train": True, "dev": True,
                                         "heldout_eval": evaluation_population != "full-tail"},
        "near_document_deduplication": False,
        "prior_exclusion_roles": ["train", "dev"] if evaluation_population == "full-tail" else list(ROLES),
        "excluded_ids_sha256": canonical_hash(sorted(exclusions)),
        "excluded_sources": exclusion_sources or {},
        "boundary_policy": {"bos_id": tokenizer.bos_token_id, "eos_id": tokenizer.eos_token_id,
            "payload_tokens": length - 2, "append_eos_per_document": True,
            "outer_bos_eos_per_row": True, "tail_policy": "drop one incomplete payload per split",
            "packing_order": "selected document order", "attention_reset_at_eos": False,
            "pair_edges_at_eos": "retain adjacent factors involving the separator; never connect different rows"},
        "preparation_source_sha256": file_sha256(Path(__file__)),
        "data_loader_sha256": file_sha256(ROOT / "chain_crf/data.py")}
    output.mkdir(parents=True, exist_ok=False)
    # Presence of manifest.json, written LAST, is the completion marker.
    atomic_json(selected, output / "selected_documents.json")
    summaries, row_roles = {}, {}
    for role in ROLES:
        report = {}
        rows = packed_rows(dataset, selected[role], tokenizer, length=length, report=report)
        result = write_split(output, role, rows, length=length, provenance=provenance, row_roles=row_roles)
        summaries[role] = {**report, **result}
        print(json.dumps({"event": "packed_split", "split": role,
                          "rows": result["rows"], "tokens": report["packed_tokens"]}), flush=True)
    atomic_json({**provenance, "splits": summaries,
                 "files": {path.name: file_sha256(path) for path in sorted(output.iterdir()) if path.is_file()}},
                output / "manifest.json")
    return selected, summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-documents", type=int, required=True)
    parser.add_argument("--dev-documents", type=int, required=True)
    parser.add_argument("--heldout-documents", type=int, required=True,
                        help="Evaluation document count; must be 100,000 for full-tail mode")
    parser.add_argument("--evaluation-population", choices=("full-tail", "sampled"), required=True,
                        help="Full-tail preserves all validation documents and their source order")
    parser.add_argument("--seed", required=True, help="Dataset selection seed, independent of training seeds")
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--tokenizer-cache", type=Path)
    parser.add_argument("--exclude", type=Path, action="append", required=True,
                        help="Repeat for previous train/dev/test identity files; no automatic globbing")
    parser.add_argument("--source-audit", type=Path,
                        help="Optional evidence about historical source/split alignment, copied into the manifest")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    quotas = {"train": args.train_documents, "dev": args.dev_documents, "heldout_eval": args.heldout_documents}
    if (min(quotas.values()) < 1 or quotas["train"] + quotas["dev"] > SOURCE_ROWS - HELDOUT_DOCUMENTS
            or quotas["heldout_eval"] > HELDOUT_DOCUMENTS):
        raise ValueError("Document quotas exceed source partitions or are nonpositive")
    if not args.seed:
        raise ValueError("Dataset seed must be nonempty")
    if args.evaluation_population == "full-tail" and args.heldout_documents != HELDOUT_DOCUMENTS:
        raise ValueError("Full-tail evaluation requires --heldout-documents 100000")
    excluded = prior_document_ids(args.exclude)
    exclusion_sources = {str(p.resolve()): file_sha256(p) for p in args.exclude}
    if not excluded:
        raise ValueError("Exclusion files contain no document identities")
    audit = None
    if args.source_audit:
        audit = {"file_sha256": file_sha256(args.source_audit),
                 "record": json.loads(args.source_audit.read_text())}
    from datasets import load_dataset
    print(json.dumps({"event": "loading_pinned_source", "repository": OWT_REPOSITORY,
                      "revision": OWT_REVISION, "may_download_full_source": True}), flush=True)
    dataset = load_dataset(OWT_REPOSITORY, "plain_text", revision=OWT_REVISION,
        split="train", streaming=False, cache_dir=str(args.dataset_cache), trust_remote_code=False)
    if len(dataset) != SOURCE_ROWS or dataset.column_names != ["text"]:
        raise ValueError("Pinned OWT source row count or schema differs")
    tokenizer = load_tokenizer(args.tokenizer_cache)
    if (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.vocab_size) != (50256, 50256, 50257):
        raise ValueError("Unexpected pinned GPT-2 tokenizer boundary IDs/vocabulary")
    source = {"repository": OWT_REPOSITORY, "revision": OWT_REVISION, "config": "plain_text",
        "split": "train", "rows": len(dataset), "fingerprint": dataset._fingerprint,
        "loader": "datasets.load_dataset at pinned revision, no remote Python",
        # Keep checkpoint provenance weights_only-loadable: HF metadata can
        # contain dict subclasses such as Features and SplitDict.
        "dataset_info": json.loads(json.dumps(asdict(dataset.info))),
        "historical_alignment": "not independently authenticated by this script; apply published tail rule to pinned source",
        "source_audit": audit,
        "versions": {"python": platform.python_version(),
                     **{name: importlib.metadata.version(name) for name in ("torch", "datasets", "transformers", "tokenizers")}},
        "tokenizer_backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()}
    prepare_bundle(dataset, tokenizer, args.output, quotas=quotas, seed=args.seed,
                   exclusions=excluded, exclusion_sources=exclusion_sources, source=source,
                   evaluation_population=args.evaluation_population)
    print(json.dumps({"event": "prepared", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
