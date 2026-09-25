#!/usr/bin/env python3
"""Prepare untouched WikiText-103 validation/test articles for frozen transfer.

Inputs are the original, SHA256-pinned public parquet files, never old model
outputs or wrapped token caches. No training split, model, or GPU is loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from chain_crf.backbone import file_sha256, load_tokenizer, TOKENIZER_REPOSITORY, TOKENIZER_REVISION
from chain_crf.data import atomic_json, atomic_torch_save, canonical_hash

DATASET_REPOSITORY = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_CONFIG = "wikitext-103-raw-v1"
SOURCE_FILES = {
    "validation": {"filename": "validation-00000-of-00001.parquet", "rows": 3760,
                   "bytes": 657209, "articles": 60,
                   "sha256": "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c"},
    "test": {"filename": "test-00000-of-00001.parquet", "rows": 4358,
             "bytes": 732610, "articles": 60,
             "sha256": "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"},
}
# The pinned raw file splits two equations in the "Constant k filter" article
# into lines that happen to resemble titles. Their neighboring rows continue
# the same sentences. These are exact source-row assertions, not heuristics.
AUDITED_BODY_ROWS = {
    "validation": {},
    "test": {2148: " = 1 rad / s and a nominal impedance k = \n",
             2150: " = 1 henry and capacitance C = \n"},
}
_HEADING = re.compile(
    r"(?P<left>(?:=[ \t]*)+)(?P<title>[^=\s](?:.*?[^=\s])?)(?P<right>(?:[ \t]*=)+)")


def heading(text):
    """Recognize balanced raw '= = Section = =' and '= Article =' markers."""
    stripped = text.strip()
    if not stripped.startswith("=") or not stripped.endswith("="):
        return None
    match = _HEADING.fullmatch(stripped)
    if match is None or match["left"].count("=") != match["right"].count("="):
        raise ValueError(f"Malformed or unbalanced WikiText heading: {stripped!r}")
    return match["left"].count("="), match["title"]


def reconstruct_articles(texts, *, audited_body_rows=None):
    """Concatenate exact source strings, without adding/removing whitespace.

    Only a balanced, level-one heading starts an article. Empty preamble rows
    contain no text and are audited; nonempty preamble content is an error.
    Source indices are zero-based and stop indices are exclusive.
    """
    audited_body_rows = {} if audited_body_rows is None else audited_body_rows
    for index, expected in audited_body_rows.items():
        if not 0 <= index < len(texts) or texts[index] != expected:
            raise ValueError("Audited equation-row exception differs from pinned source text")
    articles, current, start, title = [], [], None, None
    levels, empty_preamble = Counter(), 0
    equality_body_rows = []

    def finish(stop):
        if start is None:
            return
        text = "".join(current)
        articles.append({"text": text, "title": title,
                         "document_id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                         "source_start_row": start, "source_stop_row": stop})

    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError("Every raw source row must be text")
        marker = None if index in audited_body_rows else heading(text)
        if text.strip().startswith("=") and marker is None:
            equality_body_rows.append(index)
        if marker:
            level, heading_title = marker
            levels[level] += 1
            if level == 1:
                finish(index)
                current, start, title = [], index, heading_title
        if start is None:
            if text != "":
                raise ValueError(f"Orphan text before first article heading at row {index}")
            empty_preamble += 1
        else:
            current.append(text)
    finish(len(texts))
    if not articles:
        raise ValueError("No WikiText articles found")
    if len({a["document_id"] for a in articles}) != len(articles):
        raise ValueError("Duplicate complete articles in source split")
    audit = {"source_rows": len(texts), "articles": len(articles),
             "empty_preamble_rows": empty_preamble,
             "equals_prefixed_body_rows": equality_body_rows,
             "audited_equation_row_exceptions": [
                 {"row": row, "text": text, "preceding_source_row": texts[row - 1],
                  "following_source_row": texts[row + 1],
                  "reason": "equation continuation inside Constant k filter; not an article title"}
                 for row, text in sorted(audited_body_rows.items())],
             "heading_level_counts": {str(k): v for k, v in sorted(levels.items())},
             "raw_text_sha256": hashlib.sha256("".join(texts).encode("utf-8")).hexdigest(),
             "article_texts_reconstruct_all_source_text":
                 "".join(a["text"] for a in articles) == "".join(texts)}
    if not audit["article_texts_reconstruct_all_source_text"]:
        raise ValueError("Article reconstruction lost or changed raw source text")
    return articles, audit


def tokenize_articles(articles, tokenizer):
    result = []
    for article in articles:
        ids = tokenizer.encode(article["text"], add_special_tokens=False)
        if not ids or any(type(i) is not int or not 0 <= i < tokenizer.vocab_size for i in ids):
            raise ValueError("Tokenizer returned empty or invalid token IDs")
        result.append({**article, "input_ids": ids})
    return result


def chunk_articles(articles, *, length, prefix_length, split):
    if length < 2 or not 0 < prefix_length < length:
        raise ValueError("Require 0 < prefix_length < chunk length")
    if split not in SOURCE_FILES:
        raise ValueError("Only official validation and test splits are supported")
    chunks, dropped, short = [], 0, 0
    for article in articles:
        ids = article["input_ids"]
        dropped += len(ids) % length
        short += int(len(ids) < length)
        for offset in range(0, len(ids) - length + 1, length):
            chunk = ids[offset:offset + length]
            chunks.append({"input_ids": chunk, "document_id": article["document_id"],
                           "source_split": split, "source_start_row": article["source_start_row"],
                           "source_stop_row": article["source_stop_row"],
                           "token_start": offset, "token_stop": offset + length,
                           "chunk_id": canonical_hash({"split": split,
                               "document_id": article["document_id"], "offset": offset, "length": length}),
                           "prefix_length": prefix_length})
    if not chunks:
        raise ValueError(f"No complete {length}-token chunks in {split}")
    return chunks, {"chunks": len(chunks), "kept_tokens": len(chunks) * length,
                    "dropped_tail_tokens": dropped, "articles_shorter_than_chunk": short,
                    "source_tokens": sum(len(a["input_ids"]) for a in articles),
                    "articles_represented": len({c["document_id"] for c in chunks})}


def verify_source(path, split):
    if split not in SOURCE_FILES:
        raise ValueError("Only official validation and test splits are supported")
    expected = SOURCE_FILES[split]
    path = Path(path)
    if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
        raise ValueError(f"Original {split} parquet bytes do not match the pinned public source")


def load_source(path, split):
    verify_source(path, split)
    import pyarrow.parquet as parquet
    table = parquet.read_table(path)
    if table.column_names != ["text"] or len(table) != SOURCE_FILES[split]["rows"]:
        raise ValueError("Pinned raw source schema or row count differs")
    texts = table["text"].to_pylist()
    articles, audit = reconstruct_articles(texts, audited_body_rows=AUDITED_BODY_ROWS[split])
    if len(articles) != SOURCE_FILES[split]["articles"]:
        raise ValueError("Article count differs from the audited pinned source")
    return articles, audit


def source_url(split):
    return (f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/resolve/"
            f"{DATASET_REVISION}/{DATASET_CONFIG}/{SOURCE_FILES[split]['filename']}")


def resolve_source(explicit, split, cache_dir):
    if explicit:
        return Path(explicit)
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(DATASET_REPOSITORY,
        f"{DATASET_CONFIG}/{SOURCE_FILES[split]['filename']}", repo_type="dataset",
        revision=DATASET_REVISION, cache_dir=cache_dir))


def atomic_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_bundle(output, tokenized, audits, *, lengths, prefix_length, provenance):
    """Emit harness-compatible clean tokens plus exact-token continuation pairs."""
    output = Path(output)
    if set(tokenized) != {"validation", "test"}:
        raise ValueError("Prepare validation and test separately in the same audited bundle")
    if set(a["document_id"] for a in tokenized["validation"]) & set(
            a["document_id"] for a in tokenized["test"]):
        raise ValueError("Official validation and test have duplicate complete articles")
    prepared = {split: {length: chunk_articles(articles, length=length,
                        prefix_length=prefix_length, split=split) for length in lengths}
                for split, articles in tokenized.items()}
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty; never overwrite a transfer bundle")
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"dataset_repository": DATASET_REPOSITORY, "dataset_revision": DATASET_REVISION,
                "dataset_config": DATASET_CONFIG,
                "split_roles": {"validation": "selection_only", "test": "final_evaluation_only"},
                "no_training_data": True, "intended_model_use": "frozen_zero_shot_transfer",
                "tokenizer_repository": TOKENIZER_REPOSITORY, "tokenizer_revision": TOKENIZER_REVISION,
                "text_transform": "identity; concatenate exact source strings within each article",
                "detokenizer": None, "add_special_tokens": False, "bos_added": False, "eos_added": False,
                "article_boundary": "balanced single-level equals heading, except two audited equation rows in pinned test; keep all section headings",
                "chunk_policy": "nonoverlapping source-order chunks within each article; drop incomplete tails",
                "lengths": list(lengths), "prefix_length": prefix_length,
                "continuation_policy": "exact token prefix plus reference suffix; never decode and re-encode",
                "source": provenance}
    report = {"schema_version": 1, "protocol": protocol,
              "protocol_sha256": canonical_hash(protocol), "boundary_audit": audits,
              "counts": {}, "files": {}}
    for split, articles in tokenized.items():
        records = [{k: v for k, v in a.items() if k not in {"text", "input_ids"}}
                   | {"token_count": len(a["input_ids"])} for a in articles]
        atomic_json(records, output / split / "articles.json")
        report["counts"][split] = {}
        for length, (chunks, stats) in prepared[split].items():
            directory = output / split / f"length-{length}"
            payload = {"tokens": torch.tensor([c["input_ids"] for c in chunks], dtype=torch.long),
                       "document_ids": [c["document_id"] for c in chunks],
                       "provenance": {"protocol_sha256": report["protocol_sha256"],
                                      "source_split": split, "length": length, "protocol": protocol}}
            atomic_torch_save(payload, directory / "tokens.pt")
            atomic_jsonl(chunks, directory / "tokens.jsonl")
            continuations = [{**{k: v for k, v in c.items() if k != "input_ids"},
                              "prefix_input_ids": c["input_ids"][:prefix_length],
                              "reference_continuation_ids": c["input_ids"][prefix_length:]}
                             for c in chunks]
            atomic_jsonl(continuations, directory / "continuation.jsonl")
            report["counts"][split][str(length)] = stats
    for path in sorted(output.rglob("*")):
        if path.is_file():
            report["files"][str(path.relative_to(output))] = {
                "sha256": file_sha256(path), "bytes": path.stat().st_size}
    atomic_json(report, output / "manifest.json")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-parquet", type=Path)
    parser.add_argument("--test-parquet", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[256, 1024])
    parser.add_argument("--prefix-length", type=int, default=64)
    parser.add_argument("--threads", type=int, choices=[1, 2], default=2)
    args = parser.parse_args(argv)
    if len(set(args.lengths)) != len(args.lengths) or any(
            not 0 < args.prefix_length < length for length in args.lengths):
        parser.error("Unique lengths greater than positive prefix length are required")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be a new or empty directory")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    import pyarrow
    pyarrow.set_cpu_count(args.threads)
    pyarrow.set_io_thread_count(args.threads)
    tokenized, audits, sources = {}, {}, {}
    tokenizer = load_tokenizer(args.cache_dir)
    if tokenizer.vocab_size != 50257:
        raise ValueError("Expected pinned GPT2 vocabulary")
    for split in SOURCE_FILES:
        path = resolve_source(getattr(args, f"{split}_parquet"), split, args.cache_dir)
        articles, audits[split] = load_source(path, split)
        tokenized[split] = tokenize_articles(articles, tokenizer)
        sources[split] = {"url": source_url(split), **SOURCE_FILES[split],
                          "observed_sha256": file_sha256(path)}
    provenance = {"files": sources, "preparation_script_sha256": file_sha256(__file__),
                  "tokenizer_implementation": type(tokenizer).__name__,
                  "tokenizer_backend_sha256": hashlib.sha256(
                      tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                  "versions": {name: importlib.metadata.version(name)
                               for name in ("torch", "transformers", "tokenizers", "pyarrow")}}
    report = write_bundle(args.output, tokenized, audits, lengths=args.lengths,
                         prefix_length=args.prefix_length, provenance=provenance)
    print(json.dumps({"output": str(args.output), "counts": report["counts"],
                      "boundary_audit": audits, "protocol_sha256": report["protocol_sha256"]}, indent=2))
    return report


if __name__ == "__main__":
    main()
