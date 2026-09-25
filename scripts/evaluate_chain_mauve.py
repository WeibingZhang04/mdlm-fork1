#!/usr/bin/env python3
"""Pinned, fixed-length MAUVE evaluation: reference, features, then compare.

The reference is the original MDLM OWT validation tail, not a new test split.
No BOS/EOS tokens are inserted, no EOS truncation or decode/retokenize is used.
Compare identical sample counts and lengths across methods. Fewer than 5000
samples per distribution are explicitly labeled exploratory here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chain_crf.backbone import file_sha256, TOKENIZER_REPOSITORY, TOKENIZER_REVISION
from scripts.prepare_chain_data import prior_document_ids, OWT_REPOSITORY, OWT_REVISION

GPT2_REVISION = "32b71b12589c2f8d625668d2335a01cac3249519"
SOURCE_ROWS = 8013769
HELDOUT_START = SOURCE_ROWS - 100000


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def versions(names):
    return {name: importlib.metadata.version(name) for name in names}


def validate_tokens(tokens, length):
    if len(tokens) != length or any(type(v) is not int or not 0 <= v < 50257 for v in tokens):
        raise ValueError("Every passage must have the requested length and clean GPT-2 token IDs")


def select_references(rows, tokenizer, length, samples, exclusions=()):
    """First N eligible distinct documents, first L tokens; no model-based choice."""
    if not 2 <= length <= 1024 or samples < 1:
        raise ValueError("Require 2 <= length <= 1024 and positive sample count")
    seen = set(exclusions)
    selected = []
    for row in rows:
        index, text = row["source_document_index"], row["text"]
        if not HELDOUT_START <= index < SOURCE_ROWS:
            raise ValueError("Reference document is outside the pinned validation tail")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < length:
            continue
        tokens = tokens[:length]
        validate_tokens(tokens, length)
        selected.append({"sample_id": len(selected), "document_id": digest,
                         "source_document_index": index, "source_token_offset": 0,
                         "prefix_length": 0, "token_ids": tokens})
        if len(selected) == samples:
            return selected
    raise ValueError(f"Only {len(selected)} eligible held-out documents; need {samples}")


def raw_heldout_rows(raw_cache, heldout_cache):
    """Memory-map only the original Arrow shards intersecting the validation tail.

    Verify pinned cache metadata, the existing processed provenance, and one
    raw-document hash/index against that independently recorded processed cache.
    Metadata is not a cryptographic verification of all original corpus bytes.
    """
    from datasets import Dataset, load_from_disk
    from data_provenance import validate_manifest
    raw_cache, heldout_cache = Path(raw_cache), Path(heldout_cache)
    sidecar = heldout_cache.with_suffix(heldout_cache.suffix + ".provenance.json")
    provenance = json.loads(sidecar.read_text())
    spec = provenance["specification"]
    validate_manifest(provenance, expected_specification=spec)
    if (spec.get("dataset_name_or_path") != OWT_REPOSITORY
            or spec.get("source_revision") != OWT_REVISION
            or spec.get("source_window") != [HELDOUT_START, SOURCE_ROWS]
            or spec.get("source_split") != "train"
            or spec.get("source_num_rows") != SOURCE_ROWS
            or spec.get("document_boundary_mode") != "source_document"):
        raise ValueError("Need the pinned original OWT validation-tail provenance")
    processed = load_from_disk(str(heldout_cache), keep_in_memory=False)
    observed = provenance["observed"]
    if (processed._fingerprint != observed["processed_fingerprint"]
            or len(processed) != observed["processed_num_sequences"]):
        raise ValueError("Processed cache does not match its provenance")
    anchor = processed[0]
    info_path = raw_cache / "dataset_info.json"
    info = json.loads(info_path.read_text())
    split = info["splits"]["train"]
    lengths = split["shard_lengths"]
    if split["num_examples"] != SOURCE_ROWS or sum(lengths) != SOURCE_ROWS or len(lengths) != 80:
        raise ValueError("Unexpected raw OWT shard layout")
    downloads = info["download_checksums"]
    expected_prefix = f"hf://datasets/{OWT_REPOSITORY}@{OWT_REVISION}/plain_text/"
    if not downloads or any(not name.startswith(expected_prefix) for name in downloads):
        raise ValueError("Raw cache metadata does not name the pinned source revision")
    shards, offset, anchor_verified = [], 0, False
    for number, count in enumerate(lengths):
        stop = offset + count
        if stop > HELDOUT_START:
            path = raw_cache / f"openwebtext-train-{number:05d}-of-00080.arrow"
            dataset = Dataset.from_file(str(path))
            if len(dataset) != count or dataset.column_names != ["text"]:
                raise ValueError("Raw Arrow shard differs from its recorded layout")
            if offset <= anchor["source_document_index"] < stop:
                text = dataset[anchor["source_document_index"] - offset]["text"]
                if hashlib.sha256(text.encode()).hexdigest() != anchor["source_document_sha256"]:
                    raise ValueError("Raw and processed source document alignment differs")
                anchor_verified = True
            shards.append((offset, dataset))
        offset = stop
    if not anchor_verified:
        raise ValueError("No independent source-document alignment anchor found")

    def rows():
        for offset, dataset in shards:
            for local in range(max(0, HELDOUT_START - offset), len(dataset)):
                yield {"source_document_index": offset + local, "text": dataset[local]["text"]}

    metadata = {"repository": OWT_REPOSITORY, "revision": OWT_REVISION,
                "split": "train", "role": "original_mdlm_validation_tail_not_test",
                "window": [HELDOUT_START, SOURCE_ROWS],
                "official_training_window": [0, HELDOUT_START],
                "dataset_info_sha256": file_sha256(info_path),
                "processed_provenance_sha256": file_sha256(sidecar),
                "processed_manifest_sha256": provenance["manifest_sha256"],
                "alignment_document_index": anchor["source_document_index"],
                "alignment_document_sha256": anchor["source_document_sha256"],
                "verification": "pinned metadata, shard row counts, one cross-cache document anchor"}
    return rows(), metadata


def read_records(path, *, length, samples):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(records) < samples or samples < 1:
        raise ValueError("Input has fewer than the requested positive sample count")
    records = records[:samples]
    ids = [record["sample_id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs")
    for record in records:
        validate_tokens(record["token_ids"], length)
        if record.get("prefix_length", 0) != 0:
            raise ValueError("This protocol is for unconditional generation, not prompted passages")
    return records


@torch.inference_mode()
def terminal_features(model, records, *, device, batch_size):
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    result = []
    model.eval()
    for offset in range(0, len(records), batch_size):
        ids = torch.tensor([r["token_ids"] for r in records[offset:offset + batch_size]],
                           dtype=torch.long, device=device)
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    use_cache=False, return_dict=True)
        # GPT2Model applies its final layer norm in last_hidden_state. Keeping
        # only the terminal vector avoids retaining every layer's activations.
        result.append(out.last_hidden_state[:, -1].float().cpu().numpy())
    features = np.concatenate(result)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite features")
    return features


def feature_protocol(length):
    return {"model": "gpt2-large", "revision": GPT2_REVISION,
            "tokenizer_revision": TOKENIZER_REVISION, "length": length,
            "tokens": "original_gpt2_ids_no_inserted_specials_no_eos_stop_no_retokenization",
            "features": "final_layer_terminal_hidden_state", "dtype": "float32",
            "tf32": False}


def load_features(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / "features.npy"
    if file_sha256(path) != manifest["features_sha256"]:
        raise ValueError("Feature file checksum differs from its manifest")
    features = np.load(path, allow_pickle=False)
    if (features.ndim != 2 or features.shape[0] != manifest["samples"]
            or features.dtype != np.float32 or not np.isfinite(features).all()):
        raise ValueError("Invalid feature array")
    return features, manifest


def compare_features(p_features, q_features, p_manifest, q_manifest, compute):
    if p_manifest["role"] != "reference" or q_manifest["role"] != "generation":
        raise ValueError("Need human reference p and generated q features")
    if p_manifest["protocol"] != q_manifest["protocol"]:
        raise ValueError("Feature protocols differ")
    if p_features.shape != q_features.shape or len(p_features) < 100:
        raise ValueError("Equal feature shapes and at least 100 samples each are required")
    count = len(p_features)
    protocol = dict(num_buckets=max(2, round(count / 10)), pca_max_data=-1,
                    kmeans_explained_var=.9, kmeans_num_redo=5, kmeans_max_iter=500,
                    divergence_curve_discretization_size=25, mauve_scaling_factor=5,
                    seed=25, verbose=False)
    result = compute(p_features=p_features, q_features=q_features, **protocol)
    output = {name: float(getattr(result, name)) for name in
              ("mauve", "mauve_star", "frontier_integral", "frontier_integral_star")}
    output.update({name: np.asarray(getattr(result, name)).tolist() for name in
                   ("p_hist", "q_hist", "divergence_curve")})
    output.update(samples_per_distribution=count, protocol=protocol,
                  interpretation="exploratory_small_sample" if count < 5000 else "fixed_final_comparison")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ref = commands.add_parser("reference")
    ref.add_argument("--raw-cache", type=Path, required=True)
    ref.add_argument("--heldout-cache", type=Path, required=True)
    ref.add_argument("--exclude-documents", type=Path, nargs="+", required=True)
    ref.add_argument("--cache-dir", type=Path)
    feat = commands.add_parser("features")
    feat.add_argument("--input", type=Path, required=True)
    feat.add_argument("--role", choices=["reference", "generation"], required=True)
    feat.add_argument("--cache-dir", type=Path)
    feat.add_argument("--device", default="cuda")
    feat.add_argument("--batch-size", type=int, default=4)
    compare = commands.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--generation", type=Path, required=True)
    compare.add_argument("--threads", type=int, default=4)
    for command in (ref, feat):
        command.add_argument("--length", type=int, choices=[256, 1024], required=True)
        command.add_argument("--samples", type=int, default=5000)
    for command in (ref, feat, compare):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Output exists; use a new destination to preserve the evaluation record")
    if args.command == "reference":
        from transformers import AutoTokenizer
        rows, source = raw_heldout_rows(args.raw_cache, args.heldout_cache)
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPOSITORY, revision=TOKENIZER_REVISION,
                                                  cache_dir=args.cache_dir, local_files_only=True)
        excluded = prior_document_ids(args.exclude_documents)
        if not excluded:
            raise ValueError("Exclusion files must contain recorded training/development document IDs")
        records = select_references(rows, tokenizer, args.length, args.samples, excluded)
        args.output.mkdir(parents=True)
        path = args.output / "references.jsonl"
        with path.open("x") as handle:
            for row in records:
                handle.write(json.dumps(row) + "\n")
        write_json(args.output / "manifest.json", {"role": "reference", "source": source,
                   "selection": "first_N_distinct_eligible_documents_first_L_raw_tokens",
                   "population": "validation_documents_with_at_least_L_raw_GPT2_tokens",
                   "samples": len(records), "length": args.length,
                   "tokenizer_revision": TOKENIZER_REVISION, "records_sha256": file_sha256(path),
                   "excluded_document_count": len(excluded),
                   "exclusion_file_sha256": [file_sha256(p) for p in args.exclude_documents]})
    elif args.command == "features":
        from transformers import AutoModel
        input_digest = file_sha256(args.input)
        records = read_records(args.input, length=args.length, samples=args.samples)
        reference = None
        if args.role == "reference":
            reference = json.loads((args.input.parent / "manifest.json").read_text())
            if (reference.get("role") != "reference" or reference.get("length") != args.length
                    or reference.get("records_sha256") != file_sha256(args.input)):
                raise ValueError("Reference selection manifest does not match the supplied passages")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = AutoModel.from_pretrained("gpt2-large", revision=GPT2_REVISION,
                                         cache_dir=args.cache_dir, local_files_only=True,
                                         torch_dtype=torch.float32).to(args.device).eval()
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        features = terminal_features(model, records, device=args.device, batch_size=args.batch_size)
        elapsed = time.perf_counter() - start
        if file_sha256(args.input) != input_digest:
            raise ValueError("Input changed during feature extraction; use completed generation artifacts")
        args.output.mkdir(parents=True)
        path = args.output / "features.npy"
        with path.open("xb") as handle:
            np.save(handle, features, allow_pickle=False)
        write_json(args.output / "manifest.json", {"role": args.role,
                   "protocol": feature_protocol(args.length), "samples": len(records),
                   "input_sha256": input_digest, "features_sha256": file_sha256(path),
                   "reference_selection": reference, "sample_ids": [r["sample_id"] for r in records],
                   "draw_ids": [r.get("draw_id") for r in records],
                   "batch_size": args.batch_size, "device": args.device,
                   "feature_seconds_excluding_model_load": elapsed,
                   "versions": versions(["torch", "transformers", "numpy"]),
                   "evaluator_sha256": file_sha256(__file__)})
    else:
        import mauve
        import faiss
        from threadpoolctl import threadpool_limits
        if args.threads < 1:
            raise ValueError("Thread count must be positive")
        p, pm = load_features(args.reference)
        q, qm = load_features(args.generation)
        faiss.omp_set_num_threads(args.threads)
        start = time.perf_counter()
        with threadpool_limits(limits=args.threads):
            result = compare_features(p, q, pm, qm, mauve.compute_mauve)
        result.update(cpu_seconds=time.perf_counter() - start, threads=args.threads,
                      versions=versions(["mauve-text", "faiss-cpu", "scikit-learn", "numpy"]),
                      reference_manifest=pm, generation_manifest=qm)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        print(json.dumps({k: result[k] for k in ("mauve", "mauve_star", "samples_per_distribution", "interpretation")}))


if __name__ == "__main__":
    main()
