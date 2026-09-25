import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.evaluate_chain_mauve import (
    GPT2_BOUNDARY_ID, HELDOUT_START, OWT_REPOSITORY, OWT_REVISION, SOURCE_ROWS,
    TOKENIZER_REVISION, WRAPPED_REFERENCE_FORMAT, compare_features,
    comparison_input_protocol, feature_protocol, file_sha256, load_features,
    main, read_records, select_references, terminal_features, validate_tokens,
    validate_wrapped_reference, wrap_reference,
)


class Tokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [int(v) for v in text.split()]


def test_reference_first_raw_tokens_distinct_docs_exclusions_and_short_documents():
    texts = ["1 2 3 4", "1 2 3 4", "6 7 8 9", "10", "11 12 13 14"]
    rows = [{"text": text, "source_document_index": HELDOUT_START + i} for i, text in enumerate(texts)]
    excluded = [hashlib.sha256(texts[2].encode()).hexdigest()]
    selected = select_references(rows, Tokenizer(), 3, 2, excluded)
    assert [r["token_ids"] for r in selected] == [[1, 2, 3], [11, 12, 13]]
    assert [r["source_document_index"] for r in selected] == [HELDOUT_START, HELDOUT_START + 4]
    assert all(r["source_token_offset"] == 0 for r in selected)


def test_reference_rejects_training_window_and_insufficient_holdout():
    with pytest.raises(ValueError, match="outside"):
        select_references([{"text": "1 2", "source_document_index": 1}], Tokenizer(), 2, 1)
    with pytest.raises(ValueError, match="Only 0"):
        select_references([], Tokenizer(), 2, 1)


@pytest.mark.parametrize("tokens", [[50257, 1], [True, 1], [-1, 1], [1]])
def test_invalid_tokens_rejected(tokens):
    with pytest.raises(ValueError):
        validate_tokens(tokens, 2)


def test_generated_records_preserve_eos_interior_and_reject_prefix(tmp_path):
    path = tmp_path / "samples.jsonl"
    row = {"sample_id": 5000, "token_ids": [1, 50256, 3], "prefix_length": 0}
    path.write_text(json.dumps(row) + "\n")
    assert read_records(path, length=3, samples=1)[0]["token_ids"] == [1, 50256, 3]
    row["prefix_length"] = 1
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="unconditional"):
        read_records(path, length=3, samples=1)


def test_terminal_features_use_last_position_final_output_no_padding():
    class Model(torch.nn.Module):
        def forward(self, input_ids, attention_mask, use_cache, return_dict):
            assert not self.training and use_cache is False and return_dict is True
            assert attention_mask.eq(1).all()
            return SimpleNamespace(last_hidden_state=torch.stack((input_ids, input_ids + 10), -1).float())
    rows = [{"token_ids": [1, 2, 3]}, {"token_ids": [5, 50256, 7]}, {"token_ids": [9, 10, 11]}]
    result = terminal_features(Model(), rows, device="cpu", batch_size=2)
    np.testing.assert_array_equal(result, [[3, 13], [7, 17], [11, 21]])
    assert result.dtype == np.float32


def manifests():
    return ({"role": "reference", "protocol": feature_protocol(256)},
            {"role": "generation", "protocol": feature_protocol(256)})


def test_comparison_pins_clustering_and_labels_small_sample():
    p, q = np.zeros((100, 3), np.float32), np.ones((100, 3), np.float32)
    def compute(**kwargs):
        assert kwargs["p_features"] is p and kwargs["q_features"] is q
        assert kwargs["num_buckets"] == 10 and kwargs["seed"] == 25
        assert kwargs["kmeans_num_redo"] == 5 and kwargs["kmeans_max_iter"] == 500
        return SimpleNamespace(mauve=.5, mauve_star=.6, frontier_integral=.2,
                               frontier_integral_star=.3, p_hist=np.ones(10)/10,
                               q_hist=np.ones(10)/10, divergence_curve=np.zeros((25, 2)))
    result = compare_features(p, q, *manifests(), compute)
    assert result["mauve"] == .5 and result["interpretation"] == "exploratory_small_sample"


def test_mismatched_feature_protocol_or_counts_fail_before_clustering():
    p, q = np.zeros((100, 3), np.float32), np.zeros((101, 3), np.float32)
    with pytest.raises(ValueError, match="Equal"):
        compare_features(p, q, *manifests(), None)
    pm, qm = manifests()
    qm["protocol"]["length"] = 1024
    with pytest.raises(ValueError, match="protocols"):
        compare_features(p, p, pm, qm, None)


def test_feature_checksum_prevents_stale_reuse(tmp_path):
    path = tmp_path / "features.npy"
    np.save(path, np.zeros((100, 3), np.float32))
    (tmp_path / "manifest.json").write_text(json.dumps({"features_sha256": "incorrect", "samples": 100}))
    with pytest.raises(ValueError, match="checksum"):
        load_features(tmp_path)


def raw_reference_fixture(tmp_path, length=6, samples=3):
    directory = tmp_path / "raw"
    directory.mkdir()
    records = [{"sample_id": i, "document_id": hashlib.sha256(str(i).encode()).hexdigest(),
                "source_document_index": HELDOUT_START + i, "source_token_offset": 0,
                "prefix_length": 0, "token_ids": [i + 1, 50256] + list(range(20, length + 18))}
               for i in range(samples)]
    path = directory / "references.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    manifest = {"role": "reference", "length": length, "samples": samples,
                "selection": "first_N_distinct_eligible_documents_first_L_raw_tokens",
                "population": "validation_documents_with_at_least_L_raw_GPT2_tokens",
                "tokenizer_revision": TOKENIZER_REVISION,
                "records_sha256": file_sha256(path),
                "excluded_document_count": 7, "exclusion_file_sha256": ["a" * 64],
                "source": {"repository": OWT_REPOSITORY, "revision": OWT_REVISION,
                           "window": [HELDOUT_START, SOURCE_ROWS],
                           "role": "original_mdlm_validation_tail_not_test"}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return path, manifest, records


def test_wrap_reference_preserves_documents_order_payload_interior_eos_and_original_bytes(tmp_path):
    path, parent, records = raw_reference_fixture(tmp_path)
    before = path.read_bytes(), (path.parent / "manifest.json").read_bytes()
    destination = tmp_path / "wrapped"
    result = wrap_reference(path, destination)
    output = read_records(destination / "references.jsonl", length=6, samples=3)
    for original, wrapped in zip(records, output):
        assert wrapped["token_ids"] == [GPT2_BOUNDARY_ID] + original["token_ids"][:4] + [GPT2_BOUNDARY_ID]
        assert wrapped["token_ids"][2] == 50256  # Interior specials are not deleted.
        for key in ("sample_id", "document_id", "source_document_index", "source_token_offset"):
            assert wrapped[key] == original[key]
        assert wrapped["raw_parent_token_ids_sha256"] == hashlib.sha256(
            json.dumps(original["token_ids"], separators=(",", ":")).encode()).hexdigest()
    assert result["reference_format"] == WRAPPED_REFERENCE_FORMAT
    assert result["parent_reference"]["manifest"] == parent
    assert result["parent_reference"]["records_sha256"] == file_sha256(path)
    assert result["records_sha256"] == file_sha256(destination / "references.jsonl")
    assert result["records_sha256"] != result["parent_reference"]["records_sha256"]
    assert result["excluded_document_count"] == parent["excluded_document_count"]
    assert before == (path.read_bytes(), (path.parent / "manifest.json").read_bytes())
    validate_wrapped_reference(result)


def test_wrap_reference_rejects_changed_parent_existing_output_and_double_wrap(tmp_path):
    path, _, _ = raw_reference_fixture(tmp_path)
    result = tmp_path / "wrapped"
    wrap_reference(path, result)
    with pytest.raises(FileExistsError):
        wrap_reference(path, result)
    with pytest.raises(ValueError, match="original raw reference"):
        wrap_reference(result / "references.jsonl", tmp_path / "double")
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="authenticated"):
        wrap_reference(path, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()


@pytest.mark.parametrize("mutation", ["count", "order", "documents", "window", "tokenizer"])
def test_wrap_reference_rejects_selection_and_source_drift(tmp_path, mutation):
    path, manifest, records = raw_reference_fixture(tmp_path)
    if mutation == "count":
        records.pop()
    elif mutation == "order":
        records.reverse()
    elif mutation == "documents":
        records[1]["document_id"] = records[0]["document_id"]
    elif mutation == "window":
        manifest["source"]["window"] = [0, 100]
    else:
        manifest["tokenizer_revision"] = "unversioned"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    manifest["records_sha256"] = file_sha256(path)
    (path.parent / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        wrap_reference(path, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def wrapped_feature_manifests(tmp_path):
    path, _, _ = raw_reference_fixture(tmp_path)
    reference = wrap_reference(path, tmp_path / "wrapped")
    p = {"role": "reference", "protocol": feature_protocol(6, version=2),
         "reference_selection": reference, "samples": 3,
         "input_sha256": reference["records_sha256"]}
    q = {"role": "generation", "protocol": feature_protocol(6)}
    return p, q


def test_wrapped_protocol_accepts_unchanged_legacy_generation_extractor_only(tmp_path):
    p, q = wrapped_feature_manifests(tmp_path)
    before = json.dumps(q, sort_keys=True)
    result = comparison_input_protocol(p, q)
    assert result["reference_format"] == WRAPPED_REFERENCE_FORMAT
    assert result["generation_format"] == "unchanged_generated_token_ids"
    assert json.dumps(q, sort_keys=True) == before
    assert "schema_version" not in feature_protocol(6)  # Existing manifests unchanged.
    assert "no_inserted_specials" not in p["protocol"]["tokens"]
    assert comparison_input_protocol(*manifests()) is None


@pytest.mark.parametrize("mutation", ["length", "dtype", "model", "revision", "tokens", "tf32",
                                      "parent_hash", "source_hash", "transform"])
def test_wrapped_protocol_rejects_feature_and_parent_drift(tmp_path, mutation):
    p, q = wrapped_feature_manifests(tmp_path)
    if mutation == "parent_hash":
        p["reference_selection"]["parent_reference"]["manifest_sha256"] = "bad"
    elif mutation == "source_hash":
        p["input_sha256"] = "bad"
    elif mutation == "transform":
        p["reference_selection"]["transformation"]["eos_id"] = 1
    else:
        q["protocol"][mutation] = "different"
    with pytest.raises(ValueError):
        comparison_input_protocol(p, q)


def test_wrap_reference_cli_outputs_new_explicit_manifest(tmp_path, capsys):
    path, _, _ = raw_reference_fixture(tmp_path)
    output = tmp_path / "cli"
    main(["wrap-reference", "--input", str(path), "--output", str(output)])
    summary = json.loads(capsys.readouterr().out)
    assert summary["reference_format"] == WRAPPED_REFERENCE_FORMAT
    assert summary["samples"] == 3 and summary["length"] == 6
    assert summary["records_sha256"] == file_sha256(output / "references.jsonl")


def test_wrapped_feature_cli_reuses_legacy_generation_features_without_relabeling(tmp_path, monkeypatch):
    class Model(torch.nn.Module):
        def forward(self, input_ids, attention_mask, use_cache, return_dict):
            assert attention_mask.eq(1).all()
            return SimpleNamespace(last_hidden_state=torch.stack((input_ids, input_ids + 1), -1).float())

    calls = []
    def load(model_name, **kwargs):
        calls.append((model_name, kwargs))
        assert kwargs["torch_dtype"] == torch.float32
        return Model()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModel=SimpleNamespace(from_pretrained=load)))
    path, _, _ = raw_reference_fixture(tmp_path, length=256, samples=100)
    wrapped_dir = tmp_path / "wrapped"
    reference = wrap_reference(path, wrapped_dir)
    for role, input_path, output in (
            ("reference", wrapped_dir / "references.jsonl", tmp_path / "ref_features"),
            ("generation", path, tmp_path / "gen_features")):
        main(["features", "--input", str(input_path), "--role", role,
              "--length", "256", "--samples", "100", "--device", "cpu",
              "--batch-size", "16", "--output", str(output)])
    p, pm = load_features(tmp_path / "ref_features")
    q, qm = load_features(tmp_path / "gen_features")
    assert pm["protocol"] == feature_protocol(256, version=2)
    assert qm["protocol"] == feature_protocol(256)
    assert pm["reference_selection"] == reference
    assert pm["input_sha256"] == reference["records_sha256"]
    assert np.all(p[:, 0] == GPT2_BOUNDARY_ID)
    assert np.all(q[:, 0] == 273)  # Original raw final token, never wrapped.
    def compute(**kwargs):
        assert kwargs["num_buckets"] == 10
        return SimpleNamespace(mauve=.5, mauve_star=.6, frontier_integral=.2,
                               frontier_integral_star=.3, p_hist=np.ones(10)/10,
                               q_hist=np.ones(10)/10, divergence_curve=np.zeros((25, 2)))
    result = compare_features(p, q, pm, qm, compute)
    assert result["comparison_input_protocol"]["reference_format"] == WRAPPED_REFERENCE_FORMAT
    assert result["comparison_input_protocol"]["parent_raw_reference_sha256"] == file_sha256(path)
    assert len(calls) == 2


def test_wrapped_feature_cli_rejects_subset_before_loading_model(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Model must not load for invalid reference provenance")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModel=SimpleNamespace(from_pretrained=forbidden)))
    path, _, _ = raw_reference_fixture(tmp_path, length=256)
    wrapped = tmp_path / "wrapped"
    wrap_reference(path, wrapped)
    with pytest.raises(ValueError, match="all selected"):
        main(["features", "--input", str(wrapped / "references.jsonl"), "--role", "reference",
              "--length", "256", "--samples", "2", "--device", "cpu",
              "--output", str(tmp_path / "invalid")])
