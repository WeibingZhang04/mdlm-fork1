import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.evaluate_chain_mauve import (
    HELDOUT_START, compare_features, feature_protocol, load_features,
    read_records, select_references, terminal_features, validate_tokens,
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
