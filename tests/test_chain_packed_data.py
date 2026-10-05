"""Packed-data separation and provenance checks; no network, model, or GPU."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from chain_crf.data import assert_disjoint, load_token_data
from scripts.prepare_chain_data import prior_document_ids
from scripts.prepare_chain_packed_data import (
    packed_rows, prepare_bundle, select_documents, shuffled_indices, write_split,
)


class Tokenizer:
    bos_token_id = eos_token_id = 99
    vocab_size = 100

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [int(value) for value in text.split()]


def identity(dataset, index, role="train"):
    return {"source_document_index": index, "split": role,
            "document_id": hashlib.sha256(dataset[index]["text"].encode()).hexdigest()}


class PackedDataTest(unittest.TestCase):
    def test_index_order_is_complete_reproducible_and_seeded(self):
        first = list(shuffled_indices(200, "dataset-seed"))
        self.assertEqual(sorted(first), list(range(200)))
        self.assertEqual(first, list(shuffled_indices(200, "dataset-seed")))
        self.assertNotEqual(first, list(shuffled_indices(200, "other-seed")))

    def test_split_before_packing_excludes_tail_duplicates_and_prior_documents(self):
        dataset = [{"text": f"{i} 3 4 5"} for i in range(1, 71)]
        dataset[0] = dict(dataset[-1])  # duplicate in original train and validation
        dataset[1] = dict(dataset[2])   # duplicate within original train
        excluded = {identity(dataset, 6)["document_id"], identity(dataset, 67)["document_id"]}
        quotas = {"train": 30, "dev": 10, "heldout_eval": 5}
        selected, report = select_documents(dataset, quotas=quotas, seed="one",
                                             heldout_documents=10, exclusions=excluded)
        again, again_report = select_documents(dataset, quotas=quotas, seed="one",
                                               heldout_documents=10, exclusions=excluded)
        self.assertEqual((selected, report), (again, again_report))
        docs = [row["document_id"] for rows in selected.values() for row in rows]
        self.assertEqual(len(docs), len(set(docs)))
        self.assertFalse(set(docs) & excluded)
        tail = {identity(dataset, i)["document_id"] for i in range(60, 70)}
        for role in ("train", "dev"):
            self.assertTrue(all(row["source_document_index"] < 60 for row in selected[role]))
            self.assertFalse(tail & {row["document_id"] for row in selected[role]})
        self.assertTrue(all(row["source_document_index"] >= 60 for row in selected["heldout_eval"]))

    def test_increasing_train_quota_preserves_dev_and_training_prefix(self):
        dataset = [{"text": str(i)} for i in range(40)]
        small, _ = select_documents(dataset, quotas={"train": 4, "dev": 3, "heldout_eval": 2},
                                    seed="fixed", heldout_documents=5)
        large, _ = select_documents(dataset, quotas={"train": 12, "dev": 3, "heldout_eval": 2},
                                    seed="fixed", heldout_documents=5)
        self.assertEqual(small["dev"], large["dev"])
        self.assertEqual(small["train"], large["train"][:4])
        self.assertEqual(small["heldout_eval"], large["heldout_eval"])

    def test_full_tail_preserves_order_duplicates_blanks_and_prior_overlap(self):
        dataset = [{"text": str(i)} for i in range(20)]
        dataset[-4:] = [{"text": "2"}, {"text": "2"}, {"text": ""}, {"text": "19"}]
        excluded = {identity(dataset, 19)["document_id"]}
        args = dict(quotas={"train": 5, "dev": 2, "heldout_eval": 4},
                    heldout_documents=4, exclusions=excluded, evaluation_population="full-tail")
        selected, report = select_documents(dataset, seed="first", **args)
        other, _ = select_documents(dataset, seed="other", **args)
        self.assertEqual(selected["heldout_eval"], other["heldout_eval"])
        self.assertEqual([row["source_document_index"] for row in selected["heldout_eval"]],
                         list(range(16, 20)))
        self.assertEqual(report["validation_tail_audit"], {
            "empty_documents": 1, "duplicate_content_documents": 1,
            "prior_experiment_overlap_documents": 1})
        tail_hashes = {row["document_id"] for row in selected["heldout_eval"]}
        for role in ("train", "dev"):
            self.assertFalse(tail_hashes & {row["document_id"] for row in selected[role]})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "full-tail"
            prepare_bundle(dataset, Tokenizer(), output, seed="first", length=6, **args)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertFalse(manifest["exact_document_deduplication"]["heldout_eval"])
            self.assertEqual(manifest["prior_exclusion_roles"], ["train", "dev"])
            self.assertEqual(manifest["splits"]["heldout_eval"]["selected_documents"], 4)
            self.assertEqual(manifest["splits"]["heldout_eval"]["source_eos_tokens"], 4)

    def test_full_tail_requires_full_quota(self):
        dataset = [{"text": str(i)} for i in range(20)]
        with self.assertRaisesRegex(ValueError, "entire validation"):
            select_documents(dataset, quotas={"train": 5, "dev": 2, "heldout_eval": 3},
                             seed="test", heldout_documents=4, evaluation_population="full-tail")

    def test_exact_payload_boundaries_source_spans_and_tail_accounting(self):
        dataset = [{"text": "1 2"}, {"text": "3 4 5 6 7 8"}]
        documents = [identity(dataset, i) for i in range(2)]
        report = {}
        rows = list(packed_rows(dataset, documents, Tokenizer(), length=6, report=report))
        self.assertEqual([row["input_ids"] for row in rows],
                         [[99, 1, 2, 99, 3, 99], [99, 4, 5, 6, 7, 99]])
        self.assertEqual(rows[0]["document_ids"], [doc["document_id"] for doc in documents])
        self.assertEqual(rows[1]["document_ids"], [documents[1]["document_id"]])
        self.assertEqual(report["dropped_tail_tokens"], 2)
        self.assertEqual(report["packed_text_tokens"], 7)
        self.assertEqual(report["packed_source_eos_tokens"], 1)
        for row in rows:
            reconstructed = []
            for span in row["source_spans"]:
                source = Tokenizer().encode(dataset[span["source_document_index"]]["text"]) + [99]
                piece = source[span["source_token_start"]:span["source_token_stop"]]
                self.assertEqual(row["input_ids"][span["row_token_start"]:span["row_token_stop"]], piece)
                reconstructed.extend(piece)
            self.assertEqual(reconstructed, row["input_ids"][1:-1])
        self.assertEqual(report["source_text_tokens"] + report["source_eos_tokens"],
                         report["rows"] * 4 + report["dropped_tail_tokens"])

    def test_packing_rejects_source_mutation(self):
        dataset = [{"text": "1 2 3 4"}]
        documents = [identity(dataset, 0)]
        dataset[0]["text"] = "5 6 7 8"
        with self.assertRaisesRegex(ValueError, "changed"):
            list(packed_rows(dataset, documents, Tokenizer(), length=6, report={}))

    def test_loader_checks_every_document_of_a_packed_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packed.jsonl"
            path.write_text(json.dumps({"input_ids": [1, 2, 3], "document_ids": ["a", "b"]}) + "\n")
            tokens, docs, _ = load_token_data(path, length=3, vocab_size=10, mask_id=9)
            with self.assertRaisesRegex(ValueError, "document overlap"):
                assert_disjoint(tokens, docs, torch.tensor([[4, 5, 6]]), ["b"])
            assert_disjoint(tokens, docs, torch.tensor([[4, 5, 6]]), ["c"])
            path.write_text(json.dumps({"input_ids": [1, 2, 3], "document_ids": []}) + "\n")
            with self.assertRaisesRegex(ValueError, "document identity"):
                load_token_data(path, length=3, vocab_size=10, mask_id=9)

    def test_bundle_roundtrip_hashes_exclusions_and_no_overwrite(self):
        dataset = [{"text": " ".join([str(i)] * 5)} for i in range(1, 41)]
        args = dict(quotas={"train": 8, "dev": 3, "heldout_eval": 2}, seed="test",
                    heldout_documents=5, length=6)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "first"
            selected, _ = prepare_bundle(dataset, Tokenizer(), output, **args)
            other = Path(directory) / "second"
            prepare_bundle(dataset, Tokenizer(), other, **args)
            manifest = json.loads((output / "manifest.json").read_text())
            second = json.loads((other / "manifest.json").read_text())
            for role in ("train", "dev", "heldout_eval"):
                pt, docs, source = load_token_data(output / f"{role}.pt", length=6, vocab_size=100, mask_id=98)
                jl, json_docs, _ = load_token_data(output / f"{role}.jsonl", length=6, vocab_size=100, mask_id=98)
                torch.testing.assert_close(pt, jl)
                self.assertEqual(docs, json_docs)
                self.assertEqual(source["split"], role)
                self.assertEqual(manifest["splits"][role], second["splits"][role])
                for name in (f"{role}.pt", f"{role}.jsonl"):
                    self.assertEqual(manifest["files"][name], hashlib.sha256((output / name).read_bytes()).hexdigest())
                self.assertEqual(prior_document_ids([output / f"{role}.jsonl"]),
                                 {doc for group in docs for doc in group})
            self.assertEqual(prior_document_ids([output / "selected_documents.json"]),
                             {doc["document_id"] for rows in selected.values() for doc in rows})
            with self.assertRaises(FileExistsError):
                prepare_bundle(dataset, Tokenizer(), output, **args)

    def test_identical_token_rows_in_different_roles_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [{"input_ids": [99, 1, 2, 99], "document_ids": ["a"]}]
            roles = {}
            write_split(Path(directory), "dev", iter(rows), length=4, provenance={}, row_roles=roles)
            rows[0]["document_ids"] = ["different-document"]
            with self.assertRaisesRegex(ValueError, "Identical packed token row"):
                write_split(Path(directory), "train", iter(rows), length=4, provenance={}, row_roles=roles)

    def test_count_builder_rejects_reserved_evaluation_role(self):
        from scripts.build_chain_counts import main
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "heldout.pt", Path(directory) / "counts.pt"
            torch.save({"tokens": torch.tensor([[1, 2, 3]]),
                        "provenance": {"split": "heldout_eval"}}, source)
            with patch("sys.argv", ["build_chain_counts", "--data", str(source), "--output", str(output)]):
                with self.assertRaisesRegex(ValueError, "training data"):
                    main()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
