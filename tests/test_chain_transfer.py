"""Offline WikiText boundary, chunk, split and provenance fixtures."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from chain_crf.data import load_token_data
from scripts.prepare_chain_transfer import (
    chunk_articles, heading, reconstruct_articles, tokenize_articles,
    verify_source, write_bundle,
)


class CharacterTokenizer:
    vocab_size = 256

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode())


class ChainTransferTest(unittest.TestCase):
    def fixture(self, title="A"):
        raw = ["", f" = {title} = \n", "", "first paragraph\n", " = = Section = = \n",
               "second paragraph\n", " = = = Detail = = = \n", "last\n", "", " = B = \n", "other\n"]
        articles, audit = reconstruct_articles(raw)
        return raw, tokenize_articles(articles, CharacterTokenizer()), audit

    def test_balanced_heading_levels_and_internal_equals(self):
        self.assertEqual(heading(" = A = \n"), (1, "A"))
        self.assertEqual(heading(" = = Section = = \n"), (2, "Section"))
        self.assertEqual(heading("== Section =="), (2, "Section"))
        self.assertEqual(heading(" = A = B = \n"), (1, "A = B"))
        self.assertIsNone(heading("paragraph with = content"))
        self.assertIsNone(heading(" = Qualified for the next round \n"))
        for malformed in (" = A = = \n", "= = Section =", "==="):
            with self.assertRaisesRegex(ValueError, "Malformed|unbalanced"):
                heading(malformed)

    def test_article_boundaries_preserve_all_original_bytes(self):
        raw, articles, audit = self.fixture()
        self.assertEqual(len(articles), 2)
        self.assertEqual("".join(a["text"] for a in articles), "".join(raw))
        self.assertEqual(articles[0]["source_start_row"], 1)
        self.assertEqual(articles[0]["source_stop_row"], 9)
        self.assertEqual(articles[1]["source_start_row"], 9)
        self.assertEqual(articles[0]["document_id"], hashlib.sha256("".join(raw[1:9]).encode()).hexdigest())
        self.assertEqual(audit["heading_level_counts"], {"1": 2, "2": 1, "3": 1})
        self.assertEqual(audit["empty_preamble_rows"], 1)

    def test_refuses_orphan_text_whitespace_and_missing_titles(self):
        for prefix in ("orphan\n", " \n", " = = Section = = \n"):
            with self.assertRaisesRegex(ValueError, "Orphan"):
                reconstruct_articles([prefix, " = A = \n", "body\n"])
        with self.assertRaisesRegex(ValueError, "No WikiText"):
            reconstruct_articles(["", ""])

    def test_audited_equations_and_unclosed_legend_lines_remain_body(self):
        raw = ["", " = Filter = \n", "", "nominal impedance k\n",
               " = 1 rad / s and a nominal impedance k = \n", "1 ohm\n",
               " = Qualified for the next round \n"]
        articles, audit = reconstruct_articles(raw, audited_body_rows={4: raw[4]})
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["text"], "".join(raw))
        self.assertEqual(audit["equals_prefixed_body_rows"], [4, 6])
        self.assertEqual(audit["audited_equation_row_exceptions"][0]["row"], 4)
        with self.assertRaisesRegex(ValueError, "differs from pinned source"):
            reconstruct_articles(raw, audited_body_rows={4: "wrong line"})

    def test_chunks_never_cross_articles_or_add_special_tokens(self):
        _, articles, _ = self.fixture()
        chunks, stats = chunk_articles(articles, length=16, prefix_length=4, split="validation")
        by_id = {a["document_id"]: a for a in articles}
        for chunk in chunks:
            source = by_id[chunk["document_id"]]["input_ids"]
            self.assertEqual(chunk["input_ids"], source[chunk["token_start"]:chunk["token_stop"]])
            self.assertEqual(chunk["token_start"] % 16, 0)
            self.assertEqual(chunk["source_split"], "validation")
        self.assertEqual(stats["source_tokens"], stats["kept_tokens"] + stats["dropped_tail_tokens"])
        self.assertEqual(stats["articles_shorter_than_chunk"], 1)

    def test_refuses_training_source_and_modified_parquet(self):
        _, articles, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "Only official"):
            chunk_articles(articles, length=16, prefix_length=4, split="train")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fake.parquet"
            path.write_bytes(b"not the pinned public data")
            with self.assertRaisesRegex(ValueError, "pinned public source"):
                verify_source(path, "validation")

    def test_bundle_is_harness_compatible_and_keeps_exact_prefix_ids(self):
        raw, articles, audit = self.fixture()
        other_raw = [" = C = \n", "test article content that is not validation\n"]
        other, other_audit = reconstruct_articles(other_raw)
        other = tokenize_articles(other, CharacterTokenizer())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new"
            report = write_bundle(output, {"validation": articles, "test": other},
                {"validation": audit, "test": other_audit}, lengths=[16, 32],
                prefix_length=4, provenance={"fixture": True})
            tokens, docs, provenance = load_token_data(output / "validation/length-16/tokens.pt",
                length=16, vocab_size=257, mask_id=256)
            self.assertEqual(tokens.dtype, torch.long)
            self.assertEqual(provenance["source_split"], "validation")
            self.assertFalse(provenance["protocol"]["bos_added"])
            self.assertFalse(provenance["protocol"]["eos_added"])
            rows = [json.loads(line) for line in (output / "validation/length-16/continuation.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["prefix_input_ids"] + rows[0]["reference_continuation_ids"], tokens[0].tolist())
            self.assertEqual(rows[0]["document_id"], docs[0])
            for name, info in report["files"].items():
                self.assertEqual(info["sha256"], hashlib.sha256((output / name).read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "new or empty"):
                write_bundle(output, {"validation": articles, "test": other},
                    {"validation": audit, "test": other_audit}, lengths=[16],
                    prefix_length=4, provenance={})

    def test_refuses_cross_split_article_duplication_before_writing(self):
        _, articles, audit = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            with self.assertRaisesRegex(ValueError, "duplicate complete articles"):
                write_bundle(output, {"validation": articles, "test": articles},
                    {"validation": audit, "test": audit}, lengths=[16],
                    prefix_length=4, provenance={})
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
