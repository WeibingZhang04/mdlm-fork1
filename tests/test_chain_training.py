"""Offline end-to-end checks for data separation, head training and resume."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest

import torch

from chain_crf.backbone import SyntheticBackbone
from chain_crf.data import BatchStream, assert_disjoint, atomic_torch_save, canonical_hash, document_split, load_token_data
from scripts.prepare_chain_data import prepare
from scripts.train_chain_crf import (
    checkpoint_payload, continue_initialization_stream, corrupt, evaluate, initialization_token_exposures,
    main, make_head, restore, score_batch,
)


class ChainTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_hash_splits_keep_all_document_chunks_together(self):
        self.assertEqual(document_split("a"), document_split("a"))
        found = {name: [] for name in ("train", "dev", "test")}
        for index in range(10000):
            doc = f"document-{index}"
            found[document_split(doc)].append(doc)
        self.assertTrue(all(found.values()))
        rows = []
        for split, docs in found.items():
            for index, doc in enumerate(docs[:4]):
                rows.append({"document_id": doc, "input_ids": [index+1, len(rows)+1, 3, 4]})
        selected, docs, report = prepare(rows, tokenizer=SimpleNamespace(vocab_size=100),
            length=4, targets={"train": 2, "dev": 2, "test": 2}, exclusions=[found["test"][0]])
        self.assertEqual(report["split_examples"], {"train": 2, "dev": 2, "test": 2})
        self.assertFalse(set(docs["train"]) & set(docs["dev"]))
        self.assertNotIn(found["test"][0], docs["test"])

    def test_disjoint_rejects_document_and_token_leakage(self):
        x, y = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])
        with self.assertRaisesRegex(ValueError, "document overlap"):
            assert_disjoint(x, ["a"], y, ["a"])
        with self.assertRaisesRegex(ValueError, "duplication"):
            assert_disjoint(x, ["a"], x, ["b"])
        assert_disjoint(x, ["a"], y, ["b"])

    def test_cached_rows_partition_without_joining_or_changing_document_identity(self):
        docs = {}
        for index in range(10000):
            doc = f"document-{index}"
            docs.setdefault(document_split(doc), doc)
            if len(docs) == 3:
                break
        rows = [{"document_id": doc, "input_ids": [offset*20+i for i in range(10)]}
                for offset, doc in enumerate(docs.values())]
        selected, identities, _ = prepare(rows, tokenizer=SimpleNamespace(vocab_size=100),
            length=4, targets={"train": 2, "dev": 2, "test": 2})
        for split in docs:
            row = next(row for row in rows if row["document_id"] == docs[split])
            self.assertEqual(selected[split], [row["input_ids"][:4], row["input_ids"][4:8]])
            self.assertEqual(identities[split], [docs[split], docs[split]])

    def test_stream_resume_in_middle_of_epoch_and_wraparound(self):
        x = torch.arange(15).reshape(5, 3)
        stream = BatchStream(x, 4, 9)
        stream.next()
        state = stream.state_dict()
        expected = [stream.next() for _ in range(3)]
        resumed = BatchStream(x, 4, 1)
        resumed.load_state_dict(state)
        for batch in expected:
            torch.testing.assert_close(resumed.next(), batch)

    def test_fresh_corruption_reproducible_and_never_uses_gold_selection(self):
        x = torch.arange(24).reshape(4, 6) % 16
        a = corrupt(x, 16, torch.Generator().manual_seed(4))
        b = corrupt(x, 16, torch.Generator().manual_seed(4))
        for first, second in zip(a, b):
            torch.testing.assert_close(first, second)
        self.assertTrue(torch.equal(a[0][~a[1]], x[~a[1]]))
        self.assertTrue(bool((a[0][a[1]] == 16).all()))

    def test_all_three_heads_train_finitely_and_backbone_remains_frozen(self):
        backbone = SyntheticBackbone()
        tokens = torch.randint(0, 16, (4, 6))
        original = {k: v.clone() for k, v in backbone.state_dict().items()}
        for mode in ("global", "contextual", "independent"):
            head = make_head(mode, 17, 12, 3, 8)
            optimizer = torch.optim.AdamW(head.parameters(), lr=.03)
            generator = torch.Generator().manual_seed(19)
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = score_batch(backbone, head, mode, tokens, k=4, device="cpu", generator=generator)
                loss.backward()
                optimizer.step()
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(metrics["masked_tokens"], 0)
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in head.parameters()))
        for name, value in backbone.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    def test_optimizer_rng_resume_matches_uninterrupted_step(self):
        backbone = SyntheticBackbone()
        tokens = torch.randint(0, 16, (5, 6))
        head = make_head("global", 17, 12, 3)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        stream = BatchStream(tokens, 3, 5)
        rng = torch.Generator().manual_seed(7)

        def update():
            optimizer.zero_grad(set_to_none=True)
            loss, _ = score_batch(backbone, head, "global", stream.next(), k=3, device="cpu", generator=rng)
            loss.backward(); optimizer.step(); scheduler.step()
            return loss.detach().clone()

        update()
        identity = {"fixture": True}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"checkpoint.pt"
            atomic_torch_save(checkpoint_payload(head, optimizer, scheduler, stream, rng, 1,
                                                 identity, None, {"mode": "global"}), path)
            expected_loss = update()
            expected = {k: v.clone() for k,v in head.state_dict().items()}
            payload = torch.load(path, weights_only=True)
            restore(payload, head, optimizer, scheduler, stream, rng, identity)
            actual = update()
            torch.testing.assert_close(actual, expected_loss, rtol=0, atol=0)
            for key, value in head.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_contextual_warm_start_continues_next_global_batch_and_corruption(self):
        tokens = torch.arange(60).reshape(10, 6) % 16
        original = BatchStream(tokens, 4, 1001)
        original_rng = torch.Generator().manual_seed(2001)
        for _ in range(4):
            corrupt(original.next(), 16, original_rng)
        payload = {"stream": original.state_dict(), "mask_rng": original_rng.get_state(),
                   "step": 4, "identity": {"train_sha256": "same-bytes", "train_examples": len(tokens)},
                   "config": {"seed": 1, "batch_size": 4, "length": 6}}
        expected_batch = original.next()
        expected_corruption = corrupt(expected_batch, 16, original_rng)
        contextual = BatchStream(tokens, 4, 1001)
        contextual_rng = torch.Generator().manual_seed(2001)
        steps = continue_initialization_stream(payload, contextual, contextual_rng,
            train_sha256="same-bytes", config={"seed": 1, "batch_size": 4, "length": 6})
        self.assertEqual(steps, 4)
        actual_batch = contextual.next()
        torch.testing.assert_close(actual_batch, expected_batch, rtol=0, atol=0)
        for actual, expected in zip(corrupt(actual_batch, 16, contextual_rng), expected_corruption):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "different training data"):
            continue_initialization_stream(payload, contextual, contextual_rng,
                train_sha256="wrong-bytes", config=payload["config"])

    def test_initialization_exposure_checks_its_own_identity(self):
        config = {"batch_size": 8, "length": 16}
        identity = {"config": config}
        initial = {"step": 5, "config": config, "identity": identity,
                   "identity_sha256": canonical_hash(identity)}
        self.assertEqual(initialization_token_exposures(initial), 640)
        initial["config"] = {"batch_size": 2, "length": 4}
        with self.assertRaisesRegex(ValueError, "identity checksum/configuration"):
            initialization_token_exposures(initial)

    def test_cli_offline_run_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root/"data"
            generator = torch.Generator().manual_seed(1)
            for split, n in (("train", 8), ("dev", 4)):
                atomic_torch_save({"tokens": torch.randint(0, 16, (n, 6), generator=generator),
                                  "document_ids": [f"{split}-{i}" for i in range(n)]}, data/f"{split}.pt")
            args = ["--data", str(data), "--output", str(root/"run"), "--mode", "global",
                    "--synthetic-backbone", "--device", "cpu", "--length", "6", "--k", "4",
                    "--rank", "3", "--steps", "2", "--eval-every", "2", "--save-every", "1",
                    "--batch-size", "2", "--threads", "1"]
            main(args)
            checkpoint = root/"run/last.pt"
            self.assertEqual(torch.load(checkpoint, weights_only=True)["step"], 2)
            args[args.index("--steps")+1] = "3"
            main(args + ["--resume", str(checkpoint)])
            self.assertEqual(torch.load(checkpoint, weights_only=True)["step"], 3)
            # A contextual warm start continues stream state but has its own
            # update counter and optimizer; those semantics survive resume.
            context_args = list(args)
            context_args[context_args.index("--output")+1] = str(root/"context")
            context_args[context_args.index("--mode")+1] = "contextual"
            context_args[context_args.index("--steps")+1] = "2"
            main(context_args + ["--init-global", str(checkpoint), "--continue-init-stream"])
            contextual_checkpoint = root/"context/last.pt"
            context_payload = torch.load(contextual_checkpoint, weights_only=True)
            self.assertEqual(context_payload["identity"]["initialization_steps"], 3)
            self.assertTrue(context_payload["identity"]["continue_init_stream"])
            global_payload = torch.load(checkpoint, weights_only=True)
            expected_stream = BatchStream(torch.load(data/"train.pt", weights_only=True)["tokens"], 2, 1001)
            expected_stream.load_state_dict(global_payload["stream"])
            expected_rng = torch.Generator()
            expected_rng.set_state(global_payload["mask_rng"])
            for _ in range(2):
                corrupt(expected_stream.next(), 16, expected_rng)
            self.assertEqual(context_payload["stream"]["cursor"], expected_stream.cursor)
            torch.testing.assert_close(context_payload["mask_rng"], expected_rng.get_state(), rtol=0, atol=0)
            context_args[context_args.index("--steps")+1] = "3"
            main(context_args + ["--resume", str(contextual_checkpoint)])
            resumed_context = torch.load(contextual_checkpoint, weights_only=True)
            self.assertEqual(resumed_context["step"], 3)
            self.assertEqual(resumed_context["identity"]["initialization_steps"], 3)
            self.assertTrue(resumed_context["identity"]["continue_init_stream"])
            # Without stream continuation, warm starts may intentionally use
            # smaller dimensions. Count the original 3*2*6 exposures unchanged.
            smaller_args = list(context_args)
            smaller_args[smaller_args.index("--output")+1] = str(root/"smaller-context")
            smaller_args[smaller_args.index("--length")+1] = "3"
            smaller_args[smaller_args.index("--batch-size")+1] = "1"
            smaller_args[smaller_args.index("--steps")+1] = "2"
            main(smaller_args + ["--init-global", str(checkpoint)])
            result = json.loads((root/"smaller-context/results.json").read_text())
            self.assertEqual(result["initialization_token_exposures"], 36)
            self.assertEqual(result["trained_tokens"], 6)
            self.assertEqual(result["total_training_token_exposures"], 42)


if __name__ == "__main__":
    unittest.main()
