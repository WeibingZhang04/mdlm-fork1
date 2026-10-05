"""Launch gates and explicit training budgets; no Slurm, model, or GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import launch_chain_draft_training as launcher


class DraftLauncherTest(unittest.TestCase):
    def fixture(self, base):
        data, validation = base / "data", base / "validation"
        data.mkdir()
        validation.mkdir()
        (base / "checkpoints").mkdir()
        checkpoint = base / "checkpoints/mdlm-owt.pt"
        checkpoint.write_bytes(b"not a real checkpoint: launch-plan test only")
        for role in ("train", "dev"):
            (data / f"{role}.pt").write_bytes(role.encode())
        manifest = {"length": 1024, "selection_report": {"evaluation_population": "full-tail"},
            "files": {f"{role}.pt": launcher.sha(data / f"{role}.pt") for role in ("train", "dev")},
            "splits": {"train": {"rows": 100, "packed_text_tokens": 100000, "packed_tokens": 102400},
                       "dev": {"rows": 7}}}
        (data / "manifest.json").write_text(json.dumps(manifest))
        report = {"status": "passed", "data_manifest_sha256": launcher.sha(data / "manifest.json"),
            "gpu": "test-gpu", "backbone": {"loaded_file_sha256": launcher.sha(checkpoint)},
            "training_profiles": {mode: {"config": {"rank": 64 if mode == "independent" else 32,
                                                     "batch_size": 4, "k": 64}} for mode in launcher.MODES}}
        (validation / "validation.json").write_text(json.dumps(report))
        sources = {"chain_crf/core.py": launcher.sha(launcher.ROOT / "chain_crf/core.py")}
        for directory, filename, content in (
            ("cli-train", "protocol.json", {"identity": {"source_sha256": sources}}),
            ("generation-backbone", "manifest.json", {"source_sha256": sources})):
            (validation / directory).mkdir()
            (validation / directory / filename).write_text(json.dumps(content))
        return data, validation

    def test_gate_rejects_failed_checks_or_modified_data(self):
        with tempfile.TemporaryDirectory() as temp:
            data, validation = self.fixture(Path(temp))
            launcher.check_validation(data, validation)
            (data / "train.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Data changed"):
                launcher.check_validation(data, validation)
            report = json.loads((validation / "validation.json").read_text())
            report["status"] = "failed"
            (validation / "validation.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "Successful step-3"):
                launcher.check_validation(data, validation)

    def test_gate_rejects_changed_training_code(self):
        with tempfile.TemporaryDirectory() as temp:
            data, validation = self.fixture(Path(temp))
            p = validation / "generation-backbone/manifest.json"
            p.write_text(json.dumps({"source_sha256": {"chain_crf/core.py": "wrong"}}))
            with self.assertRaisesRegex(ValueError, "Code changed"):
                launcher.check_validation(data, validation)

    def test_dry_plan_uses_full_train_dev_count_budget_and_never_submits(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data, validation = self.fixture(base)
            args = ["launch", "--data", str(data), "--validation", str(validation),
                "--steps", "20000", "--seeds", "1", "2", "--parallel-heads", "2",
                "--wall-hours", "8", "--eval-every", "2000"]
            with patch.object(launcher, "BASE", base), patch("sys.argv", args), patch.object(launcher.subprocess, "run") as submit:
                launcher.main()
                submit.assert_not_called()
            run = next((base / "runs/draft-full-scale-training").iterdir())
            plan = json.loads((run / "plan.json").read_text())
            self.assertEqual(len(plan["tasks"]), 6)
            self.assertEqual(plan["tokens_seen_per_head"], 20000 * 4 * 1024)
            self.assertEqual(plan["all_dev_rows"], 7)
            self.assertTrue(plan["backbone_frozen"])
            for task in plan["tasks"]:
                command = task["command"]
                self.assertNotIn("--max-train-examples", command)
                self.assertNotIn("--init-global", command)
                self.assertEqual(command[command.index("--max-dev-examples") + 1], "7")
                self.assertEqual(command[command.index("--length") + 1], "1024")
            self.assertIn("--max-tokens 102400", (run / "fit-counts.sbatch").read_text())
            self.assertIn("--array=0-5%2", plan["submission_commands"][1])


if __name__ == "__main__":
    unittest.main()
