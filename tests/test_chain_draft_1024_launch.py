"""Launch gates and scheduler wiring, without submitting real jobs."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import launch_chain_draft_1024 as launcher


class DraftLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        self.source.mkdir()
        for name in launcher.GATED_FILES:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture: " + name)
        self.data = self.base / "data"
        self.data.mkdir()
        for role in ("train", "dev"):
            (self.data / f"{role}.pt").write_bytes(role.encode())
        self.checkpoint = self.base / "backbone.pt"
        self.checkpoint.write_bytes(b"released-frozen-fixture")
        self.manifest = {"length": 1024, "document_quotas": {
            "train": 200000, "dev": 2000, "heldout_eval": 100000},
            "files": {role + ".pt": launcher.sha256(self.data / f"{role}.pt") for role in ("train", "dev")},
            "splits": {"train": {"rows": 7, "packed_tokens": 7168}, "dev": {"rows": 3}}}
        launcher.save_json(self.manifest, self.data / "manifest.json")
        self.config = launcher.configuration(SimpleNamespace(epochs=1, seeds=[1], parallel_heads=2))
        self.config.update(data=str(self.data), validation=str(self.base / "validation.json"),
                           checkpoint=str(self.checkpoint), run_parent=str(self.base / "runs"))
        self.validation = {"status": "passed", "data_manifest_sha256": launcher.sha256(self.data / "manifest.json"),
            "source_sha256": {name: launcher.sha256(self.source / name) for name in launcher.GATED_FILES},
            "backbone": {"loaded_file_sha256": launcher.sha256(self.checkpoint)},
            "training_profiles": {head["mode"]: {"backbone_unchanged": True, "config": {
                **{name: self.config[name] for name in ("length", "k", "batch_size", "backbone_batch_size")},
                "rank": head["rank"]}} for head in self.config["heads"]}}
        launcher.save_json(self.validation, self.config["validation"])

    def test_default_preview_never_submits(self):
        with patch("sys.argv", ["launch_chain_draft_1024.py"]), patch.object(launcher, "submit") as submit:
            with redirect_stdout(io.StringIO()) as output:
                launcher.main()
        submit.assert_not_called()
        self.assertFalse(json.loads(output.getvalue())["submitted"])

    def test_gate_resolves_one_pass_and_all_development_rows(self):
        result = launcher.gate(self.config, self.source)
        self.assertEqual(result["target_steps"], 2)
        self.assertEqual(result["target_token_exposures_per_head"], 8192)
        self.assertEqual(result["dev_rows"], 3)

    def test_gate_rejects_failed_validation_changed_data_and_changed_code(self):
        for change, error in (("status", "Step 3"), ("manifest", "manifest"),
                              ("tokens", "tokens changed"), ("code", "code differs"),
                              ("backbone", "backbone differs")):
            with self.subTest(change=change):
                if change == "status":
                    path = Path(self.config["validation"])
                    content = json.dumps({**self.validation, "status": "failed"}).encode()
                elif change == "manifest":
                    path, content = self.data / "manifest.json", b"{}"
                elif change == "tokens":
                    path, content = self.data / "train.pt", b"changed tokens"
                elif change == "code":
                    path, content = self.source / "chain_crf/core.py", b"changed code"
                else:
                    path, content = self.checkpoint, b"changed backbone"
                original = path.read_bytes()
                path.write_bytes(content)
                with self.assertRaisesRegex(ValueError, error):
                    launcher.gate(self.config, self.source)
                path.write_bytes(original)

    def test_submission_dependencies_and_cpu_gpu_resources(self):
        for name in ("scripts/train_four_models.sh", "scripts/launch_chain_draft_1024.py",
                     "chain_crf/__init__.py", "models/__init__.py", "CHAIN_CRF.md", "requirements-chain-crf.txt",
                     "notes/mdlm-crf-1024-decisions.md"):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
        responses = [SimpleNamespace(returncode=0, stdout="9001\n", stderr=""),
                     SimpleNamespace(returncode=0, stdout="9002\n", stderr="")]
        with patch.object(launcher, "BASE", self.base), patch.object(launcher, "ROOT", self.source), \
             patch.object(launcher.subprocess, "run", side_effect=responses) as sbatch, \
             patch.object(launcher.subprocess, "check_output", return_value="fixture-commit\n"), \
             redirect_stdout(io.StringIO()):
            launcher.submit(self.config)
        calls = [call.args[0] for call in sbatch.call_args_list]
        self.assertEqual(len(calls), 2)
        for command in calls:
            self.assertIn("--dependency=afterok:1579295:1579324", command)
        self.assertFalse(any(arg.startswith("--gres") for arg in calls[0]))
        self.assertIn("--gres=gpu:1", calls[1])
        self.assertIn("--array=0-2%2", calls[1])
        self.assertIn("--signal=B:TERM@300", calls[1])


if __name__ == "__main__":
    unittest.main()
