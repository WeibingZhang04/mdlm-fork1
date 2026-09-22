"""CPU-only checks; sbatch is mocked and no model or GPU is loaded."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "experiments/original_table"
sys.path.insert(0, str(HERE))
import run
import run_fd_dd_6k as profile


class FDLauncherTests(unittest.TestCase):
    def test_historical_training_arguments_are_preserved(self):
        study = Path("/study")
        state = {"backbone": "/backbone", "cache": "/cache"}
        full = run.protocol(study)
        limited = profile.protocol(study)
        self.assertEqual([a["id"] for a in limited["arms"]], ["basic_fd", "basic_dd"])
        for arm in limited["arms"]:
            self.assertEqual(arm["phases"], ["basic_1000", "basic_3000", "basic_6000"])
            original = next(a for a in full["arms"] if a["id"] == arm["id"])
            for phase in arm["phases"]:
                self.assertEqual(run.overrides(limited, arm, phase, study, state),
                                 run.overrides(full, original, phase, study, state))

    def test_preparation_tracks_checkout_and_only_six_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp) / "study"
            subprocess.run([sys.executable, str(HERE / "run_fd_dd_6k.py"),
                            "prepare", "--study", str(study), "--cache", tmp,
                            "--source-only"], check=True, capture_output=True)
            state = run.read(study / "study.json")
            self.assertEqual(state["repo"], str(REPO))
            self.assertEqual(state["profile"], "basic_fd_dd_6k")
            for name in ("run_fd_dd_6k.py", "train_fd_dd_6k.sbatch", "evaluate.sbatch"):
                key = "experiments/original_table/" + name
                self.assertEqual(state["source_identities"][key], run.sha(REPO / key))
            cells = run.read(study / "confirmation-cells.json")
            self.assertEqual([(c["arm_id"], c["sampling_steps"]) for c in cells],
                             [(a, s) for a in ("basic_fd", "basic_dd") for s in (8, 16, 32)])
            self.assertTrue(all(c["step"] == 6000 and c["num_samples"] == 100
                                and c["base_seed"] == 100001 for c in cells))
            self.assertEqual(run.read(study / "pilot-cells.json"), [])
            self.assertFalse(list(study.rglob("*.py")) + list(study.rglob("*.sbatch")))

    def test_submission_paths_dependency_and_exclusion(self):
        # Preparation checks the real frozen backbone. Only sbatch is replaced.
        cache = os.environ.get("CCF_TEST_CACHE")
        if not cache:
            self.skipTest("Set CCF_TEST_CACHE to verify submission with the real backbone")
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp) / "study"
            calls = Path(tmp) / "calls"
            env = dict(os.environ, CCF_CACHE_ROOT=cache, CCF_STUDY=str(study),
                       CCF_PREPARE_PYTHON=sys.executable, CCF_EXCLUDE_NODES="watgpu608",
                       CCF_TEST_CALLS=str(calls), PYTHONDONTWRITEBYTECODE="1")
            shell = """
sbatch() {
    printf '%s\\n' "$@" >> "$CCF_TEST_CALLS"
    printf '%s\\n' END >> "$CCF_TEST_CALLS"
    printf '%s\\n' '900001;test'
}
launcher_path="$1"
shift
source "$launcher_path"
"""
            subprocess.run(["bash", "-c", shell, "test",
                            str(REPO / "scripts/launch_original_table_fd_dd_6k.sh")],
                           env=env, check=True, capture_output=True)
            commands = calls.read_text().split("END\n")
            self.assertEqual(len(commands), 3)
            self.assertIn(str(HERE / "train_fd_dd_6k.sbatch"), commands[0])
            self.assertIn(str(HERE / "evaluate.sbatch"), commands[1])
            self.assertIn("--dependency=afterok:900001", commands[1])
            for cmd in commands[:2]:
                self.assertIn("--exclude=watgpu608", cmd)
                self.assertIn("--chdir=" + str(REPO), cmd)
            self.assertEqual((study / "train-job-id.txt").read_text().strip(), "900001")
            self.assertFalse(list(study.rglob("*.py")) + list(study.rglob("*.sbatch")))


if __name__ == "__main__":
    unittest.main()
