#!/usr/bin/env python3
"""Preview or submit draft frozen-head training, gated by successful step 3.

No arguments means a read-only preview. --submit is an explicit launch action.
The first draft trains each head from scratch for one approximate data pass;
it is not a claim of convergence. Defaults and rationale are recorded in notes.
Only Python's standard library is needed on the login node.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/u401/n23zhang/rework-data")
DATA = BASE / "data/draft-full-scale-training/owt-1024-2026-10-01_050424_UTC"
VALIDATION = BASE / "runs/draft-full-scale-training/step3-owt-1024-2026-10-01_051416_UTC/results/validation.json"
GATED_FILES = (
    "chain_crf/backbone.py", "chain_crf/core.py", "chain_crf/heads.py", "chain_crf/data.py",
    "chain_crf/counts.py", "chain_crf/generation.py", "models/dit.py", "configs/model/small.yaml",
    "scripts/train_chain_crf.py", "scripts/build_chain_counts.py", "scripts/evaluate_chain_crf.py",
    "scripts/prepare_released_mdlm_owt.py", "scripts/validate_chain_1024.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def configuration(args):
    return {
        "stage": "draft_full_scale_head_training", "backbone_frozen": True,
        "data": str(DATA), "validation": str(VALIDATION),
        "checkpoint": str(BASE / "checkpoints/mdlm-owt.pt"), "cache_dir": str(BASE / "hf-cache/hub"),
        "dependencies": ["1579295", "1579324"],
        "run_parent": str(BASE / "runs/draft-full-scale-training"),
        "folder_pattern": "train_YYYY_MM_DD_HH_MM_SS_UTC",
        "heads": [{"mode": "global", "rank": 32}, {"mode": "contextual", "rank": 32},
                  {"mode": "independent", "rank": 64}],
        "seeds": args.seeds, "epochs": args.epochs,
        "steps_rule": "ceil(epochs * prepared_train_rows / batch_size); at most three repeated rows in final batch",
        "length": 1024, "k": 64, "batch_size": 4, "backbone_batch_size": 1,
        "learning_rate": 3e-4, "weight_decay": 0., "gradient_clip": 1.,
        "warmup_steps": 1000, "mlp_size": 128, "initialization": "fresh heads; no warm start",
        "eval_every": 5000, "save_every": 500,
        "development": "all prepared dev rows; initial, every 5000 updates, and final evaluation",
        "checkpoint_selection": "lowest joint denoising NLL per masked token on fixed dev corruption draws",
        "counts": "all prepared training tokens; within-row edges; smoothing 0.1; CPU only",
        "partition": "ALL", "node": "watgpu108", "expected_gpu": "NVIDIA RTX 6000 Ada Generation",
        "parallel_heads": args.parallel_heads, "gpu_per_head": 1, "cpus_per_task": 4,
        "host_memory_gb": 24, "head_wall_hours": 24, "head_max_seconds": 23 * 3600,
        "count_wall_hours": 12, "count_cpus": 4, "count_host_memory_gb": 24,
        "time_limit_policy": "save resumable checkpoints; report incomplete if target not reached",
        "sampling": "deferred; no generation or final held-out scoring is launched",
    }


def gate(config, source_root):
    """Fail before learning if data, validation, source, or backbone differ."""
    validation = json.loads(Path(config["validation"]).read_text())
    if validation.get("status") != "passed":
        raise ValueError("Step 3 has not passed")
    data = Path(config["data"])
    if sha256(data / "manifest.json") != validation["data_manifest_sha256"]:
        raise ValueError("Data manifest differs from the step-3 validated bundle")
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest["length"] != 1024 or manifest["document_quotas"] != {
            "train": 200000, "dev": 2000, "heldout_eval": 100000}:
        raise ValueError("Unexpected draft dataset dimensions")
    for name in GATED_FILES:
        if sha256(source_root / name) != validation["source_sha256"][name]:
            raise ValueError(f"Training code differs from step 3: {name}")
    if sha256(config["checkpoint"]) != validation["backbone"]["loaded_file_sha256"]:
        raise ValueError("Released backbone differs from step 3")
    for role in ("train", "dev"):
        if sha256(data / f"{role}.pt") != manifest["files"][f"{role}.pt"]:
            raise ValueError(f"Prepared {role} tokens changed after validation")
    for head in config["heads"]:
        profile = validation["training_profiles"][head["mode"]]
        for name in ("length", "k", "batch_size", "backbone_batch_size"):
            if profile["config"][name] != config[name]:
                raise ValueError(f"Unvalidated profile setting: {name}")
        if profile["config"]["rank"] != head["rank"] or not profile["backbone_unchanged"]:
            raise ValueError("Head architecture or backbone-freeze check differs")
    train_rows = manifest["splits"]["train"]["rows"]
    steps = (config["epochs"] * train_rows + config["batch_size"] - 1) // config["batch_size"]
    return {"target_steps": steps, "train_rows": train_rows,
            "dev_rows": manifest["splits"]["dev"]["rows"],
            "unique_training_row_tokens": manifest["splits"]["train"]["packed_tokens"],
            "target_token_exposures_per_head": steps * config["batch_size"] * config["length"],
            "data_manifest_sha256": validation["data_manifest_sha256"],
            "validation_sha256": sha256(config["validation"])}


def snapshot(root, destination):
    destination.mkdir()
    # Copy only this launcher's dependencies, not unrelated work in the checkout.
    names = (*GATED_FILES, "chain_crf/__init__.py", "models/__init__.py",
             "scripts/launch_chain_draft_1024.py", "scripts/train_four_models.sh", "CHAIN_CRF.md",
             "requirements-chain-crf.txt", "notes/mdlm-crf-1024-decisions.md")
    paths = {root / name for name in names}
    hashes = {}
    for original in sorted(paths):
        name = original.relative_to(root)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        hashes[str(name)] = sha256(target)
    return hashes


def submit(config):
    if BASE.stat().st_uid != os.getuid() or ROOT.stat().st_uid != os.getuid():
        raise PermissionError("Only submit from the user's own project and data directories")
    stamp = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S_UTC")
    run = Path(config["run_parent"]) / ("train_" + stamp)
    run.mkdir(parents=True, exist_ok=False)
    config["source_sha256"] = snapshot(ROOT, run / "source")
    try:
        config["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        config["git_head"] = "unavailable"
    save_json(config, run / "launch-config.json")
    # If step 3 has already completed, check it before any job is submitted.
    # Otherwise Slurm dependencies and the mandatory worker gate defer checks.
    if Path(config["validation"]).exists():
        save_json(gate(config, run / "source"), run / "validated-budget.json")
    common = ["sbatch", "--parsable", "--partition=" + config["partition"],
        "--nodelist=" + config["node"], "--dependency=afterok:" + ":".join(config["dependencies"]),
        "--chdir=" + str(run / "source"),
        "--export=ALL,CHAIN_TRAIN_RUN_DIR=" + str(run)]
    script = str(run / "source/scripts/train_four_models.sh")
    jobs = []
    requests = [
        ("counts", ["--job-name=draft1024-count", f"--cpus-per-task={config['count_cpus']}",
                    f"--mem={config['count_host_memory_gb']}G", f"--time={config['count_wall_hours']:02d}:00:00",
                    "--output=" + str(run / "counts_slurm_%j.out"), "--error=" + str(run / "counts_slurm_%j.err"),
                    script, "--worker", "count"]),
        ("heads", ["--job-name=draft1024-head", f"--cpus-per-task={config['cpus_per_task']}",
                   f"--mem={config['host_memory_gb']}G", f"--time={config['head_wall_hours']:02d}:00:00",
                   "--gres=gpu:1", "--signal=B:TERM@300",
                   f"--array=0-{len(config['heads']) * len(config['seeds']) - 1}%{config['parallel_heads']}",
                   "--output=" + str(run / "head_slurm_%A_%a.out"), "--error=" + str(run / "head_slurm_%A_%a.err"),
                   script, "--worker", "head"]),
    ]
    for name, options in requests:
        result = subprocess.run(common + options, capture_output=True, text=True)
        record = {"kind": name, "returncode": result.returncode,
                  "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
        if result.returncode == 0:
            record["job_id"] = result.stdout.strip().split(";")[0]
        jobs.append(record)
        save_json({"run_directory": str(run), "jobs": jobs}, run / "submitted-jobs.json")
        if result.returncode:
            raise RuntimeError(f"Submission failed; inspect {run / 'submitted-jobs.json'}; existing submissions were retained")
    print(json.dumps({"run_directory": str(run), "jobs": jobs}, indent=2))


def worker(kind, run):
    config = json.loads((run / "launch-config.json").read_text())
    source = run / "source"
    resolved = gate(config, source)
    if kind == "count":
        name = "counts"
        arguments = ["scripts/build_chain_counts.py", "--data", str(Path(config["data"]) / "train.pt"),
            "--output", str(run / "counts/owt-counts.pt"), "--max-tokens", str(resolved["unique_training_row_tokens"]),
            "--vocab-size", "50258", "--mask-id", "50257", "--smoothing", ".1"]
    else:
        import torch
        actual_gpu = torch.cuda.get_device_name(0)
        if actual_gpu != config["expected_gpu"]:
            raise ValueError(f"Expected {config['expected_gpu']}; got {actual_gpu}")
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        seed = config["seeds"][index // len(config["heads"])]
        head = config["heads"][index % len(config["heads"])]
        name = head["mode"] + "-seed" + str(seed)
        arguments = ["scripts/train_chain_crf.py", "--data", config["data"],
            "--checkpoint", config["checkpoint"], "--cache-dir", config["cache_dir"],
            "--output", str(run / name), "--mode", head["mode"], "--rank", str(head["rank"]),
            "--seed", str(seed), "--device", "cuda", "--steps", str(resolved["target_steps"]),
            "--max-dev-examples", str(resolved["dev_rows"])]
        for flag, key in (("--length", "length"), ("--k", "k"), ("--batch-size", "batch_size"),
            ("--backbone-batch-size", "backbone_batch_size"), ("--learning-rate", "learning_rate"),
            ("--weight-decay", "weight_decay"), ("--warmup-steps", "warmup_steps"),
            ("--gradient-clip", "gradient_clip"), ("--eval-every", "eval_every"),
            ("--save-every", "save_every"), ("--mlp-size", "mlp_size"), ("--max-seconds", "head_max_seconds"),
            ("--threads", "cpus_per_task")):
            arguments.extend([flag, str(config[key])])
    save_json({**resolved, "command": [sys.executable, "-u", *arguments]}, run / "resolved" / (name + ".json"))
    print(json.dumps({"starting": name, **resolved}), flush=True)
    log = (run / (name + ".log")).open("x")
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    os.chdir(source)
    os.execv(sys.executable, [sys.executable, "-u", *arguments])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true", help="Read-only preview (default)")
    action.add_argument("--submit", action="store_true", help="Submit jobs; use only after reviewing the plan")
    action.add_argument("--worker", choices=("count", "head"), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--parallel-heads", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or not 1 <= args.parallel_heads <= 3 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Need positive epochs, one to three concurrent heads, and distinct seeds")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Seeds must be nonnegative")
    if args.worker:
        if args.run_dir is None:
            raise ValueError("Worker needs its saved run directory")
        worker(args.worker, args.run_dir)
        return
    config = configuration(args)
    if args.submit:
        submit(config)
    else:
        manifest_path = Path(config["data"]) / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            config["prepared_train_rows"] = manifest["splits"]["train"]["rows"]
            config["target_steps"] = (config["epochs"] * config["prepared_train_rows"] + 3) // 4
        config["submitted"] = False
        print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
