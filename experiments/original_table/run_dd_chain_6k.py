#!/usr/bin/env python3
"""Train Basic DD with active-chain proposals through 6k, then diagnose it."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import run


original_protocol = run.protocol
original_overrides = run.overrides


def protocol(study):
    cfg = original_protocol(study)
    cfg["arms"] = [
        arm.copy() for arm in cfg["arms"] if arm["id"] == "basic_dd"]
    cfg["arms"][0]["phases"] = ["basic_6000"]
    return cfg


def overrides(cfg, arm, phase, study, state):
    args, run_dir, unused_resume = original_overrides(
        cfg, arm, phase, study, state)
    if phase != "basic_6000":
        raise ValueError("continuous DD study permits only basic_6000")
    args = [
        "checkpointing.resume_from_ckpt=false"
        if value == "checkpointing.resume_from_ckpt=true" else value
        for value in args
        if not value.startswith("checkpointing.resume_ckpt_path=")
    ]
    return args, run_dir, None


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _source_files(repo):
    tracked = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "-z"])
    files = {item.decode() for item in tracked.split(b"\0") if item}
    files.update({
        "experiments/original_table/run_dd_chain_6k.py",
        "experiments/original_table/train_dd_chain_6k.sbatch",
        "scripts/launch_dd_chain_6k.sh",
    })
    return sorted(files)


def prepare(options):
    precision = os.environ.get("CCF_ROTARY_CACHE_PRECISION", "bf16")
    if precision != "bf16":
        raise ValueError("DD chain 6k study requires BF16 rotary cache")
    repo = options.repo.resolve()
    dest = options.study.resolve()
    cache = options.cache.resolve()
    if dest.exists():
        raise FileExistsError("Choose a new output directory: " + str(dest))
    if repo == dest or repo in dest.parents:
        raise ValueError("Study must be outside the repository")
    branch = subprocess.check_output(
        ["git", "-C", str(repo), "branch", "--show-current"],
        text=True).strip()
    if branch != "original_table_base_crf-recovery":
        raise ValueError("Run from original_table_base_crf-recovery")

    cfg = protocol(dest)
    backbone = cache / "checkpoints/mdlm-owt-backbone.pt"
    if not options.source_only and run.sha(backbone) != cfg["backbone_sha256"]:
        raise ValueError("Pinned backbone hash mismatch")
    if repo != Path(__file__).resolve().parents[2]:
        raise ValueError("Use this checked-out repository as --repo")

    source_files = _source_files(repo)
    missing = [name for name in source_files if not (repo / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing source files: " + repr(missing))
    identities = {name: run.sha(repo / name) for name in source_files}
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True)
    diff = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--binary"])

    dest.mkdir(parents=True)
    (dest / "logs").mkdir()
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True).strip()
    state = {
        "cache": str(cache),
        "backbone": str(backbone),
        "source_commit": commit,
        "repository_clean": not bool(status.strip()),
        "source_status": status.splitlines(),
        "source_diff_sha256": _sha256_bytes(diff),
        "repo": str(repo),
        "source_only": options.source_only,
        "source_identities": identities,
        "protocol_sha256": run.sha(
            Path(__file__).resolve().parent / "protocol.json"),
        "rotary_cache_precision": "bf16",
        "eval_rotary_cache_precision": "bf16",
        "edge_source_diagnostics": True,
        "profile": "basic_dd_chain_6k_continuous",
        "training_phases": ["basic_6000"],
        "continuous_dataset_iteration": True,
    }
    run.write(dest / "study.json", state)
    cells = [{
        "arm_id": "basic_dd",
        "family": "B",
        "arm": "dynamic_dynamic",
        "rank": 16,
        "embedding": "shared",
        "step": 6000,
        "sampling_steps": steps,
        "num_samples": 100,
        "base_seed": 100001,
        "mode": "structured_joint",
    } for steps in (8, 16, 32)]
    run.write(dest / "pilot-cells.json", [])
    run.write(dest / "confirmation-cells.json", cells)
    print(json.dumps({
        "study": str(dest),
        "train_tasks": 1,
        "confirmation_cells": len(cells),
        "rotary_cache_precision": "bf16",
        "edge_source_diagnostics": True,
        "training_phases": state["training_phases"],
        "continuous_dataset_iteration": True,
        "repository_clean": state["repository_clean"],
    }, indent=2))


if __name__ == "__main__":
    run.protocol = protocol
    run.overrides = overrides
    run.prepare = prepare
    run.main()
