#!/usr/bin/env python3
"""Validate the draft 1024 pipeline on real data and a frozen released MDLM.

This is a bounded diagnostic, not a main training run or sampler selection.
Only train/dev examples are used for learning and measurement. Held-out files
are audited for integrity and separation, never used to choose settings.
The success marker is validation.json with status=passed, written last.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from chain_crf.backbone import FrozenMDLM, file_sha256
from chain_crf.core import build_candidates, chain_log_partition, chain_marginals, gold_log_prob, sample_chain
from chain_crf.counts import CountBigramHead
from chain_crf.data import BatchStream, atomic_json, atomic_torch_save, canonical_hash, source_document_set
from chain_crf.generation import potentials
from scripts.evaluate_chain_crf import load_head, score_gpt2
from scripts.train_chain_crf import checkpoint_payload, corrupt, make_head, restore, score_batch

VALIDATED_SOURCE_FILES = (
    "chain_crf/backbone.py", "chain_crf/core.py", "chain_crf/heads.py", "chain_crf/data.py",
    "chain_crf/counts.py", "chain_crf/generation.py", "models/dit.py", "configs/model/small.yaml",
    "scripts/train_chain_crf.py", "scripts/build_chain_counts.py", "scripts/evaluate_chain_crf.py",
    "scripts/prepare_released_mdlm_owt.py", "scripts/validate_chain_1024.py",
)


def emit(event, **values):
    print(json.dumps({"event": event, **values}, allow_nan=False), flush=True)


def tensor_state_hash(module):
    result = hashlib.sha256()
    for name, value in module.state_dict().items():
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def run(output, label, arguments):
    emit("command_started", label=label)
    with (output / f"{label}.log").open("w") as handle:
        subprocess.run([sys.executable, *arguments], cwd=ROOT, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)
    emit("command_passed", label=label)


def audit_data(data, output):
    manifest = json.loads((data / "manifest.json").read_text())
    assert manifest["length"] == 1024
    assert manifest["document_quotas"] == {"train": 200000, "dev": 2000, "heldout_eval": 100000}
    assert manifest["selection_report"]["evaluation_population"] == "full-tail"
    for filename, expected in manifest["files"].items():
        assert Path(filename).name == filename
        assert file_sha256(data / filename) == expected, filename
    selected = json.loads((data / "selected_documents.json").read_text())
    start, stop = manifest["source_validation_window"]
    assert [row["source_document_index"] for row in selected["heldout_eval"]] == list(range(start, stop))
    sets = {role: {row["document_id"] for row in rows} for role, rows in selected.items()}
    for first, second in itertools.combinations(sets, 2):
        assert not sets[first] & sets[second], (first, second)
    assert canonical_hash(selected) == manifest["selection_report"]["selection_sha256"]
    report, row_roles = {}, {}
    fixture = output / "reference-data"
    fixture.mkdir()
    for role in ("train", "dev", "heldout_eval"):
        payload = torch.load(data / f"{role}.pt", map_location="cpu", weights_only=True, mmap=True)
        tokens, docs = payload["tokens"], payload["document_ids"]
        assert tokens.dtype == torch.long and tokens.shape == (manifest["splits"][role]["rows"], 1024)
        assert len(tokens) == len(docs) and payload["provenance"]["split"] == role
        assert source_document_set(docs) <= sets[role]
        digest, eos_count = hashlib.sha256(), 0
        for offset in range(0, len(tokens), 4096):
            block = tokens[offset:offset + 4096]
            assert bool(((block >= 0) & (block < 50257)).all())
            assert bool(block[:, [0, -1]].eq(50256).all())
            digest.update(block.numpy().tobytes())
            eos_count += int(block.eq(50256).sum())
            for row in block:
                key = hashlib.sha256(row.numpy().tobytes()).digest()
                assert row_roles.setdefault(key, role) == role, "Cross-role duplicate token row"
        assert digest.hexdigest() == manifest["splits"][role]["tokens_sha256_le_int64"]
        report[role] = {**manifest["splits"][role], "unique_source_documents": len(sets[role]),
                        "eos_tokens_including_boundaries": eos_count}
        if role in ("train", "dev"):
            count = 32 if role == "train" else 4
            atomic_torch_save({"tokens": tokens[:count].clone(), "document_ids": docs[:count],
                "provenance": {"split": role, "purpose": "step3_validation_only",
                    "original_file_sha256": manifest["files"][f"{role}.pt"],
                    "source_row_indices": list(range(count))}}, fixture / f"{role}.pt")
            # Preserve full row/source-span records for the fixed diagnostic examples.
            with (data / f"{role}.jsonl").open() as source, (fixture / f"{role}.jsonl").open("w") as dest:
                for index in range(count):
                    line = next(source)
                    row = json.loads(line)
                    assert row["input_ids"] == tokens[index].tolist()
                    assert row["document_ids"] == docs[index]
                    dest.write(line)
        del payload, tokens
        emit("data_split_audited", split=role, rows=report[role]["rows"])
    atomic_json(report, output / "data-audit.json")
    return fixture, report


@torch.no_grad()
def exact_gpu_check(device):
    rng = torch.Generator(device=device).manual_seed(731)
    unary = torch.randn(1, 3, 3, dtype=torch.float64, device=device, generator=rng)
    edges = torch.randn(1, 2, 3, 3, dtype=torch.float64, device=device, generator=rng)
    states = torch.tensor(list(itertools.product(range(3), repeat=3)), device=device)
    values = unary[0, torch.arange(3, device=device), states].sum(-1)
    values += edges[0, torch.arange(2, device=device), states[:, :-1], states[:, 1:]].sum(-1)
    logz = values.logsumexp(0)
    torch.testing.assert_close(chain_log_partition(unary, edges)[0], logz, rtol=1e-12, atol=1e-12)
    expected = torch.zeros_like(unary)
    probabilities = values.softmax(0)
    for state, probability in zip(states, probabilities):
        expected[0, torch.arange(3, device=device), state] += probability
    torch.testing.assert_close(chain_marginals(unary, edges), expected, rtol=1e-12, atol=1e-12)
    draws = sample_chain(unary.expand(24000, -1, -1), edges.expand(24000, -1, -1, -1), generator=rng)
    # CUDA does not implement matrix-vector multiplication for int64 tensors.
    codes = draws[:, 0] * 9 + draws[:, 1] * 3 + draws[:, 2]
    frequencies = torch.bincount(codes, minlength=27) / 24000
    error = float((frequencies - probabilities).abs().max())
    assert error < .015
    return {"enumerated_states": 27, "draws": 24000, "max_joint_frequency_error": error}


@torch.no_grad()
def neutral_checks(backbone, train, output, device):
    clean = train[:1].to(device)
    corrupted, active, times = corrupt(clean.cpu(), backbone.mask_id, torch.Generator().manual_seed(103))
    corrupted, active, times = [value.to(device) for value in (corrupted, active, times)]
    pred = backbone(corrupted, times)
    packet = build_candidates(pred["log_probs"], corrupted, backbone.mask_id, 64, gold=clean)
    expected = (packet.normalized_log_probs.gather(-1, clean[..., None]).squeeze(-1) * active).sum()
    heads = {mode: make_head(mode, backbone.vocab_size, backbone.hidden_size,
              64 if mode == "independent" else 32).to(device)
             for mode in ("global", "contextual", "independent")}
    # Fresh contextual initialization deliberately has small nonzero pair scores.
    # Construct its identity control explicitly; leave actual training initialization intact.
    heads["contextual"].load_global(heads["global"])
    heads["count"] = CountBigramHead(backbone.vocab_size, strength=0.).fit(train[:8]).to(device)
    reports = {}
    for mode, head in heads.items():
        unary, edges = potentials(packet, head, mode, pred["hidden"], times)
        delta = head(packet.candidate_ids, pred["hidden"], times) * active[..., None] if mode == "independent" else None
        actual = gold_log_prob(packet, edges, delta).sum()
        error = float((actual - expected).abs() / active.sum())
        assert error < 2e-5, (mode, error)
        marginals = chain_marginals(unary, edges)
        marginal_error = float((marginals - packet.unary.softmax(-1)).abs().max())
        assert marginal_error < 2e-4, (mode, marginal_error)
        reports[mode] = {"nll_error_per_masked_token": error, "max_candidate_marginal_error": marginal_error}
    positions = torch.arange(0, 1024, 32, device=device)
    atomic_torch_save({"clean": clean.cpu(), "corrupted": corrupted.cpu(), "active": active.cpu(),
        "times": times.cpu(), "candidate_ids": packet.candidate_ids.cpu(), "unary": packet.unary.cpu(),
        "hidden": pred["hidden"].cpu(), "log_probability_positions": positions.cpu(),
        "log_probabilities": pred["log_probs"][:, positions].cpu()}, output / "neutral-reference.pt")
    atomic_json(reports, output / "neutral-checks.json")
    return reports


def profile_heads(backbone, train, output, device):
    reports = {}
    original = tensor_state_hash(backbone)
    assert all(not p.requires_grad for p in backbone.parameters()) and not backbone.training
    for mode in ("global", "contextual", "independent"):
        torch.manual_seed(1)
        rank = 64 if mode == "independent" else 32
        head = make_head(mode, backbone.vocab_size, backbone.hidden_size, rank).to(device)
        initial = {name: value.detach().clone() for name, value in head.state_dict().items()}
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.)
        assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in head.parameters()}
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        stream, mask_rng = BatchStream(train, 4, 1001), torch.Generator().manual_seed(2001)
        config = {"mode": mode, "vocab_size": backbone.vocab_size, "hidden_size": backbone.hidden_size,
                  "rank": rank, "mlp_size": 128, "k": 64, "length": 1024, "batch_size": 4,
                  "backbone_batch_size": 1, "seed": 1, "purpose": "validation_only"}
        identity = {"config": config, "backbone": backbone.provenance,
                    "train_sha256": file_sha256(output / "reference-data/train.pt")}
        torch.cuda.reset_peak_memory_stats(device)
        records = []

        def update():
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            loss, metrics = score_batch(backbone, head, mode, stream.next(), k=64, device=device,
                                       generator=mask_rng, backbone_batch_size=1)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize(device)
            metrics.update(total_step_seconds=time.perf_counter() - start, grad_norm=float(grad_norm))
            assert all(p.grad is None for p in backbone.parameters())
            return metrics

        for step in range(1, 4):
            records.append(update())
            emit("profile_update", mode=mode, step=step, seconds=records[-1]["total_step_seconds"])
        checkpoint = output / f"{mode}-resume-reference.pt"
        atomic_torch_save(checkpoint_payload(head, optimizer, scheduler, stream, mask_rng, 3,
                          identity, None, config), checkpoint)
        expected_metrics = update()
        expected = {name: value.detach().clone() for name, value in head.state_dict().items()}
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert restore(payload, head, optimizer, scheduler, stream, mask_rng, identity)[0] == 3
        replay_metrics = update()
        assert abs(replay_metrics["nll_per_masked_token"] - expected_metrics["nll_per_masked_token"]) < 1e-6
        maximum_resume_error = 0.
        for name, value in head.state_dict().items():
            maximum_resume_error = max(maximum_resume_error, float((value - expected[name]).abs().max()))
            torch.testing.assert_close(value, expected[name], rtol=1e-6, atol=1e-7)
        records.append(expected_metrics)
        for step in range(5, 9):
            records.append(update())
            emit("profile_update", mode=mode, step=step, seconds=records[-1]["total_step_seconds"])
        checkpoint = output / f"{mode}-validation-head.pt"
        atomic_torch_save(checkpoint_payload(head, optimizer, scheduler, stream, mask_rng, 8,
                          identity, None, config), checkpoint)
        loaded, _ = load_head(SimpleNamespace(mode=mode, head=checkpoint, device=device), backbone)
        for name, value in head.state_dict().items():
            torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
        changed = [name for name, value in head.state_dict().items() if not torch.equal(value, initial[name])]
        assert changed and tensor_state_hash(backbone) == original
        reports[mode] = {"parameter_count": sum(p.numel() for p in head.parameters()), "config": config,
            "changed_head_parameters": changed, "backbone_unchanged": True,
            "resume_max_parameter_error": maximum_resume_error,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "median_seconds_per_update": statistics.median(row["total_step_seconds"] for row in records[1:]),
            "updates": records}
        atomic_json(reports, output / "training-profile.json")
        del head, loaded, optimizer, scheduler, payload, initial, expected
        gc.collect()
        torch.cuda.empty_cache()
    return reports


def generation_checks(args, output):
    all_records, reports = [], {}
    for mode in ("backbone", "count", "global", "contextual", "independent"):
        target = output / f"generation-{mode}"
        command = ["scripts/evaluate_chain_crf.py", "--output", str(target), "--mode", mode,
            "--backbone-checkpoint", str(args.checkpoint), "--cache-dir", str(args.cache_dir),
            "--device", args.device, "--length", "1024", "--steps", "16", "--samples", "2",
            "--batch-size", "2", "--k", "64", "--warmup", "0", "--sample-offset", "90000000"]
        if mode == "count":
            command += ["--counts", str(output / "validation-counts.pt"), "--strength", ".1"]
        elif mode != "backbone":
            command += ["--head", str(output / f"{mode}-validation-head.pt")]
        run(output, f"generate-{mode}", command)
        path = target / "samples.jsonl"
        expected = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(expected) == 2
        assert [row["sample_id"] for row in expected] == [0, 1]
        assert [row["draw_id"] for row in expected] == [90000000, 90000001]
        for row in expected:
            assert len(row["token_ids"]) == 1024 and all(0 <= t < 50257 for t in row["token_ids"])
            assert row["generated_length"] == 1024 and row["prefix_length"] == 0
        # Deliberately interrupt only this disposable validation output at a
        # partial-batch boundary. Retain the complete reference before replay.
        (target / "samples.reference.jsonl").write_bytes(path.read_bytes())
        path.write_text(json.dumps(expected[0]) + "\n")
        run(output, f"resume-generation-{mode}", command + ["--resume"])
        actual = [json.loads(line) for line in path.read_text().splitlines()]
        for first, second in zip(actual, expected):
            for key in ("sample_id", "draw_id", "token_ids", "prefix_length", "batch_id", "batch_size"):
                assert first[key] == second[key], (mode, key)
        assert len(actual) == len(expected)
        reports[mode] = {"samples": len(actual), "length": 1024, "partial_batch_resume_identical": True,
                         "samples_sha256": file_sha256(path)}
        for row in actual:
            all_records.append({**row, "method": mode, "sample_id": len(all_records)})
    atomic_json(reports, output / "generation-checks.json")
    score_hash = canonical_hash(all_records)
    first = score_gpt2(all_records, args.device)
    gc.collect()
    torch.cuda.empty_cache()
    second = score_gpt2(all_records, args.device)
    assert canonical_hash(all_records) == score_hash
    assert first["scored_tokens"] == second["scored_tokens"] == 10 * 1023
    assert abs(first["mean_nll"] - second["mean_nll"]) < 1e-7
    for left, right in zip(first["samples"], second["samples"]):
        assert left["sample_id"] == right["sample_id"] and abs(left["nll"] - right["nll"]) < 1e-5
    atomic_json(first, output / "scorer-first.json")
    atomic_json(second, output / "scorer-repeat.json")
    return {"generation": reports, "rescoring_mean_nll_difference": abs(first["mean_nll"] - second["mean_nll"]),
            "scored_tokens": first["scored_tokens"], "scoring_policy": "current raw-token-ID path; final paper protocol deferred"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-gpu", default="NVIDIA RTX 6000 Ada Generation")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    try:
        source_hashes = {name: file_sha256(ROOT / name) for name in VALIDATED_SOURCE_FILES}
        torch.set_num_threads(4)
        assert torch.cuda.is_available() and args.device.startswith("cuda")
        gpu = torch.cuda.get_device_name(args.device)
        assert gpu == args.expected_gpu, (gpu, args.expected_gpu)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        tests = ["test_chain_crf_core.py", "test_chain_crf_heads.py", "test_chain_generation.py",
                 "test_chain_evaluation.py", "test_chain_training.py", "test_chain_packed_data.py",
                 "test_chain_segments.py"]
        run(args.output, "unit-tests", ["-m", "pytest", "-q", *["tests/" + name for name in tests]])
        fixture, data_report = audit_data(args.data, args.output)
        train = torch.load(fixture / "train.pt", weights_only=True)["tokens"]
        exact = exact_gpu_check(args.device)
        backbone = FrozenMDLM(args.checkpoint, device=args.device, cache_dir=args.cache_dir)
        neutral = neutral_checks(backbone, train, args.output, args.device)
        emit("neutral_controls_passed")
        profile = profile_heads(backbone, train, args.output, args.device)
        provenance = backbone.provenance
        CountBigramHead(backbone.vocab_size, strength=.1).fit(train[:8]).save(args.output / "validation-counts.pt")
        del backbone, train
        gc.collect()
        torch.cuda.empty_cache()
        common = ["scripts/train_chain_crf.py", "--data", str(fixture), "--output", str(args.output / "cli-train"),
            "--mode", "global", "--checkpoint", str(args.checkpoint), "--cache-dir", str(args.cache_dir),
            "--device", args.device, "--length", "1024", "--k", "64", "--rank", "32", "--steps", "2",
            "--batch-size", "4", "--backbone-batch-size", "1", "--max-dev-examples", "4",
            "--eval-every", "2", "--save-every", "1"]
        run(args.output, "real-training-cli", common)
        resumed = list(common)
        resumed[resumed.index("--steps") + 1] = "3"
        run(args.output, "real-training-cli-resume", resumed + ["--resume", str(args.output / "cli-train/last.pt")])
        assert torch.load(args.output / "cli-train/last.pt", weights_only=True)["step"] == 3
        generation = generation_checks(args, args.output)
        assert source_hashes == {name: file_sha256(ROOT / name) for name in VALIDATED_SOURCE_FILES}
        result = {"status": "passed", "gpu": gpu, "backbone": provenance,
            "source_sha256": source_hashes,
            "data_manifest_sha256": file_sha256(args.data / "manifest.json"),
            "validation_script_sha256": file_sha256(Path(__file__)), "data": data_report,
            "exact_gpu": exact, "neutral_controls": neutral, "training_profiles": profile, **generation,
            "limits": ["Eight short profile updates per head do not establish convergence.",
                       "Current sampler/scorer paths tested; final sampling protocol remains undecided.",
                       "Historical MDLM source ordering is not independently authenticated."]}
        atomic_json(result, args.output / "validation.json")
        emit("validation_passed", output=str(args.output))
    except Exception as error:
        atomic_json({"status": "failed", "error_type": type(error).__name__, "error": str(error)},
                    args.output / "failure.json")
        raise


if __name__ == "__main__":
    main()
