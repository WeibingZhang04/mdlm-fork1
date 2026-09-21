#!/usr/bin/env python3
"""Summarize matched per-seed reference scores without loading a model."""
import argparse
import csv
import json
import math
from pathlib import Path

ARMS = ("static_static", "fixed_dynamic", "dynamic_fixed", "dynamic_dynamic")
PAIR_FIELDS = ("pair_seed", "pair_key", "batch_seed", "initial_token_ids",
               "active_mask", "sampling_mode", "requested_nfe_budget",
               "global_pairing_digest")
REFERENCE_FIELDS = ("model_name_or_path", "revision", "sequence_policy")

def summarize(records):
    scores = [r["reference_lm"] for r in records]
    if not scores or any(s["token_count"] <= 0 or
                         s["mean_nll_nats"] is None or
                         not math.isfinite(s["mean_nll_nats"]) for s in scores):
        raise ValueError("Missing or invalid per-seed GPT-2 scores")
    tokens = sum(s["token_count"] for s in scores)
    nll = math.fsum(s["mean_nll_nats"] * s["token_count"] for s in scores) / tokens
    return {"ppl": math.exp(min(nll, 80)), "nll": nll, "tokens": tokens,
            "mean_scored_tokens": tokens / len(scores),
            "mean_repeat4": math.fsum(r["metrics"]["repetition_rate"]["4"]
                                     for r in records) / len(records)}

def compare(left, right):
    left = sorted(left, key=lambda r: r["pair_seed"])
    right = sorted(right, key=lambda r: r["pair_seed"])
    if len(left) != len(right) or len({r["pair_seed"] for r in left}) != len(left):
        raise ValueError("Missing or duplicate seed pairs")
    for a, b in zip(left, right):
        if any(a[k] != b[k] for k in PAIR_FIELDS):
            raise ValueError("Generation inputs are not paired")
        if any(a["reference_lm"][k] != b["reference_lm"][k] for k in REFERENCE_FIELDS):
            raise ValueError("Reference scoring policies differ")
        if a["timing"]["unresolved_mask_tokens"] or b["timing"]["unresolved_mask_tokens"]:
            raise ValueError("Unresolved generated masks")
    before, after = summarize(left), summarize(right)
    differences = [b["reference_lm"]["mean_nll_nats"] -
                   a["reference_lm"]["mean_nll_nats"] for a, b in zip(left, right)]
    return before, after, differences, left, right

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--num-seeds", type=int, default=5)
    args = parser.parse_args()
    if args.num_seeds < 2:
        parser.error("--num-seeds must be at least two")
    root = args.root
    result, per_seed = {}, []
    lines = [f"# CCF 6000 versus 7000: {args.num_seeds} paired seeds", "",
             "GPT-2-large; lower PPL is better. Repetition uses full generated sequences.",
             "GPT-2 scoring ends at the first non-leading EOS. This is a small exploratory comparison.", "",
             "| Arm | 6000 PPL | 7000 PPL | 7000 wins | Mean scored tokens 6k / 7k | Repeat-4 % 6k / 7k |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        records = []
        for step in (6000, 7000):
            path = root / f"step{step}" / arm / "generation" / "samples.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if sorted(r["pair_seed"] for r in rows) != list(range(91001, 91001 + args.num_seeds)):
                raise ValueError(f"{arm}/{step}: expected {args.num_seeds} seeds starting at 91001")
            records.append(rows)
        before, after, delta, left, right = compare(*records)
        wins = sum(d < 0 for d in delta)
        # Seed 91001 was already used to select this checkpoint comparison.
        fresh_before, fresh_after, fresh_delta, _, _ = compare(
            [r for r in left if r["pair_seed"] != 91001],
            [r for r in right if r["pair_seed"] != 91001])
        result[arm] = {"6000": before, "7000": after, "wins_7000": wins,
                       "mean_paired_nll_delta_7000_minus_6000": math.fsum(delta) / len(delta),
                       "new_seeds_only": {"6000": fresh_before, "7000": fresh_after,
                                          "wins_7000": sum(d < 0 for d in fresh_delta)}}
        lines.append(f"| {arm} | {before['ppl']:.3f} | {after['ppl']:.3f} | {wins}/{args.num_seeds} | "
                     f"{before['mean_scored_tokens']:.1f} / {after['mean_scored_tokens']:.1f} | "
                     f"{100*before['mean_repeat4']:.2f} / {100*after['mean_repeat4']:.2f} |")
        for a, b, d in zip(left, right, delta):
            per_seed.append({"arm": arm, "seed": a["pair_seed"],
                             "ppl_6000": a["reference_lm"]["perplexity"],
                             "ppl_7000": b["reference_lm"]["perplexity"],
                             "nll_delta_7000_minus_6000": d,
                             "tokens_6000": a["reference_lm"]["token_count"],
                             "tokens_7000": b["reference_lm"]["token_count"],
                             "repeat4_6000": a["metrics"]["repetition_rate"]["4"],
                             "repeat4_7000": b["metrics"]["repetition_rate"]["4"]})
    lines += ["", f"## New seeds only ({args.num_seeds - 1})", "",
              "Seed 91001 informed checkpoint selection; this table excludes it.", "",
              "| Arm | 6000 PPL | 7000 PPL | 7000 wins |",
              "| --- | ---: | ---: | ---: |"]
    for arm in ARMS:
        new = result[arm]["new_seeds_only"]
        lines.append(f"| {arm} | {new['6000']['ppl']:.3f} | {new['7000']['ppl']:.3f} | {new['wins_7000']}/{args.num_seeds - 1} |")
    report = "\n".join(lines) + "\n"
    # Exclusive creation: a repeated summary must not silently replace results.
    with (root / "paired-comparison.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    with (root / "paired-seed-scores.tsv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(per_seed)
    with (root / "paired-comparison.md").open("x") as f:
        f.write(report)
    print(report)

if __name__ == "__main__":
    main()
