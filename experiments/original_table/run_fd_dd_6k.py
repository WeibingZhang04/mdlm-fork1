#!/usr/bin/env python3
"""Run Basic FD/DD through 6k directly from this checkout."""
import json
from pathlib import Path

import run

original_protocol = run.protocol
original_prepare = run.prepare


def protocol(study):
    cfg = original_protocol(study)
    cfg["arms"] = [a.copy() for a in cfg["arms"]
                   if a["id"] in {"basic_fd", "basic_dd"}]
    for arm in cfg["arms"]:
        arm["phases"] = ["basic_1000", "basic_3000", "basic_6000"]
    return cfg


def prepare(options):
    original_prepare(options)
    study = options.study.resolve()
    cells = run.read(study / "confirmation-cells.json")
    targets = []
    for arm in ("basic_fd", "basic_dd"):
        for steps in (8, 16, 32):
            cell = next(x.copy() for x in cells
                        if x["arm_id"] == arm and x["sampling_steps"] == steps)
            cell.update(step=6000, num_samples=100, base_seed=100001)
            targets.append(cell)
    # This profile schedules only these six confirmation cells, no pilot cells.
    (study / "confirmation-cells.json").write_text(json.dumps(targets, indent=2) + "\n")
    (study / "pilot-cells.json").write_text("[]\n")
    state = run.read(study / "study.json")
    state["profile"] = "basic_fd_dd_6k"
    (study / "study.json").write_text(json.dumps(state, indent=2) + "\n")
    print("FD/DD 6k profile: 2 training tasks, 0 pilot cells, 6 confirmation cells.")


if __name__ == "__main__":
    run.protocol = protocol
    run.prepare = prepare
    run.main()
