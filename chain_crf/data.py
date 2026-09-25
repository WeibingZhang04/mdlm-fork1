"""Document-disjoint, hash-assigned data and safe training checkpoint helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def document_split(document_id: str, salt="chain-crf-20260925-v1") -> str:
    bucket = int(hashlib.sha256(f"{salt}:{document_id}".encode()).hexdigest()[:16], 16) % 10000
    return "dev" if bucket < 100 else "test" if bucket < 200 else "train"


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_token_data(path, *, length, vocab_size, mask_id, max_examples=None):
    """Read a prepared .pt or document-identified .jsonl; never invent IDs."""
    path = Path(path)
    if path.suffix == ".jsonl":
        rows, docs = [], []
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                ids = row.get("input_ids")
                doc = row.get("document_id", row.get("source_document_sha256"))
                if not isinstance(doc, str) or not doc:
                    raise ValueError("Every training row needs a source document identity")
                if not isinstance(ids, list) or any(type(v) is not int for v in ids):
                    raise ValueError("Every input_ids row must be a list of integer token IDs")
                if len(ids) < length:
                    continue
                rows.append(ids[:length]); docs.append(doc)
                if max_examples and len(rows) >= max_examples:
                    break
        if not rows:
            raise ValueError(f"No sufficiently long rows in {path}")
        tokens, source = torch.tensor(rows, dtype=torch.long), {"format": "jsonl"}
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        tokens, docs = payload["tokens"], payload["document_ids"]
        source = payload.get("provenance", {})
        if tokens.ndim != 2 or tokens.shape[1] < length:
            raise ValueError("Prepared data is shorter than requested sequence length")
        tokens, docs = tokens[:max_examples, :length].contiguous(), docs[:max_examples]
    if tokens.dtype != torch.long or len(tokens) != len(docs) or not len(tokens):
        raise ValueError("Invalid tokens/document_ids")
    if bool(((tokens < 0) | (tokens >= vocab_size) | (tokens == mask_id)).any()):
        raise ValueError("Clean training data contains invalid tokens or absorbing masks")
    if any(not isinstance(doc, str) or not doc for doc in docs):
        raise ValueError("Invalid document identity")
    return tokens, docs, source


def assert_disjoint(train, train_docs, dev, dev_docs):
    if set(train_docs) & set(dev_docs):
        raise ValueError("Training/development document overlap")
    train_hash = {hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in train.cpu()}
    if any(hashlib.sha256(row.numpy().tobytes()).hexdigest() in train_hash for row in dev.cpu()):
        raise ValueError("Training/development token-sequence duplication")


class BatchStream:
    """Shuffled epochs with serializable position and RNG, including wraparound."""

    def __init__(self, tokens, batch_size, seed):
        if len(tokens) < 1 or batch_size < 1:
            raise ValueError("Nonempty tokens and positive batch size required")
        self.tokens, self.batch_size = tokens, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(tokens), generator=self.generator)
        self.cursor, self.epoch = 0, 0

    def next(self):
        indices = []
        remaining = self.batch_size
        while remaining:
            if self.cursor == len(self.order):
                self.order = torch.randperm(len(self.tokens), generator=self.generator)
                self.cursor = 0
                self.epoch += 1
            n = min(remaining, len(self.order) - self.cursor)
            indices.append(self.order[self.cursor:self.cursor+n])
            self.cursor += n
            remaining -= n
        return self.tokens[torch.cat(indices)]

    def state_dict(self):
        return {"rng": self.generator.get_state(), "order": self.order,
                "cursor": self.cursor, "epoch": self.epoch}

    def load_state_dict(self, state):
        if sorted(state["order"].tolist()) != list(range(len(self.tokens))):
            raise ValueError("Resume data permutation does not match training data")
        if not 0 <= state["cursor"] <= len(self.tokens):
            raise ValueError("Invalid resume cursor")
        self.generator.set_state(state["rng"].cpu())
        self.order, self.cursor, self.epoch = state["order"].cpu(), state["cursor"], state["epoch"]
