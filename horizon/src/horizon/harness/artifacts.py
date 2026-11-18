"""Atomic JSON/NPZ artifacts with hashes and no pickle-based deserialization."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from ..policy.dataset import SampleBlock
from ..rsi.loop import load_params

_ARRAY_FIELDS = ("x_hier", "x_flat", "y", "k", "stage")


def _json_bytes(obj, *, pretty: bool = False) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      indent=2 if pretty else None,
                      separators=None if pretty else (",", ":")).encode("utf-8")


def canonical_hash(obj) -> str:
    """SHA-256 of canonical UTF-8 JSON; rejects NaN/Inf and non-JSON values."""
    return hashlib.sha256(_json_bytes(obj)).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
                                         mode="wb", delete=False) as stream:
            tmp = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def write_json(path: str | Path, obj) -> None:
    data = _json_bytes(obj, pretty=True) + b"\n"  # validate before replacing an existing file
    _atomic_write(Path(path), lambda stream: stream.write(data))


def _safe_array(value, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        raise ValueError(f"{name} must be a NumPy array without object dtype")
    return value


def _check_block_arrays(arrays: dict[str, np.ndarray], n: int) -> None:
    for name, array in arrays.items():
        _safe_array(array, name)
        dimensions = 1 if name in ("k", "stage") else 2
        if array.ndim != dimensions or array.shape[0] != n:
            raise ValueError(f"Inconsistent SampleBlock shape for {name}: {array.shape}, expected n={n}")


def save_blocks(path: str | Path, blocks: list[SampleBlock]) -> dict:
    """Save numeric arrays and JSON metadata; return a manifest with both hashes.

    ``SampleBlock.meta`` must be JSON-compatible. ``blocks`` in the returned
    manifest is the block count, while ``n`` counts supervised transitions.
    """
    path = Path(path).resolve()
    metadata_path = path.with_name(path.name + ".meta.json")
    arrays, items = {}, []
    for index, block in enumerate(blocks):
        values = {field: getattr(block, field) for field in _ARRAY_FIELDS}
        _check_block_arrays(values, block.n)
        if not isinstance(block.meta, dict):
            raise ValueError("SampleBlock.meta must be a JSON object")
        arrays.update({f"block{index}_{field}": value for field, value in values.items()})
        items.append({"key": block.key, "family": block.family, "H": block.H,
                      "provenance": block.provenance, "meta": block.meta, "n": block.n})
    n = sum(item["n"] for item in items)
    metadata = {"version": 1, "n": n, "blocks": items}
    metadata_bytes = _json_bytes(metadata, pretty=True) + b"\n"
    _atomic_write(path, lambda stream: np.savez_compressed(stream, **arrays))
    _atomic_write(metadata_path, lambda stream: stream.write(metadata_bytes))
    return {"path": str(path), "sha256": file_hash(path), "metadata_path": str(metadata_path),
            "metadata_sha256": file_hash(metadata_path), "n": n, "blocks": len(items)}


def _verified_path(record: dict, path_key: str = "path", hash_key: str = "sha256") -> Path:
    path = Path(record[path_key])
    if file_hash(path) != record[hash_key]:
        raise ValueError(f"Artifact hash mismatch: {path}")
    return path


def load_blocks(record: dict) -> list[SampleBlock]:
    path = _verified_path(record)
    metadata_path = _verified_path(record, "metadata_path", "metadata_sha256")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _json_bytes(metadata)  # also reject non-finite JSON accepted by Python's permissive reader
    if metadata.get("version") != 1 or not isinstance(metadata.get("blocks"), list):
        raise ValueError("Unsupported block metadata schema")
    items = metadata["blocks"]
    if len(items) != record["blocks"] or metadata["n"] != record["n"]:
        raise ValueError("Block counts disagree with artifact manifest")
    blocks = []
    with np.load(path, allow_pickle=False) as archive:
        expected = {f"block{i}_{field}" for i in range(len(items)) for field in _ARRAY_FIELDS}
        if set(archive.files) != expected:
            raise ValueError("NPZ arrays disagree with block metadata")
        for index, item in enumerate(items):
            arrays = {field: archive[f"block{index}_{field}"].copy() for field in _ARRAY_FIELDS}
            _check_block_arrays(arrays, item["n"])
            if not isinstance(item["meta"], dict):
                raise ValueError("SampleBlock.meta must be a JSON object")
            blocks.append(SampleBlock(key=item["key"], family=item["family"], H=item["H"],
                                      provenance=item["provenance"], meta=item["meta"], **arrays))
    if sum(block.n for block in blocks) != record["n"]:
        raise ValueError("Transition count disagrees with artifact manifest")
    return blocks


def save_policy(path: str | Path, params: dict) -> dict:
    """Save the existing NumPy policy format, including arch and optional n_layers."""
    path = Path(path).resolve()
    arrays = {name: _safe_array(value, name) for name, value in params.items()
              if name not in ("arch", "n_layers") and isinstance(value, np.ndarray)}
    unsupported = set(params) - set(arrays) - {"arch", "n_layers"}
    if unsupported:
        raise ValueError(f"Unsupported policy values: {sorted(unsupported)}")
    arrays["arch"] = np.array(str(params.get("arch", "mlp")))
    if "n_layers" in params:
        arrays["n_layers"] = np.array(int(params["n_layers"]))
    _atomic_write(path, lambda stream: np.savez_compressed(stream, **arrays))
    return {"path": str(path), "sha256": file_hash(path)}


def load_policy(record: dict) -> dict:
    return load_params(_verified_path(record))
