#!/usr/bin/env python3
"""
Split a zip file into N chunks and write reconstruction metadata JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path: Path, buf_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split_zip(input_zip: Path, num_chunks: int, out_dir: Path, json_name: str) -> Path:
    if num_chunks <= 0:
        raise ValueError("num_chunks must be > 0")
    if not input_zip.is_file():
        raise FileNotFoundError(f"input zip not found: {input_zip}")
    if input_zip.suffix.lower() != ".zip":
        raise ValueError(f"input must be a .zip file: {input_zip}")

    out_dir.mkdir(parents=True, exist_ok=True)

    original_size = input_zip.stat().st_size
    chunk_size = math.ceil(original_size / num_chunks)
    original_hash = sha256_file(input_zip)

    base = input_zip.name
    chunks: list[dict[str, object]] = []

    with input_zip.open("rb") as src:
        for idx in range(num_chunks):
            start = idx * chunk_size
            if start >= original_size:
                break
            to_read = min(chunk_size, original_size - start)
            data = src.read(to_read)
            if len(data) != to_read:
                raise IOError("unexpected EOF while reading input")

            chunk_name = f"{base}.part{idx:04d}"
            chunk_path = out_dir / chunk_name
            with chunk_path.open("wb") as c:
                c.write(data)

            chunks.append(
                {
                    "index": idx,
                    "file": chunk_name,
                    "size": len(data),
                    "offset": start,
                    "sha256": sha256_bytes(data),
                }
            )

    manifest = {
        "version": 1,
        "original_file_name": base,
        "original_size": original_size,
        "original_sha256": original_hash,
        "num_chunks_requested": num_chunks,
        "num_chunks_written": len(chunks),
        "chunks": chunks,
    }

    manifest_path = out_dir / json_name
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split a .zip file into N chunks and create a manifest JSON."
    )
    parser.add_argument("input_zip", help="Path to source .zip file")
    parser.add_argument(
        "-n",
        "--num-chunks",
        type=int,
        required=True,
        help="Number of chunks to create",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Directory where chunks + manifest are written",
    )
    parser.add_argument(
        "--manifest",
        default="master.json",
        help="Manifest JSON filename (default: master.json)",
    )
    args = parser.parse_args()

    input_zip = Path(args.input_zip).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = split_zip(input_zip, args.num_chunks, out_dir, args.manifest)
    print(f"wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
