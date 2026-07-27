#!/usr/bin/env python3
"""Download and verify the source data for Turner et al. (2018)."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path("tracks/ed/turner-2018-data.json")
DEFAULT_OUTPUT = Path(".external/official-data/turner-2018")
USER_AGENT = "quantum-harness/turner-2018-data"


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: str | Path, entry: dict[str, Any]) -> bool:
    path = Path(path)
    return (
        path.is_file()
        and path.stat().st_size == entry["size"]
        and sha256_file(path) == entry["sha256"]
    )


def download_file(entry: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / entry["name"]
    if verify_file(destination, entry):
        print(f"verified {destination}")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(entry["url"], headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            with open(partial, "wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
        if not verify_file(partial, entry):
            raise RuntimeError(f"checksum mismatch: {entry['name']}")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)

    print(f"downloaded {destination}")
    return destination


def sync_dataset(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> list[Path]:
    manifest = load_manifest(manifest_path)
    return [download_file(entry, output_dir) for entry in manifest["files"]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and verify the Turner et al. 2018 source dataset."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check local files without downloading missing or corrupt files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    if args.verify_only:
        failures = [
            entry["name"]
            for entry in manifest["files"]
            if not verify_file(args.output_dir / entry["name"], entry)
        ]
        if failures:
            print("invalid or missing: " + ", ".join(failures))
            return 1
        print(f"verified {len(manifest['files'])} files in {args.output_dir}")
        return 0

    sync_dataset(args.manifest, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
