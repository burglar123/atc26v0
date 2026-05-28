#!/usr/bin/env python3
"""Inspect depth-specific rolling fields for bounded-N refactor planning.

This script is read-only. It scans source/docs files and reports where
hard-coded rolling depth references are concentrated.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_SCAN_PATHS = (
    "benchmark",
    "nano_pearl",
    "docs/phase1h_project_book.md",
    "docs/phase1h_bounded_n_refactor_design.md",
)

TEXT_SUFFIXES = {".md", ".py", ".toml", ".json", ".yaml", ".yml", ".txt"}

DEPTH_REF_RE = re.compile(r"(?:rolling_)?depth(?:[0-9]+|_gt[0-9]+)")
ROLLING_FIELD_RE = re.compile(
    r"\b(?:rolling_depth[0-9]+|rolling_depth_gt[0-9]+|"
    r"continuous_eager|eager_committed|max_rolling_continuous_depth)"
    r"[A-Za-z0-9_]*\b"
)
DEPTH_TOKEN_RE = re.compile(r"depth(?:[0-9]+|_gt[0-9]+)")


def iter_files(root: Path, scan_paths: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for rel in scan_paths:
        path = root / rel
        if path.is_file():
            if path.suffix in TEXT_SUFFIXES:
                files.append(path)
            continue
        if not path.is_dir():
            continue
        for candidate in path.rglob("*"):
            if not candidate.is_file():
                continue
            if candidate.suffix not in TEXT_SUFFIXES:
                continue
            if any(part in {".git", "__pycache__", ".pytest_cache"} for part in candidate.parts):
                continue
            files.append(candidate)
    return sorted(set(files))


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def field_group(token: str) -> str:
    parts = token.split("_")
    if token.startswith("rolling_depth"):
        if len(parts) >= 4 and parts[2] in {"real", "child", "parent", "commit"}:
            return "_".join(parts[:4])
        if len(parts) >= 3:
            return "_".join(parts[:3])
    if token.startswith("continuous_eager"):
        return "continuous_eager"
    if token.startswith("eager_committed"):
        return "eager_committed"
    if token.startswith("max_rolling_continuous_depth"):
        return "max_rolling_continuous_depth"
    return token


def normalized_suffix(token: str) -> tuple[str, str] | None:
    match = DEPTH_TOKEN_RE.search(token)
    if not match:
        return None
    depth = match.group(0)
    suffix = token[: match.start()] + "depthN" + token[match.end() :]
    return depth, suffix


def canonical_depth_ref(token: str) -> str:
    return token.removeprefix("rolling_")


def inspect(root: Path, scan_paths: Iterable[str]) -> dict[str, object]:
    files = iter_files(root, scan_paths)
    depth_refs_by_file: dict[str, Counter[str]] = {}
    field_counts: Counter[str] = Counter()
    field_groups: Counter[str] = Counter()
    suffix_depths: dict[str, set[str]] = defaultdict(set)
    files_by_depth: dict[str, list[str]] = defaultdict(list)

    for path in files:
        text = read_text(path)
        if not text:
            continue
        rel = str(path.relative_to(root))
        depth_refs = Counter(canonical_depth_ref(match.group(0)) for match in DEPTH_REF_RE.finditer(text))
        if depth_refs:
            depth_refs_by_file[rel] = depth_refs
            for depth_ref in sorted(depth_refs):
                files_by_depth[depth_ref].append(rel)

        for match in ROLLING_FIELD_RE.finditer(text):
            token = match.group(0)
            field_counts[token] += 1
            field_groups[field_group(token)] += 1
            suffix = normalized_suffix(token)
            if suffix is not None:
                depth, normalized = suffix
                suffix_depths[normalized].add(depth)

    repeated_suffixes = {
        suffix: sorted(depths)
        for suffix, depths in sorted(suffix_depths.items())
        if len(depths) > 1
    }

    return {
        "root": str(root),
        "files_scanned": len(files),
        "files_with_depth_refs": {
            name: dict(counter)
            for name, counter in sorted(depth_refs_by_file.items())
        },
        "files_by_depth_ref": {
            depth: sorted(paths)
            for depth, paths in sorted(files_by_depth.items())
        },
        "top_field_tokens": field_counts.most_common(80),
        "field_groups": field_groups.most_common(),
        "repeated_depth_suffixes": repeated_suffixes,
    }


def print_text_report(report: dict[str, object], limit: int) -> None:
    print("Bounded-N refactor readiness inspection")
    print(f"root: {report['root']}")
    print(f"files scanned: {report['files_scanned']}")
    print(f"files with depth refs: {len(report['files_with_depth_refs'])}")
    print()

    print("Files by depth reference:")
    files_by_depth = report["files_by_depth_ref"]
    assert isinstance(files_by_depth, dict)
    for depth, paths in files_by_depth.items():
        assert isinstance(paths, list)
        shown = ", ".join(paths[:limit])
        suffix = "" if len(paths) <= limit else f" ... (+{len(paths) - limit})"
        print(f"  {depth}: {len(paths)} files")
        if shown:
            print(f"    {shown}{suffix}")
    print()

    print("Top field groups:")
    for group, count in report["field_groups"][:limit]:
        print(f"  {group}: {count}")
    print()

    print("Repeated depth-normalized field suffixes:")
    repeated = report["repeated_depth_suffixes"]
    assert isinstance(repeated, dict)
    for suffix, depths in list(repeated.items())[:limit]:
        print(f"  {suffix}: {', '.join(depths)}")
    if len(repeated) > limit:
        print(f"  ... (+{len(repeated) - limit})")
    print()

    print("Top concrete field tokens:")
    for token, count in report["top_field_tokens"][:limit]:
        print(f"  {token}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect hard-coded depth-specific fields for bounded-N refactor planning."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root to scan",
    )
    parser.add_argument(
        "--path",
        action="append",
        dest="paths",
        help="relative path to scan; may be provided multiple times",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--limit", type=int, default=20, help="max rows per text section")
    args = parser.parse_args()

    root = args.root.resolve()
    scan_paths = args.paths if args.paths else DEFAULT_SCAN_PATHS
    report = inspect(root, scan_paths)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(report, max(1, args.limit))


if __name__ == "__main__":
    main()
