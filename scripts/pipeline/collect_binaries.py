#!/usr/bin/env python3
"""
Collect and validate ELF executables from build trees into a flat output directory.W
"""

import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path

from elftools.elf.elffile import ELFFile
from elftools.common.exceptions import ELFError


MIN_TEXT_SIZE_DEFAULT = 64


def validate_elf(path: Path, min_text_size: int) -> bool:
    """Validate an ELF binary has useful content for training."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"\x7fELF":
                return False
            f.seek(0)
            elf = ELFFile(f)

            text_section = elf.get_section_by_name(".text")
            if text_section is None or text_section.data_size < min_text_size:
                return False

            symtab = elf.get_section_by_name(".symtab")
            has_func_symbols = False
            if symtab:
                from elftools.elf.sections import SymbolTableSection

                if isinstance(symtab, SymbolTableSection):
                    for sym in symtab.iter_symbols():
                        if sym.entry.st_info.type == "STT_FUNC":
                            has_func_symbols = True
                            break

            has_eh_frame = elf.get_section_by_name(".eh_frame") is not None

            if not has_func_symbols and not has_eh_frame:
                return False

            return True
    except (ELFError, Exception):
        return False


def find_elf_files(build_root: Path) -> list[Path]:
    """Find all ELF executable files in build tree, skipping .o files."""
    elf_files = []
    for root, dirs, files in os.walk(build_root):
        for fname in files:
            fpath = Path(root) / fname
            if fpath.suffix == ".o":
                continue
            if not fpath.is_file():
                continue
            # Quick ELF magic check
            try:
                with open(fpath, "rb") as f:
                    if f.read(4) == b"\x7fELF":
                        elf_files.append(fpath)
            except Exception:
                continue
    return elf_files


def parse_build_path(elf_path: Path, build_root: Path) -> dict | None:
    """Extract compiler config, source type, package name from the build tree path.

    Expected build tree structure:
        build_root/{source_type}/{package_name}/{compiler}_O{level}/.../{binary}
    """
    try:
        rel = elf_path.relative_to(build_root)
        parts = rel.parts
        if len(parts) < 4:
            return None

        source_type = parts[0]
        package_name = parts[1]
        config = parts[2]

        match = re.match(r"^(gcc|clang)_O([0-3])$", config)
        if not match:
            return None

        compiler = match.group(1)
        opt_level = match.group(2)
        binary_name = elf_path.stem

        return {
            "source_type": source_type,
            "package": package_name,
            "compiler": compiler,
            "opt_level": opt_level,
            "binary_name": binary_name,
            "config": config,
        }
    except Exception:
        return None


def make_output_name(info: dict) -> str:
    """Create output filename: {compiler}_O{level}_{source}_{package}_{binary}"""
    def sanitize(s):
        return re.sub(r"[^a-zA-Z0-9_.-]", "_", s)

    parts = [
        info["config"],
        sanitize(info["source_type"]),
        sanitize(info["package"]),
        sanitize(info["binary_name"]),
    ]
    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Collect validated ELF binaries from build trees")
    parser.add_argument("--build-root", type=Path, required=True, help="Root of build trees")
    parser.add_argument("--output-dir", type=Path, required=True, help="Flat output directory for binaries")
    parser.add_argument("--min-text-size", type=int, default=MIN_TEXT_SIZE_DEFAULT, help=f"Minimum .text section size in bytes (default: {MIN_TEXT_SIZE_DEFAULT})")
    parser.add_argument("--manifest", type=Path, default=None, help="Path to write CSV manifest")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {args.build_root} for ELF files...")
    elf_files = find_elf_files(args.build_root)
    print(f"Found {len(elf_files)} ELF files")

    collected = 0
    skipped_validation = 0
    skipped_parse = 0
    skipped_duplicate = 0
    seen_names = set()
    manifest_rows = []

    for elf_path in elf_files:
        info = parse_build_path(elf_path, args.build_root)
        if info is None:
            skipped_parse += 1
            continue

        if not validate_elf(elf_path, args.min_text_size):
            skipped_validation += 1
            continue

        out_name = make_output_name(info)

        if out_name in seen_names:
            idx = 1
            while f"{out_name}_{idx}" in seen_names:
                idx += 1
            out_name = f"{out_name}_{idx}"

        seen_names.add(out_name)
        dest = args.output_dir / out_name

        try:
            shutil.copy2(elf_path, dest)
            collected += 1
            manifest_rows.append(
                {
                    "filename": out_name,
                    "source_type": info["source_type"],
                    "package": info["package"],
                    "compiler": info["compiler"],
                    "opt_level": info["opt_level"],
                    "original_path": str(elf_path),
                }
            )
        except Exception as e:
            print(f"  [error] Failed to copy {elf_path}: {e}")

    print(f"\nCollected: {collected}")
    print(f"Skipped (validation): {skipped_validation}")
    print(f"Skipped (path parse): {skipped_parse}")
    print(f"Skipped (duplicate): {skipped_duplicate}")

    manifest_path = args.manifest or (args.output_dir / "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "source_type", "package", "compiler", "opt_level", "original_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
