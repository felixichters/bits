#!/usr/bin/env python3
"""
Generate package_list.txt by walking the sources/ directory.

Output format: one line per package: source_type|pkg_name|source_dir
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate package list from sources directory")
    parser.add_argument("--sources-dir", type=Path, required=True, help="Root sources directory containing {debian,gnu,thestack}/ subdirs")
    parser.add_argument("--output", type=Path, required=True, help="Output path for package_list.txt")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(args.output, "w") as f:
        for source_type_dir in sorted(args.sources_dir.iterdir()):
            if not source_type_dir.is_dir():
                continue
            source_type = source_type_dir.name
            for pkg_dir in sorted(source_type_dir.iterdir()):
                if not pkg_dir.is_dir():
                    continue
                has_sources = any(
                    pkg_dir.rglob("*.c")
                ) or any(
                    pkg_dir.rglob("*.cpp")
                )
                if not has_sources:
                    has_sources = any(pkg_dir.rglob("*.cc")) or any(pkg_dir.rglob("*.cxx"))

                if has_sources:
                    f.write(f"{source_type}|{pkg_dir.name}|{pkg_dir}\n")
                    count += 1

    print(f"Generated {args.output} with {count} packages")


if __name__ == "__main__":
    main()
