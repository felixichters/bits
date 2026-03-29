#!/usr/bin/env python3
"""
Batch compile orchestrator: reads package_list.txt and compiles all packages
with all 8 compiler configurations (gcc/clang x O0-O3).

Uses multiprocessing to parallelize across packages. Tracks progress in
compile_status.json for resumability.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compile_package import compile_package


COMPILERS = ["gcc", "clang"]
OPT_LEVELS = [0, 1, 2, 3]
CONFIGS = [(c, o) for c in COMPILERS for o in OPT_LEVELS]


def load_package_list(path: Path) -> list[dict]:
    """Load package_list.txt. Format: source_type|pkg_name|source_dir"""
    packages = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", 2)
            if len(parts) != 3:
                continue
            packages.append({
                "source_type": parts[0],
                "name": parts[1],
                "source_dir": parts[2],
            })
    return packages


def load_status(path: Path) -> dict:
    """Load compile_status.json for resuming."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_status(path: Path, status: dict):
    """Save compile_status.json atomically."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp, path)


def compile_one(source_type: str, pkg_name: str, source_dir: str,
                build_root: str, compiler: str, opt_level: int, timeout: int) -> dict:
    """Compile a single package with a single config. Called in worker process."""
    config = f"{compiler}_O{opt_level}"
    build_dir = Path(build_root) / source_type / pkg_name / config

    result = compile_package(
        source_dir=Path(source_dir),
        build_dir=build_dir,
        compiler=compiler,
        optimization=opt_level,
        timeout=timeout,
    )
    result["source_type"] = source_type
    return result


def main():
    parser = argparse.ArgumentParser(description="Batch compile all packages with all configs")
    parser.add_argument("--package-list", type=Path, required=True, help="Path to package_list.txt")
    parser.add_argument("--build-root", type=Path, required=True, help="Root directory for build trees")
    parser.add_argument("--workers", type=int, default=0, help="Number of worker processes (0 = cpus/2)")
    parser.add_argument("--timeout", type=int, default=1800, help="Build timeout per package in seconds")
    parser.add_argument("--resume", action="store_true", help="Resume from compile_status.json")
    parser.add_argument("--status-file", type=Path, default=None, help="Path to compile_status.json")
    args = parser.parse_args()

    packages = load_package_list(args.package_list)
    print(f"Loaded {len(packages)} packages, {len(CONFIGS)} configs = {len(packages) * len(CONFIGS)} total tasks")

    status_file = args.status_file or (args.build_root / "compile_status.json")
    status = load_status(status_file) if args.resume else {}

    workers = args.workers if args.workers > 0 else max(1, os.cpu_count() // 2)
    print(f"Using {workers} workers")

    tasks = []
    for pkg in packages:
        for compiler, opt_level in CONFIGS:
            key = f"{pkg['source_type']}|{pkg['name']}|{compiler}_O{opt_level}"
            if key in status and status[key].get("status") in ("success", "partial", "failed"):
                continue
            tasks.append((pkg, compiler, opt_level))

    print(f"Tasks to run: {len(tasks)} (skipped {len(packages) * len(CONFIGS) - len(tasks)} already done)")

    counters = {"success": 0, "partial": 0, "failed": 0}
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for pkg, compiler, opt_level in tasks:
            future = executor.submit(
                compile_one,
                pkg["source_type"],
                pkg["name"],
                pkg["source_dir"],
                str(args.build_root),
                compiler,
                opt_level,
                args.timeout,
            )
            key = f"{pkg['source_type']}|{pkg['name']}|{compiler}_O{opt_level}"
            futures[future] = key

        done_count = 0
        total = len(futures)

        for future in as_completed(futures):
            key = futures[future]
            done_count += 1

            try:
                result = future.result()
                s = result.get("status", "failed")
                counters[s] = counters.get(s, 0) + 1
                status[key] = {
                    "status": s,
                    "binaries_found": result.get("binaries_found", 0),
                    "build_system": result.get("build_system", ""),
                    "error": result.get("error", ""),
                }
            except Exception as e:
                counters["failed"] += 1
                status[key] = {"status": "failed", "error": str(e)}

            if done_count % 10 == 0 or done_count == total:
                elapsed = time.time() - start_time
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (total - done_count) / rate if rate > 0 else 0
                print(
                    f"[{done_count}/{total}] "
                    f"success={counters['success']} partial={counters['partial']} "
                    f"failed={counters['failed']} "
                    f"({rate:.1f} tasks/s, ETA {eta/60:.0f}m)"
                )
                save_status(status_file, status)

    save_status(status_file, status)
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Compilation complete in {elapsed/60:.1f} minutes")
    print(f"  Success: {counters['success']}")
    print(f"  Partial: {counters['partial']}")
    print(f"  Failed:  {counters['failed']}")
    print(f"Status saved to {status_file}")


if __name__ == "__main__":
    main()
