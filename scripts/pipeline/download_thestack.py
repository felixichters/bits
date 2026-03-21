#!/usr/bin/env python3
"""
Download C/C++ source files from The Stack v1 (bigcode/the-stack) via HuggingFace datasets.

Writes source files directly to disk as sources/thestack/{repo_name}/{path},
grouping by repository so compile_package.py can attempt native builds
where Makefiles/CMakeLists.txt exist.

Prerequisites:
    1. Accept the dataset terms at https://huggingface.co/datasets/bigcode/the-stack
    2. Authenticate via one of:
       - `huggingface-cli login`
       - Set HF_TOKEN environment variable
       - Pass --token hf_xxxx to this script
"""

import argparse
import os
import re
import sys
from pathlib import Path

LANGUAGE_DATA_DIRS = {
    "c": "data/c",
    "c++": "data/c++",
}

def sanitize_path(s: str) -> str:
    """Sanitize a path component for filesystem use."""
    s = re.sub(r"[^\w./-]", "_", s)
    s = s.strip("_./")
    return s[:200]

def sanitize_repo_name(repo_name: str) -> str:
    """Convert repo name like 'owner/repo' to 'owner_repo'."""
    return re.sub(r"[^\w.-]", "_", repo_name).strip("_")

def main():
    parser = argparse.ArgumentParser(description="Download C/C++ source from The Stack v1 to disk")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for source files")
    parser.add_argument("--max-files", type=int, default=0, help="Max files to download per language (0 = unlimited)")
    parser.add_argument("--min-stars", type=int, default=1, help="Minimum repository star count")
    parser.add_argument("--languages", nargs="+", default=["c", "c++"], choices=["c", "c++"], help="Languages to download")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace API token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    from datasets import load_dataset

    # Resolve token: --token flag > HF_TOKEN env var > cached login
    token = args.token or os.environ.get("HF_TOKEN")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    valid_extensions = {".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"}
    total_file_count = 0
    total_repo_count = 0

    for lang in args.languages:
        data_dir = LANGUAGE_DATA_DIRS[lang]
        print(f"Loading The Stack v1 for language: {lang} (data_dir={data_dir})")

        try:
            load_kwargs = {
                "path": "bigcode/the-stack",
                "data_dir": data_dir,
                "split": "train",
                "streaming": True,
            }
            if token:
                load_kwargs["token"] = token
            ds = load_dataset(**load_kwargs)
        except Exception as e:
            msg = str(e)
            if "401" in msg or "unauthorized" in msg.lower() or "gated" in msg.lower():
                print(f"  [error] Authentication failed for The Stack.")
            else:
                print(f"  [error] Failed to load dataset: {e}")
            continue

        file_count = 0
        repo_count = 0
        seen_repos = set()

        for sample in ds:
            if args.max_files > 0 and file_count >= args.max_files:
                break

            # The Stack v1 column names
            content = sample.get("content", "")
            path = sample.get("max_stars_repo_path", "unknown.c")
            repo_name = sample.get("max_stars_repo_name", "unknown")
            stars = sample.get("max_stars_count", 0)

            # Filters
            if stars is not None and stars < args.min_stars:
                continue

            ext = os.path.splitext(path)[1].lower()
            if ext not in valid_extensions:
                continue

            if not content or len(content) < 50:
                continue

            # Write to disk
            safe_repo = sanitize_repo_name(repo_name)
            safe_path = sanitize_path(path)
            if not safe_path:
                safe_path = f"file_{file_count}{ext}"

            dest = args.output_dir / safe_repo / safe_path
            dest.parent.mkdir(parents=True, exist_ok=True)

            try:
                dest.write_text(content, errors="ignore")
                file_count += 1
                if safe_repo not in seen_repos:
                    seen_repos.add(safe_repo)
                    repo_count += 1
            except Exception:
                continue

            if file_count % 1000 == 0:
                print(f"  [{lang}] Written {file_count} files from {repo_count} repos")

        print(f"  [{lang}] Done: {file_count} files from {repo_count} repos")
        total_file_count += file_count
        total_repo_count += repo_count

    print(f"\nTotal: {total_file_count} files from {total_repo_count} repos written to {args.output_dir}")


if __name__ == "__main__":
    main()