"""
Download C and C++ source code from The Stack (Hugging Face) and output
gzip-compressed JSONL files matching the format expected by CompilePipeline.py.

Prerequisites:
    pip install datasets huggingface_hub
    huggingface-cli login  # accept The Stack's terms at https://huggingface.co/datasets/bigcode/the-stack first

Output format (one JSON object per line, gzip compressed):
    {"repo_name": "author/repo", "path": "src/main.c", "content": "..."}

Usage:
    python DownloadTheStack.py                          # Download both C and C++, 50k files each
    python DownloadTheStack.py --languages c            # C only
    python DownloadTheStack.py --languages c c++        # Both (default)
    python DownloadTheStack.py --max-files 100000       # More files per language
    python DownloadTheStack.py --min-stars 10           # Only repos with 10+ stars
    python DownloadTheStack.py --output-dir /mnt/storage-box/github_sources
"""

import argparse
import json
import gzip
import os
import time
from pathlib import Path


# How many samples to write per output file (keeps files manageable)
SAMPLES_PER_FILE = 50000

# Map from our language names to The Stack's data_dir names
LANGUAGE_MAP = {
    "c": "data/c",
    "c++": "data/c++",
}

# File extensions we care about per language
EXTENSIONS = {
    "c": {".c", ".h"},
    "c++": {".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".h"},
}

# Output file prefix per language (matches BIGQUERY_PREFIXES in CompilePipeline.py)
OUTPUT_PREFIX = {
    "c": "github_c",
    "c++": "github_cpp",
}


def download_language(language, output_dir, max_files, min_stars, samples_per_file):
    from datasets import load_dataset

    data_dir = LANGUAGE_MAP[language]
    prefix = OUTPUT_PREFIX[language]
    valid_extensions = EXTENSIONS[language]

    print(f"\n{'='*60}")
    print(f"Downloading {language.upper()} files from The Stack")
    print(f"Max files: {max_files}, Min stars: {min_stars}")
    print(f"{'='*60}\n")

    ds = load_dataset(
        "bigcode/the-stack",
        data_dir=data_dir,
        split="train",
        streaming=True,
    )

    file_index = 0       # Which output file we're on
    sample_count = 0     # Samples in current file
    total_count = 0      # Total samples written
    writer = None
    output_path = None

    def open_next_file():
        nonlocal file_index, sample_count, writer, output_path
        if writer:
            writer.close()
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  Closed {output_path} ({size_mb:.1f} MB)")

        filename = f"{prefix}_{file_index:012d}.gz"
        output_path = os.path.join(output_dir, filename)
        writer = gzip.open(output_path, "wt", encoding="utf-8")
        sample_count = 0
        file_index += 1

    open_next_file()
    start_time = time.time()

    try:
        for sample in ds:
            if total_count >= max_files:
                break

            # Filter by extension
            ext = os.path.splitext(sample.get("max_stars_repo_path", ""))[1].lower()
            if ext not in valid_extensions:
                continue

            # Filter by stars
            stars = sample.get("max_stars_count", 0) or 0
            if stars < min_stars:
                continue

            content = sample.get("content", "")
            if not content or len(content) < 20:
                continue

            # Build JSONL entry matching the format DGithubJSON2FILE.py expects
            repo_name = sample.get("max_stars_repo_name", "unknown/unknown")
            file_path = sample.get("max_stars_repo_path", "unknown.c")

            entry = {
                "repo_name": repo_name,
                "path": file_path,
                "content": content,
            }

            writer.write(json.dumps(entry, ensure_ascii=False) + "\n")
            sample_count += 1
            total_count += 1

            if sample_count >= samples_per_file:
                open_next_file()

            if total_count % 10000 == 0:
                elapsed = time.time() - start_time
                rate = total_count / elapsed
                print(f"  [{language.upper()}] {total_count}/{max_files} files ({rate:.0f} files/sec)")

    except KeyboardInterrupt:
        print(f"\n  Interrupted. Saving progress...")

    finally:
        if writer:
            writer.close()
            if output_path and os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"  Closed {output_path} ({size_mb:.1f} MB)")

    elapsed = time.time() - start_time
    print(f"\n  [{language.upper()}] Done: {total_count} files in {elapsed:.0f}s")
    return total_count


def main():
    parser = argparse.ArgumentParser(
        description="Download C/C++ source code from The Stack (Hugging Face) "
                    "into gzip JSONL format for CompilePipeline.py"
    )
    parser.add_argument(
        "--languages", nargs="+", default=["c", "c++"], choices=["c", "c++"],
        help="Languages to download (default: c c++)"
    )
    parser.add_argument(
        "--max-files", type=int, default=50000,
        help="Maximum number of source files to download per language (default: 50000)"
    )
    parser.add_argument(
        "--min-stars", type=int, default=0,
        help="Minimum star count for repos (default: 0, set to e.g. 10 to filter toy repos)"
    )
    parser.add_argument(
        "--samples-per-file", type=int, default=SAMPLES_PER_FILE,
        help=f"Number of samples per output .gz file (default: {SAMPLES_PER_FILE})"
    )
    parser.add_argument(
        "--output-dir", type=str, default="THE_STACK_EXPORT",
        help="Output directory for .gz files (default: THE_STACK_EXPORT)"
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    total = 0
    for lang in args.languages:
        count = download_language(
            language=lang,
            output_dir=args.output_dir,
            max_files=args.max_files,
            min_stars=args.min_stars,
            samples_per_file=args.samples_per_file,
        )
        total += count

    print(f"\n{'='*60}")
    print(f"Total: {total} files downloaded to {args.output_dir}/")
    print(f"Output files use prefixes: {[OUTPUT_PREFIX[l] for l in args.languages]}")
    print(f"\nNext step: run CompilePipeline.py with --source {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
