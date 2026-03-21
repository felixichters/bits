#!/usr/bin/env python3
"""
Download Debian source packages that build C/C++ code.

Fetches the Sources.gz index from a Debian mirror, filters for packages whose
Build-Depends mention gcc/g++/cmake/autoconf, then downloads and extracts
each package with dpkg-source.
"""

import argparse
import gzip
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen, urlretrieve
from urllib.error import URLError


DEFAULT_MIRROR = "https://deb.debian.org/debian"
DEFAULT_SUITE = "bookworm"
DEFAULT_COMPONENT = "main"


def fetch_sources_index(mirror: str, suite: str, component: str) -> str:
    """Download and decompress Sources.gz index."""
    url = f"{mirror}/dists/{suite}/{component}/source/Sources.gz"
    print(f"Fetching {url}...")
    with urlopen(url, timeout=60) as resp:
        data = gzip.decompress(resp.read())
    return data.decode("utf-8", errors="ignore")


def parse_sources(text: str) -> list[dict]:
    """Parse deb822 format Sources file into list of package dicts."""
    packages = []
    current = {}

    for line in text.splitlines():
        if line == "":
            if current:
                packages.append(current)
                current = {}
            continue
        if line.startswith(" ") or line.startswith("\t"):
            # Continuation of previous field
            if current and "_last_key" in current:
                current[current["_last_key"]] += "\n" + line
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            current[key] = value
            current["_last_key"] = key

    if current:
        packages.append(current)

    # Clean up _last_key
    for pkg in packages:
        pkg.pop("_last_key", None)

    return packages


def is_c_cpp_package(pkg: dict) -> bool:
    """Check if a package likely builds C/C++ code based on Build-Depends."""
    build_deps = pkg.get("Build-Depends", "") + pkg.get("Build-Depends-Indep", "")
    build_deps_lower = build_deps.lower()

    # Must depend on a C/C++ build tool
    c_indicators = [
        "gcc", "g++", "clang", "cmake", "autoconf", "automake",
        "libtool", "meson", "debhelper",  # debhelper often implies C builds
    ]
    has_c_build = any(ind in build_deps_lower for ind in c_indicators)

    # Filter out packages that are clearly not C/C++
    pkg_name = pkg.get("Package", "").lower()
    exclude_patterns = [
        "python-", "ruby-", "node-", "golang-", "haskell-",
        "r-cran-", "r-bioc-", "ocaml-", "libghc-", "fonts-",
        "texlive-", "aspell-", "hunspell-", "myspell-",
    ]
    if any(pkg_name.startswith(p) for p in exclude_patterns):
        return False

    return has_c_build


def get_dsc_url(pkg: dict, mirror: str, suite: str) -> tuple[str | None, list[tuple[str, str]]]:
    """Extract .dsc URL and other file URLs from package Files field."""
    files_text = pkg.get("Files", "")
    directory = pkg.get("Directory", "")
    if not files_text or not directory:
        return None, []

    dsc_url = None
    file_urls = []

    for line in files_text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            md5, size, filename = parts[0], parts[1], parts[2]
            url = f"{mirror}/{directory}/{filename}"
            file_urls.append((filename, url))
            if filename.endswith(".dsc"):
                dsc_url = url

    return dsc_url, file_urls


def download_and_extract_package(
    pkg: dict, mirror: str, output_dir: Path, work_dir: Path
) -> bool:
    """Download .dsc + source files and extract with dpkg-source."""
    pkg_name = pkg.get("Package", "unknown")
    pkg_version = pkg.get("Version", "unknown")
    dest = output_dir / pkg_name

    if dest.exists() and any(dest.iterdir()):
        print(f"  [skip] {pkg_name} already exists")
        return True

    dsc_url, file_urls = get_dsc_url(pkg, mirror, "")
    if not dsc_url:
        # Rebuild URL from mirror + directory
        directory = pkg.get("Directory", "")
        if not directory:
            return False
        files_text = pkg.get("Files", "")
        dsc_url = None
        file_urls = []
        for line in files_text.strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 3:
                filename = parts[2]
                url = f"{mirror}/{directory}/{filename}"
                file_urls.append((filename, url))
                if filename.endswith(".dsc"):
                    dsc_url = url

    if not dsc_url:
        print(f"  [error] No .dsc file found for {pkg_name}")
        return False

    # Create temp download directory
    dl_dir = work_dir / pkg_name
    dl_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download all source files
        dsc_path = None
        for filename, url in file_urls:
            local_path = dl_dir / filename
            if local_path.exists():
                if filename.endswith(".dsc"):
                    dsc_path = local_path
                continue
            try:
                urlretrieve(url, local_path)
                if filename.endswith(".dsc"):
                    dsc_path = local_path
            except Exception as e:
                print(f"  [error] Failed to download {filename}: {e}")
                return False

        if not dsc_path:
            print(f"  [error] .dsc file not downloaded for {pkg_name}")
            return False

        # Extract with dpkg-source
        dest.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["dpkg-source", "-x", "--no-check", str(dsc_path), str(dest)],
            capture_output=True,
            text=True,
            errors="ignore",
            timeout=120,
            cwd=dl_dir,
        )

        if result.returncode != 0:
            # Fallback: try manual extraction of .orig.tar.* files
            extracted = False
            for filename, _ in file_urls:
                if ".orig.tar." in filename:
                    tarball = dl_dir / filename
                    if tarball.exists():
                        try:
                            subprocess.run(
                                ["tar", "xf", str(tarball), "--strip-components=1", "-C", str(dest)],
                                capture_output=True,
                                timeout=120,
                            )
                            extracted = True
                        except Exception:
                            pass
            if not extracted:
                print(f"  [error] dpkg-source failed for {pkg_name}: {result.stderr[:300]}")
                shutil.rmtree(dest, ignore_errors=True)
                return False

        return True

    except Exception as e:
        print(f"  [error] Failed to extract {pkg_name}: {e}")
        shutil.rmtree(dest, ignore_errors=True)
        return False
    finally:
        # Clean up downloaded files
        shutil.rmtree(dl_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Download Debian C/C++ source packages")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for extracted sources")
    parser.add_argument("--max-packages", type=int, default=0, help="Max packages to download (0 = all)")
    parser.add_argument("--mirror-url", type=str, default=DEFAULT_MIRROR, help="Debian mirror URL")
    parser.add_argument("--suite", type=str, default=DEFAULT_SUITE, help="Debian suite (default: bookworm)")
    parser.add_argument("--component", type=str, default=DEFAULT_COMPONENT, help="Debian component (default: main)")
    parser.add_argument("--work-dir", type=Path, default=None, help="Working directory for downloads")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="debian_dl_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    # Fetch and parse Sources index
    sources_text = fetch_sources_index(args.mirror_url, args.suite, args.component)
    all_packages = parse_sources(sources_text)
    print(f"Parsed {len(all_packages)} total source packages")

    # Filter for C/C++ packages
    c_packages = [p for p in all_packages if is_c_cpp_package(p)]
    print(f"Filtered to {len(c_packages)} C/C++ packages")

    if args.max_packages > 0:
        c_packages = c_packages[: args.max_packages]
        print(f"Limited to {len(c_packages)} packages")

    success_count = 0
    fail_count = 0

    for i, pkg in enumerate(c_packages):
        pkg_name = pkg.get("Package", "unknown")
        print(f"[{i+1}/{len(c_packages)}] {pkg_name}")
        if download_and_extract_package(pkg, args.mirror_url, args.output_dir, work_dir):
            success_count += 1
        else:
            fail_count += 1

    # Cleanup work dir
    if args.work_dir is None:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\nDone. {success_count} downloaded, {fail_count} failed/skipped.")


if __name__ == "__main__":
    main()
