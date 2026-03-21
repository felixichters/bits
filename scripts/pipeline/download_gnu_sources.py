#!/usr/bin/env python3
"""
Download GNU C/C++ project source tarballs from ftp.gnu.org.

Scrapes the FTP directory listing for the latest release tarball of each
curated GNU project, downloads and extracts them.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen, urlretrieve
from urllib.error import URLError

# Curated list of GNU C/C++ projects known to build with autoconf
GNU_PROJECTS = [
    "coreutils",
    "grep",
    "sed",
    "gawk",
    "findutils",
    "diffutils",
    "tar",
    "gzip",
    "bzip2",
    "patch",
    "make",
    "which",
    "wget",
    "screen",
    "less",
    "nano",
    "bash",
    "binutils",
    "gdb",
    "gcc",
    "glibc",
    "grub",
    "inetutils",
    "texinfo",
    "gettext",
    "libtool",
    "autoconf",
    "automake",
    "m4",
    "bison",
    "flex",
    "indent",
    "bc",
    "units",
    "time",
    "sharutils",
    "cpio",
    "ed",
    "hello",
    "mtools",
    "parted",
    "pspp",
    "gsl",
    "glpk",
    "libmicrohttpd",
    "readline",
    "ncurses",
    "libunistring",
    "libidn",
    "libidn2",
    "libgcrypt",
    "libgpg-error",
    "libtasn1",
    "gnutls",
    "nettle",
    "gnupg",
    "gpgme",
    "libassuan",
    "libksba",
    "npth",
    "pinentry",
    "gnulib",
    "plotutils",
    "recutils",
    "datamash",
    "parallel",
    "stow",
    "enscript",
    "a2ps",
    "barcode",
    "diction",
    "spell",
    "aspell",
    "wdiff",
    "libredwg",
    "octave",
    "R",
    "gmp",
    "mpfr",
    "mpc",
    "ntl",
    "freeipmi",
    "httptunnel",
    "rush",
    "dico",
    "mailutils",
    "anubis",
    "libffcall",
    "lightning",
    "libjit",
    "poke",
    "guile",
    "mit-scheme",
    "classpath",
    "sather",
    "smalltalk",
    "talkfilters",
    "rcs",
    "global",
    "idutils",
    "src-highlite",
    "gengen",
    "gengetopt",
    "help2man",
    "teximpatient",
    "complexity",
    "combine",
    "bool",
    "gperf",
    "libiconv",
    "fribidi",
    "recode",
    "denemo",
    "lilypond",
    "solfege",
    "gnubg",
    "chess",
    "xboard",
    "gnushogi",
    "cflow",
    "cssc",
    "cppi",
    "direvent",
    "acct",
    "alive",
    "dmd",
    "foliot",
    "gcal",
    "gnu-pw-mgr",
    "gama",
    "gdbm",
    "gsasl",
    "gss",
    "guix",
    "health",
    "hurd",
    "gv",
    "groff",
    "lsh",
    "macchanger",
    "mcsim",
    "mig",
    "libsigsegv",
    "pth",
    "remotecontrol",
    "serveez",
    "sipwitch",
    "sqltutor",
    "termutils",
    "teseq",
    "trueprint",
    "units",
    "vmgen",
    "xnee",
    "zile",
]

GNU_FTP_BASE = "https://ftp.gnu.org/gnu"


def get_latest_tarball_url(project: str) -> str | None:
    """Scrape ftp.gnu.org for the latest release tarball of a GNU project."""
    url = f"{GNU_FTP_BASE}/{project}/"
    try:
        with urlopen(url, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except (URLError, OSError):
        return None

    # Find tarball links: project-VERSION.tar.{gz,xz,bz2}
    pattern = rf'href="({re.escape(project)}-(\d[\d.]*\d?)\.tar\.(?:gz|xz|bz2))"'
    matches = re.findall(pattern, html)
    if not matches:
        subdir_pattern = rf'href="(\d[\d.]*\d?)/"'
        subdirs = re.findall(subdir_pattern, html)
        if subdirs:
            subdirs.sort(key=lambda v: list(map(int, re.findall(r"\d+", v))), reverse=True)
            for version in subdirs[:3]:
                suburl = f"{url}{version}/"
                try:
                    with urlopen(suburl, timeout=30) as resp:
                        subhtml = resp.read().decode("utf-8", errors="ignore")
                    sub_pattern = rf'href="({re.escape(project)}-{re.escape(version)}[^"]*\.tar\.(?:gz|xz|bz2))"'
                    sub_matches = re.findall(sub_pattern, subhtml)
                    if sub_matches:
                        return f"{suburl}{sub_matches[0]}"
                except (URLError, OSError):
                    continue
        return None

    def version_key(m):
        try:
            return list(map(int, re.findall(r"\d+", m[1])))
        except ValueError:
            return [0]

    matches.sort(key=version_key, reverse=True)
    return f"{url}{matches[0][0]}"


def download_and_extract(url: str, output_dir: Path, project: str) -> bool:
    """Download a tarball and extract it."""
    dest = output_dir / project
    if dest.exists() and any(dest.iterdir()):
        print(f"  [skip] {project} already exists at {dest}")
        return True

    print(f"  [download] {url}")
    try:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name
        urlretrieve(url, tmp_path)
    except Exception as e:
        print(f"  [error] Download failed for {project}: {e}")
        return False

    try:
        dest.mkdir(parents=True, exist_ok=True)
        if url.endswith(".tar.xz"):
            cmd = ["tar", "xJf", tmp_path, "--strip-components=1", "-C", str(dest)]
        elif url.endswith(".tar.bz2"):
            cmd = ["tar", "xjf", tmp_path, "--strip-components=1", "-C", str(dest)]
        else:
            cmd = ["tar", "xzf", tmp_path, "--strip-components=1", "-C", str(dest)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  [error] Extraction failed for {project}: {result.stderr[:500]}")
            shutil.rmtree(dest, ignore_errors=True)
            return False
        print(f"  [ok] {project} extracted to {dest}")
        return True
    except Exception as e:
        print(f"  [error] Extraction failed for {project}: {e}")
        shutil.rmtree(dest, ignore_errors=True)
        return False
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="Download GNU C/C++ source packages")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for extracted sources")
    parser.add_argument("--max-packages", type=int, default=0, help="Max packages to download (0 = all)")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    projects = GNU_PROJECTS
    if args.max_packages > 0:
        projects = projects[: args.max_packages]

    success_count = 0
    fail_count = 0

    for i, project in enumerate(projects):
        print(f"[{i+1}/{len(projects)}] {project}")
        url = get_latest_tarball_url(project)
        if url is None:
            print(f"  [skip] No tarball found for {project}")
            fail_count += 1
            continue

        if download_and_extract(url, output_dir, project):
            success_count += 1
        else:
            fail_count += 1

    print(f"\nDone. {success_count} downloaded, {fail_count} failed/skipped.")


if __name__ == "__main__":
    main()
