#!/usr/bin/env python3
"""
Compile a single source package with a specific compiler configuration.

Detects the build system (cmake > autoconf > meson > plain Makefile > fallback)
and compiles with the given compiler/optimization level via environment variables.

Output: JSON status to stdout with package name, config, status, binaries found, and errors.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


CONFIGURE_TIMEOUT = 600
MAKE_TIMEOUT = 1800

C_EXTENSIONS = {".c"}
CPP_EXTENSIONS = {".cpp", ".cc", ".cxx"}
SOURCE_EXTENSIONS = C_EXTENSIONS | CPP_EXTENSIONS


def detect_build_system(source_dir: Path) -> str:
    """Detect the build system used by a source package."""
    if (source_dir / "CMakeLists.txt").exists():
        return "cmake"
    if (source_dir / "configure").exists() or (source_dir / "configure.ac").exists():
        return "autoconf"
    if (source_dir / "meson.build").exists():
        return "meson"
    if (source_dir / "Makefile").exists() or (source_dir / "makefile").exists():
        return "makefile"
    return "fallback"


def get_compiler_env(compiler: str, optimization: int) -> dict:
    """Build environment variables for compiler override."""
    env = os.environ.copy()
    if compiler == "gcc":
        cc, cxx = "gcc", "g++"
    elif compiler == "clang":
        cc, cxx = "clang", "clang++"
    else:
        raise ValueError(f"Unknown compiler: {compiler}")

    flags = f"-O{optimization} -g -gdwarf"
    env["CC"] = cc
    env["CXX"] = cxx
    env["CFLAGS"] = flags
    env["CXXFLAGS"] = flags
    env["LDFLAGS"] = ""
    return env


def run_cmd(cmd, cwd, env, timeout, desc="command"):
    """Run a command with timeout, return (success, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            errors="ignore",
            timeout=timeout,
        )
        if result.returncode != 0:
            return False, result.stderr[:2000]
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"{desc} timed out after {timeout}s"
    except Exception as e:
        return False, str(e)


def build_cmake(source_dir: Path, build_dir: Path, env: dict, timeout: int) -> tuple[bool, str]:
    """Build with cmake."""
    build_dir.mkdir(parents=True, exist_ok=True)
    cc = env["CC"]
    cxx = env["CXX"]
    cflags = env["CFLAGS"]
    cxxflags = env["CXXFLAGS"]

    ok, err = run_cmd(
        [
            "cmake",
            str(source_dir),
            f"-DCMAKE_C_COMPILER={cc}",
            f"-DCMAKE_CXX_COMPILER={cxx}",
            f"-DCMAKE_C_FLAGS={cflags}",
            f"-DCMAKE_CXX_FLAGS={cxxflags}",
            "-DCMAKE_BUILD_TYPE=None",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=OFF",
        ],
        cwd=build_dir,
        env=env,
        timeout=CONFIGURE_TIMEOUT,
        desc="cmake configure",
    )
    if not ok:
        return False, f"cmake configure failed: {err}"

    ok, err = run_cmd(
        ["make", "-j4", "VERBOSE=1"],
        cwd=build_dir,
        env=env,
        timeout=timeout,
        desc="make",
    )
    if not ok:
        return False, f"make failed: {err}"
    return True, ""


def build_autoconf(source_dir: Path, build_dir: Path, env: dict, timeout: int) -> tuple[bool, str]:
    """Build with autoconf/configure."""
    build_dir.mkdir(parents=True, exist_ok=True)

    # Run autoreconf if configure doesn't exist but configure.ac does
    if not (source_dir / "configure").exists() and (source_dir / "configure.ac").exists():
        ok, err = run_cmd(
            ["autoreconf", "-fi"],
            cwd=source_dir,
            env=env,
            timeout=CONFIGURE_TIMEOUT,
            desc="autoreconf",
        )
        if not ok:
            return False, f"autoreconf failed: {err}"

    configure_path = source_dir / "configure"
    if not configure_path.exists():
        return False, "No configure script found after autoreconf"

    cc = env["CC"]
    cxx = env["CXX"]
    cflags = env["CFLAGS"]
    cxxflags = env["CXXFLAGS"]

    ok, err = run_cmd(
        [
            str(configure_path),
            f"CC={cc}",
            f"CXX={cxx}",
            f"CFLAGS={cflags}",
            f"CXXFLAGS={cxxflags}",
            "--disable-shared",
            "--disable-nls",
        ],
        cwd=build_dir,
        env=env,
        timeout=CONFIGURE_TIMEOUT,
        desc="configure",
    )
    if not ok:
        return False, f"configure failed: {err}"

    ok, err = run_cmd(
        ["make", "-j4"],
        cwd=build_dir,
        env=env,
        timeout=timeout,
        desc="make",
    )
    if not ok:
        return False, f"make failed: {err}"
    return True, ""


def build_meson(source_dir: Path, build_dir: Path, env: dict, timeout: int) -> tuple[bool, str]:
    """Build with meson + ninja."""
    ok, err = run_cmd(
        [
            "meson",
            "setup",
            str(build_dir),
            str(source_dir),
            "--default-library=static",
        ],
        cwd=source_dir,
        env=env,
        timeout=CONFIGURE_TIMEOUT,
        desc="meson setup",
    )
    if not ok:
        return False, f"meson setup failed: {err}"

    ok, err = run_cmd(
        ["ninja", "-j4"],
        cwd=build_dir,
        env=env,
        timeout=timeout,
        desc="ninja",
    )
    if not ok:
        return False, f"ninja failed: {err}"
    return True, ""


def build_makefile(source_dir: Path, build_dir: Path, env: dict, timeout: int) -> tuple[bool, str]:
    """Build with plain Makefile. Copies source to build_dir since in-tree build."""
    if build_dir != source_dir:
        shutil.copytree(source_dir, build_dir, dirs_exist_ok=True)

    cc = env["CC"]
    cxx = env["CXX"]
    cflags = env["CFLAGS"]
    cxxflags = env["CXXFLAGS"]

    ok, err = run_cmd(
        [
            "make",
            f"CC={cc}",
            f"CXX={cxx}",
            f"CFLAGS={cflags}",
            f"CXXFLAGS={cxxflags}",
            "-j4",
        ],
        cwd=build_dir,
        env=env,
        timeout=timeout,
        desc="make",
    )
    if not ok:
        return False, f"make failed: {err}"
    return True, ""


def build_fallback(source_dir: Path, build_dir: Path, env: dict, timeout: int) -> tuple[bool, str]:
    """Fallback: compile all source files individually, then link via symbol resolution.

    For repos without build systems.
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    cc = env["CC"]
    cxx = env["CXX"]
    cflags = env["CFLAGS"]
    cxxflags = env["CXXFLAGS"]

    source_files = []
    for ext in SOURCE_EXTENSIONS:
        source_files.extend(source_dir.rglob(f"*{ext}"))

    if not source_files:
        return False, "No source files found"

    include_dirs = set()
    for f in source_files:
        include_dirs.add(str(f.parent))
    include_flags = []
    for d in include_dirs:
        include_flags.extend(["-I", d])

    object_files = []
    for src in source_files:
        rel = src.relative_to(source_dir)
        obj = build_dir / rel.with_suffix(".o")
        obj.parent.mkdir(parents=True, exist_ok=True)

        is_cpp = src.suffix in CPP_EXTENSIONS
        compiler = cxx if is_cpp else cc
        flags = cxxflags if is_cpp else cflags

        ok, err = run_cmd(
            [compiler, "-c", *flags.split(), *include_flags, "-o", str(obj), str(src)],
            cwd=build_dir,
            env=env,
            timeout=60,
            desc=f"compile {src.name}",
        )
        if ok:
            object_files.append(obj)

    if not object_files:
        return False, "No object files compiled successfully"

    symbol_info = {}
    for obj in object_files:
        try:
            result = subprocess.run(
                ["nm", str(obj)],
                capture_output=True,
                text=True,
                errors="ignore",
                timeout=10,
            )
            defined = set()
            undefined = set()
            has_main = False
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    sym_type, sym_name = parts[-2], parts[-1]
                    if sym_type == "T":
                        defined.add(sym_name)
                        if sym_name == "main":
                            has_main = True
                    elif sym_type == "U":
                        undefined.add(sym_name)
            symbol_info[obj] = {
                "defined": defined,
                "undefined": undefined,
                "has_main": has_main,
            }
        except Exception:
            continue

    symbol_map = {}
    for obj, info in symbol_info.items():
        for sym in info["defined"]:
            symbol_map.setdefault(sym, []).append(obj)

    linked = 0
    main_files = [obj for obj, info in symbol_info.items() if info["has_main"]]

    for idx, main_obj in enumerate(main_files):
        needed = set()
        queue = [main_obj]
        visited = {main_obj}

        qi = 0
        while qi < len(queue):
            current = queue[qi]
            qi += 1
            for sym in symbol_info.get(current, {}).get("undefined", []):
                for provider in symbol_map.get(sym, []):
                    if provider not in visited:
                        if not symbol_info.get(provider, {}).get("has_main", False):
                            visited.add(provider)
                            queue.append(provider)
                            needed.add(provider)

        out_path = build_dir / f"executable{idx}"
        linker = cxx if any(o.suffix in {".cpp", ".cc", ".cxx"} for o in [main_obj] + list(needed)) else cc
        link_objs = [str(main_obj)] + [str(o) for o in needed]

        ok, err = run_cmd(
            [linker, *link_objs, "-o", str(out_path), "-lm"],
            cwd=build_dir,
            env=env,
            timeout=120,
            desc=f"link executable{idx}",
        )
        if ok:
            linked += 1

    if linked == 0 and main_files:
        return False, "All link attempts failed"
    if linked == 0:
        return False, "No main() functions found in source files"

    return True, ""


def find_elf_executables(build_dir: Path) -> list[Path]:
    """Find ELF executable files in build tree (skip .o files)."""
    executables = []
    try:
        result = subprocess.run(
            ["find", str(build_dir), "-type", "f", "-executable"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        for line in result.stdout.strip().splitlines():
            p = Path(line)
            if p.suffix == ".o":
                continue
            try:
                with open(p, "rb") as f:
                    magic = f.read(4)
                if magic == b"\x7fELF":
                    executables.append(p)
            except Exception:
                continue
    except Exception:
        pass
    return executables


def compile_package(
    source_dir: Path,
    build_dir: Path,
    compiler: str,
    optimization: int,
    timeout: int = MAKE_TIMEOUT,
) -> dict:
    """Compile a package and return status dict."""
    env = get_compiler_env(compiler, optimization)
    build_system = detect_build_system(source_dir)

    builders = {
        "cmake": build_cmake,
        "autoconf": build_autoconf,
        "meson": build_meson,
        "makefile": build_makefile,
        "fallback": build_fallback,
    }

    builder = builders[build_system]
    success, error = builder(source_dir, build_dir, env, timeout)

    binaries = find_elf_executables(build_dir) if success else []

    config = f"{compiler}_O{optimization}"
    return {
        "package": source_dir.name,
        "config": config,
        "build_system": build_system,
        "status": "success" if success and binaries else ("partial" if success else "failed"),
        "binaries_found": len(binaries),
        "binary_paths": [str(b) for b in binaries],
        "error": error if not success else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Compile a single source package")
    parser.add_argument("--source-dir", type=Path, required=True, help="Path to source directory")
    parser.add_argument("--build-dir", type=Path, required=True, help="Path to build output directory")
    parser.add_argument("--compiler", choices=["gcc", "clang"], required=True)
    parser.add_argument("--optimization", type=int, choices=[0, 1, 2, 3], required=True)
    parser.add_argument("--timeout", type=int, default=MAKE_TIMEOUT, help="Build timeout in seconds")
    args = parser.parse_args()

    result = compile_package(
        source_dir=args.source_dir,
        build_dir=args.build_dir,
        compiler=args.compiler,
        optimization=args.optimization,
        timeout=args.timeout,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
