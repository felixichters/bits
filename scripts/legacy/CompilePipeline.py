import argparse
import os
import sys
import shutil
import subprocess
import time
import glob
import json
import tarfile

# Default Paths
BASE_DIR = "/mnt/storage-box"
RAW_DATA_DIR = BASE_DIR + "/github_c" # Default directory containing the thousands of github_c/cpp_... files from Google BigQuery export
DIR_COMPILED = BASE_DIR + "/COMPILED" # Default final output directory

# Working Area
DIR_ZIPPED = "ZIPPED"           # Holds the current chunk of compressed JSON files
DIR_UNZIPPED = "UNZIPPED"       # Holds the extracted JSONs
DIR_C_COMPILE = "C_COMPILE"     # Holds the reconstructed source code
DIR_COMPILED_CHUNK = "COMPILED_CHUNK"   # Holds the Object/Binary files generated in this chunk

# Scripts
SCRIPT_UNZIP = "DGithubJSON2FILE.py"
SCRIPT_COMPILE = "SH2O.py"
SCRIPT_LINK = "ObjectToBinary.py"

# Settings
CHUNK_SIZE = 3                # Number of input files to process at once
STATE_FILE = "pipeline_state.json"

# Compiler configurations for diverse training data
COMPILERS = ["gcc", "clang"]
OPTIMIZATION_LEVELS = ["0", "1", "2", "3"]

# BigQuery file prefixes (C and C++ exports)
BIGQUERY_PREFIXES = ("github_c", "github_cpp")
# =================================================

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"processed_files": []}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

def ensure_dirs():
    for d in [DIR_ZIPPED, DIR_UNZIPPED, DIR_C_COMPILE, DIR_COMPILED_CHUNK, DIR_COMPILED]:
        os.makedirs(d, exist_ok=True)

def clean_directory(directory):
    """Deletes all contents of a directory but keeps the directory itself."""
    print(f"[*] Cleaning up directory: {directory}")
    if not os.path.exists(directory):
        return
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def run_step(script_name, args):
    """Runs a python script as a subprocess."""
    cmd = [sys.executable, script_name] + args
    print(f"[*] Running {script_name} {' '.join(args)}...")
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[!] Error running {script_name}: {e}")
        return False

def filter_repos(source_dir):
    """Remove repos with known issues (Linux kernel, etc.)."""
    disallowed_repo_names = ["linux", "android_kernel", "kernel_", "kernel_samsung", "Monarudo_GPU_M7", "shadowsocks"]
    for directory in os.listdir(source_dir):
        # Note: Folders are formatted as author_reponame
        if any(disallowed in "_" + directory for disallowed in disallowed_repo_names):
            dir_to_remove = os.path.join(source_dir, directory)
            print(f"[*] Removing unwanted repo directory: {dir_to_remove}")
            shutil.rmtree(dir_to_remove)

        # Check if sound/soc/codecs exists in the repo to get rid of Linux kernel repos
        elif os.path.exists(os.path.join(source_dir, directory, "sound", "soc", "codecs")):
            dir_to_remove = os.path.join(source_dir, directory)
            print(f"[*] Removing Linux kernel repo: {dir_to_remove}")
            shutil.rmtree(dir_to_remove)

def compile_and_link_config(compiler, opt_level, source_dir, compiled_chunk_base):
    """Compile and link source code with a specific compiler and optimization level.

    Each config gets its own output subdirectory under compiled_chunk_base:
        COMPILED_CHUNK/gcc_O2/repo1/...
    """
    config_name = f"{compiler}_O{opt_level}"
    config_dest = os.path.join(compiled_chunk_base, config_name)
    os.makedirs(config_dest, exist_ok=True)

    print(f"\n{'─'*40}")
    print(f"[*] Compiling with {config_name}")
    print(f"{'─'*40}")

    # --- Compile to object files (SH2O) ---
    success = run_step(SCRIPT_COMPILE, [
        "--source-path", source_dir,
        "--dest-path", config_dest,
        "--compiler", compiler,
        "--optimization", opt_level,
        "--number-of-processes", "8"
    ])
    if not success:
        print(f"[!] Compile step failed for {config_name}. Continuing with other configs...")
        return False

    # --- Link (ObjectToBinary) ---
    success = run_step(SCRIPT_LINK, [
        "--dest-path", config_dest,
        "--source-path", source_dir,
        "--compiler", compiler,
        "--number-of-processes", "4"
    ])
    if not success:
        print(f"[!] Link step failed for {config_name}. Continuing with other configs...")
        # Don't return False — partial results may still be useful

    return True

def main(input_dir=RAW_DATA_DIR, output_dir=DIR_COMPILED, compilers=None, opt_levels=None):
    if compilers is None:
        compilers = COMPILERS
    if opt_levels is None:
        opt_levels = OPTIMIZATION_LEVELS

    if not os.path.exists(input_dir):
        print(f"Error: Raw data directory '{input_dir}' not found.")
        return

    ensure_dirs()
    state = load_state()
    processed_set = set(state["processed_files"])

    # Collect BigQuery export files (both C and C++)
    all_files = sorted([
        f for f in os.listdir(input_dir)
        if any(f.startswith(prefix) for prefix in BIGQUERY_PREFIXES)
    ])

    # Filter out already processed files
    files_to_process = [f for f in all_files if f not in processed_set]

    total_files = len(files_to_process)
    configs = [(c, o) for c in compilers for o in opt_levels]
    print(f"[*] Total files remaining: {total_files}. Chunk size: {CHUNK_SIZE}")
    print(f"[*] Compiler configs: {[f'{c}_O{o}' for c, o in configs]}")

    # Process in chunks
    for i in range(0, total_files, CHUNK_SIZE):
        chunk = files_to_process[i : i + CHUNK_SIZE]
        print(f"\n{'='*60}")
        print(f"[*] Processing Chunk {i//CHUNK_SIZE + 1} ({len(chunk)} files)")
        print(f"[*] Files: {chunk}")
        print(f"{'='*60}\n")

        clean_directory(DIR_ZIPPED) # Ensure clean state
        clean_directory(DIR_UNZIPPED)
        clean_directory(DIR_C_COMPILE)
        clean_directory(DIR_COMPILED_CHUNK)

        # --- Move Chunk to ZIPPED ---
        for file_name in chunk:
            src = os.path.join(input_dir, file_name)
            dst = os.path.join(DIR_ZIPPED, file_name)
            shutil.copy2(src, dst)

        # --- Unzip & Reconstruct (DGithubJSON2FILE) ---
        success = run_step(SCRIPT_UNZIP, [
            "--zipped-path", DIR_ZIPPED,
            "--source-path", DIR_UNZIPPED,
            "--target-path", DIR_C_COMPILE,
            "--number-of-processes", "4"
        ])
        if not success:
            print("[!] Failure in Unzip/Reconstruct step. Stopping.")
            break

        clean_directory(DIR_UNZIPPED)
        clean_directory(DIR_ZIPPED)

        filter_repos(DIR_C_COMPILE)

        # Check if C_COMPILE is empty after filtering
        if not os.listdir(DIR_C_COMPILE):
            print("[!] No valid source files to compile after filtering. Skipping chunk.")
            state["processed_files"].extend(chunk)
            save_state(state)
            continue

        # --- Compile & link with each compiler×optimization config ---
        # SH2O modifies source files in-place (preprocessor, compiler comments),
        # so we need a fresh copy of the source for each config.
        for idx, (compiler, opt_level) in enumerate(configs):
            if idx == len(configs) - 1:
                # Last config can use the original source directory
                compile_and_link_config(compiler, opt_level, DIR_C_COMPILE, DIR_COMPILED_CHUNK)
            else:
                # Make a copy of the source for this config
                config_source = f"{DIR_C_COMPILE}_{compiler}_O{opt_level}"
                if os.path.exists(config_source):
                    shutil.rmtree(config_source)
                shutil.copytree(DIR_C_COMPILE, config_source)
                compile_and_link_config(compiler, opt_level, config_source, DIR_COMPILED_CHUNK)
                shutil.rmtree(config_source)

        # --- Archive results ---
        print("[*] Creating and compressing archives...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Delete empty directories from COMPILED_CHUNK to avoid bloated archives
        subprocess.run(f"find {DIR_COMPILED_CHUNK}/. -type d -empty -delete", shell=True)

        if not os.listdir(DIR_COMPILED_CHUNK):
            print("[!] No compiled output generated for this chunk. Skipping archiving.")
            state["processed_files"].extend(chunk)
            save_state(state)
            continue

        def _sanitize_name(name):
            """Build a safe archive name from the chunk input filenames (strip extensions and sanitize)"""
            return "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)

        # Use the original chunk file names
        chunk_basenames = [os.path.splitext(fn)[0] for fn in chunk]
        # Strip common prefixes
        for prefix in BIGQUERY_PREFIXES:
            chunk_basenames = [b.replace(prefix + "_", "") for b in chunk_basenames]
        safe_parts = [_sanitize_name(b) for b in chunk_basenames]
        joined = "_".join(safe_parts) if safe_parts else f"chunk_{i//CHUNK_SIZE+1}"

        # Create one archive per config subdirectory
        for config_dir in os.listdir(DIR_COMPILED_CHUNK):
            config_path = os.path.join(DIR_COMPILED_CHUNK, config_dir)
            if not os.path.isdir(config_path) or not os.listdir(config_path):
                continue

            archive_name = f"{config_dir}_{joined}_{timestamp}.tar.gz"
            archive_path = os.path.join(output_dir, archive_name)

            try:
                os.makedirs(output_dir, exist_ok=True)

                with tarfile.open(archive_path, "w:gz") as tar:
                    for entry in os.listdir(config_path):
                        entry_path = os.path.join(config_path, entry)
                        tar.add(entry_path, arcname=entry)

                if os.path.exists(archive_path):
                    size_bytes = os.path.getsize(archive_path)
                    print(f"[*] Archive created: {archive_path} ({size_bytes} bytes)")
                else:
                    print(f"[!] Archive creation failed: {archive_path} not found.")

            except Exception as e:
                print(f"[!] Failed to create archive {archive_path}: {e}")

        # --- Cleanup & State Update ---
        print("[*] Chunk complete. Cleaning intermediate files...")
        clean_directory(DIR_COMPILED_CHUNK)
        clean_directory(DIR_C_COMPILE)

        state["processed_files"].extend(chunk)
        save_state(state)
        print(f"[*] State saved. {len(state['processed_files'])} files total processed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Automates the compilation pipeline for C/C++ source code repositories extracted from GitHub JSON exports")

    parser.add_argument('--dest', type=str, default=DIR_COMPILED, help='Output directory path for compressed chunks of binary and object files (default: /mnt/storage-box/COMPILED).')
    parser.add_argument('--source', type=str, default=RAW_DATA_DIR,
                        help='Path to directory that contains compressed GitHub JSON files from Google BigQuery (default: /mnt/storage-box/github_c).')
    parser.add_argument('--compilers', type=str, nargs='+', default=COMPILERS, choices=['gcc', 'clang'],
                        help='Compilers to use (default: gcc clang).')
    parser.add_argument('--optimizations', type=str, nargs='+', default=OPTIMIZATION_LEVELS,
                        help='Optimization levels to compile with (default: 0 1 2 3).')
    # Parsing the arguments
    args = parser.parse_args()
    main(input_dir=args.source, output_dir=args.dest, compilers=args.compilers, opt_levels=args.optimizations)
