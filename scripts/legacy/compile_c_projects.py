"""
Script to compile C projects extracted from GitHub into ELF binaries with debug info.

Use this script after running scripts/old/DGithubJSON2FILE.py to compile the extracted C source code.
The extracted data by DGithubJSON2FILE.py should be placed in the 'data/extracted_sources' directory.
"""

import os
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SOURCE_CODE_ROOT = Path("data/extracted_sources")
OUTPUT_BIN_DIR = Path("data/train")

def compile_c_project(project_path: Path):
    """
    Attempts to compile C files in a given project directory into an ELF binary.
    """
    c_files = list(project_path.glob('**/*.c'))
    if not c_files:
        logger.debug(f"No .c files found in {project_path}")
        return None

    binary_name = project_path.name + "_compiled"
    output_path = OUTPUT_BIN_DIR / binary_name

    # GCC parameters
    # -g: Include DWARF debug information
    # -o: Output file name
    # -std=gnu11: Use C11 standard with GNU extensions
    # -l<lib>: Link with common libraries (like math library)

    # TODO: It's probably better to use the scripts from the previous group. (SH2O.py, ObjectToBinary.py)
    # TODO: Otherwise, try to parse & run Makefiles instead. Would be more reliable.
    
    # Collect all .c files and .h files
    all_source_files = [str(f) for f in c_files]
    # No need to explicitly add headers to gcc command unless they are not in the same dir as c files.
    # GCC will find them if they are referenced by the .c files via #include.

    command = [
        "gcc",
        "-g",
        "-std=gnu11",
        *all_source_files,
        "-o",
        str(output_path),
        # Add common libraries that C projects often need
        "-lm", # Math library
        "-lpthread" # Pthread library
    ]

    try:
        # Run gcc
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=3600)
        logger.info(f"Successfully compiled {project_path.name} to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.warning(f"Compilation failed for {project_path.name}:")
        logger.warning(f"  Command: {' '.join(e.cmd)}")
        logger.warning(f"  Stdout: {e.stdout.strip()}")
        logger.warning(f"  Stderr: {e.stderr.strip()}")
    except subprocess.TimeoutExpired:
        logger.warning(f"Compilation timed out for {project_path.name}")
    except Exception as e:
        logger.warning(f"Exception raised during compilation of {project_path.name}: {e}")
    return None


def main():
    if not SOURCE_CODE_ROOT.exists():
        logger.error(f"Source code root directory '{SOURCE_CODE_ROOT}' not found. Run DGithubJSON2FILE.py first")
        return

    OUTPUT_BIN_DIR.mkdir(parents=True, exist_ok=True)
    
    compiled_count = 0
    failed_count = 0

    for project_dir in SOURCE_CODE_ROOT.iterdir():
        if project_dir.is_dir():
            logger.info(f"Processing project: {project_dir.name}")
            compiled_bin = compile_c_project(project_dir)
            if compiled_bin:
                compiled_count += 1
            else:
                failed_count += 1
    
    logger.info(f"\nCompleted. Results: {compiled_count} projects compiled. {failed_count} projects failed")


if __name__ == '__main__':
    main()
