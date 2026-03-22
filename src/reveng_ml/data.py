"""
Data loading and preprocessing for the RevEng project.
"""
from pathlib import Path
from io import BytesIO
import subprocess
from tempfile import TemporaryDirectory
from random import shuffle, Random
import shutil
import torch
from torch.utils.data import Dataset
from elftools.elf.elffile import ELFFile
from elftools.common.exceptions import ELFError
from elftools.dwarf import callframe
import os
import pickle
import tqdm
import numpy
from concurrent.futures import ThreadPoolExecutor, as_completed


def _extract_repo_name(filename: str) -> str:
    """Extract the repo name from a binary filename, stripping compiler config prefixes.

    Examples:
        'gcc_O2_author_repo_executable0' -> 'author_repo'
        'clang_O1_author_repo'           -> 'author_repo'
        'author_repo_executable0'        -> 'author_repo'
        'author_repo'                    -> 'author_repo'
    """
    import re
    name = filename
    # Strip compiler config prefix (e.g., 'gcc_O2_', 'clang_O3_')
    name = re.sub(r'^(?:gcc|clang|g\+\+|clang\+\+)_O[0-3]_', '', name)
    # Strip executable suffix (e.g., '_executable0', '_executable12')
    name = re.sub(r'_executable\d+$', '', name)
    return name


def split_dataset_files(
    src_dir: Path,
    train_dir: Path,
    test_dir: Path,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, int]:
    """
    Splits binary files into train/test directories, grouping by source repo.

    All compiler variants (gcc_O0, clang_O3, etc.) of the same repo go into
    the same split to prevent data leakage.

    Returns:
        dict with keys 'train', 'test' mapping to file counts.
    """
    if test_ratio <= 0 or test_ratio >= 1:
        raise ValueError("test_ratio must be between 0 and 1 (exclusive)")

    files = [f for f in src_dir.iterdir() if f.is_file()]
    if not files:
        raise ValueError(f"No files found in {src_dir}")

    # Group files by repo name
    from collections import defaultdict
    repo_files: dict[str, list[Path]] = defaultdict(list[Path])
    for f in files:
        repo = _extract_repo_name(f.name)
        repo_files[repo].append(f)

    repos = list(repo_files.keys())
    rng = Random(seed)
    rng.shuffle(repos)

    n_repos = len(repos)
    n_test_repos = max(1, round(n_repos * test_ratio))
    n_train_repos = n_repos - n_test_repos

    if n_train_repos <= 0:
        raise ValueError(f"Not enough repos ({n_repos}) for the requested split ratios")

    train_repos = repos[:n_train_repos]
    test_repos = repos[n_train_repos:]

    splits = {
        "train": (train_dir, [f for r in train_repos for f in repo_files[r]]),
        "test":  (test_dir,  [f for r in test_repos  for f in repo_files[r]]),
    }

    counts: dict[str, int] = {}
    for split_name, (out_dir, split_files) in splits.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in split_files:
            shutil.move(str(f), str(out_dir / f.name))
        counts[split_name] = len(split_files)

    print(f"Split {n_repos} repos: {n_train_repos} train, {n_test_repos} test")
    print(f"Files: {counts['train']} train, {counts['test']} test")

    return counts

def strip_elf_debug_sections(file_path: Path, output_path: Path):
    """
    Strips debug sections from an ELF file using 'strip' CLI tool.

    Args:
        file_path (Path): Input ELF file path.
        output_path (Path): Output ELF file path without debug sections.
    """
    run_strip_command(['--strip-debug', '-o', str(output_path), str(file_path)])

def run_strip_command(args: list[str]):
    """
    Helper function to run the external 'strip' command with specified arguments and print errors.
    """
    try:
        subprocess.run(['strip'] + args, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error stripping symbols '{args}': {e.stderr.decode().strip()}")
        raise
    except FileNotFoundError:
        print("The 'strip' command-line tool is not installed on this system.")
        raise

def get_function_boundaries_from_elf(file_path: Path) -> dict[int, int]:
    """
    Parses an ELF file to extract function boundaries from the .eh_frame section.

    Args:
        file_path (Path): The path to the ELF file.

    Returns:
        A dictionary mapping function start file offsets to their sizes.
    """

    boundaries: dict[int, int] = {}
    try:
        with open(file_path, 'rb') as f:
            file_bytes = f.read()

        with (BytesIO(file_bytes) as stream):
            elffile = ELFFile(stream)

            va_to_offset_map = []
            for segment in elffile.iter_segments():
                if segment['p_type'] == 'PT_LOAD':
                    va_to_offset_map.append({
                        'vaddr_start': segment['p_vaddr'],
                        'vaddr_end': segment['p_vaddr'] + segment['p_memsz'],
                        'offset': segment['p_offset']
                    })

            def va_to_file_offset(va):
                for mapping in va_to_offset_map:
                    if mapping['vaddr_start'] <= va < mapping['vaddr_end']:
                        return mapping['offset'] + (va - mapping['vaddr_start'])
                return None

            # .symtab
            symtab = elffile.get_section_by_name('.symtab')
            if symtab:
                for sym in symtab.iter_symbols():
                    if (sym['st_info']['type'] == 'STT_FUNC'
                            and sym['st_size'] > 0
                            and sym['st_value'] != 0):
                        file_offset = va_to_file_offset(sym['st_value'])
                        if file_offset is not None and 0 <= file_offset < len(file_bytes) and \
                             (file_offset not in boundaries or sym['st_size'] > boundaries[file_offset]):
                                boundaries[file_offset] = sym['st_size']

            # .eh_frame fallback
            if not boundaries:
                dwarf_info = elffile.get_dwarf_info()
                if dwarf_info:
                    for entry in dwarf_info.EH_CFI_entries():
                        if not isinstance(entry, callframe.FDE):
                            continue
                        func_va = entry.header['initial_location']
                        func_size = entry.header['address_range']
                        # Skip zero-size functions
                        if func_size == 0:
                            continue
                        file_offset = va_to_file_offset(func_va)
                        if file_offset is not None and 0 <= file_offset < len(file_bytes) and \
                             (file_offset not in boundaries or func_size > boundaries[file_offset]):
                                boundaries[file_offset] = func_size

            if not boundaries:
                print(f"Warning: No function symbols found in {file_path.name} via .symtab or .eh_frame")

            return boundaries

    except ELFError as e:
        print(f"Error processing ELF file {file_path}: {e}")
        return {}


def get_instruction_boundaries_from_elf(file_path: Path, arch: str = "x86_64") -> set[int]:
    """
    Disassembles the .text section to find instruction start offsets.

    Args:
        file_path (Path): The path to the ELF file.
        arch (str): Architecture: "x86_64", "x86_32", or "arm".

    Returns:
        A set of local offsets (relative to .text start) where instructions begin.
    """
    from capstone import Cs, CS_ARCH_X86, CS_MODE_64, CS_MODE_32, CS_ARCH_ARM, CS_MODE_ARM

    try:
        with open(file_path, 'rb') as f:
            file_bytes = f.read()

        with BytesIO(file_bytes) as stream:
            elffile = ELFFile(stream)
            text_section = None
            for section in elffile.iter_sections():
                if section.name == ".text":
                    text_section = section
                    break

            if text_section is None:
                return set()

            text_bytes: bytes = text_section.data()
            text_vaddr: int = text_section.header['sh_addr']

            if arch == "x86_64":
                md = Cs(CS_ARCH_X86, CS_MODE_64)
            elif arch == "x86_32":
                md = Cs(CS_ARCH_X86, CS_MODE_32)
            elif arch == "arm":
                md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
            else:
                raise ValueError(f"Unsupported architecture: {arch}")

            inst_starts = set[int]()
            for insn in md.disasm(text_bytes, text_vaddr):
                local_offset = insn.address - text_vaddr
                inst_starts.add(local_offset)

            return inst_starts

    except ELFError as e:
        print(f"Error processing ELF file {file_path}: {e}")
        return set()


class BinaryChunkDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    PyTorch Dataset for binary files.

    This dataset represents binary files from a directory, and their extracted function boundaries
    to create labels, and provides chunks of the binary and corresponding labels for training.
    """
    def __init__(self, data_path: Path, chunk_size=510, stride=255, randomize_file_order=True,
                 only_include_code_segment=True, for_evaluation=False, task="both", arch="x86_64"):
        """
        Args:
            data_path (Path): Directory containing the *unstripped* binary files or path to dataset file.
            chunk_size (int): The size of each data chunk.
            stride (int): The step size to move when creating overlapping chunks.
            for_evaluation (bool): Use non-overlapping Chunks
            task (str): "function", "instruction", or "both"
            arch (str): Architecture for instruction disassembly: "x86_64", "x86_32", "arm"
        """
        self.task = task
        self.arch = arch

        if data_path.is_file():
            try:
                with open(data_path,"rb") as f:
                    dataset = pickle.load(f)
                    self.data_path = dataset[0]
                    self.chunk_size = dataset[1]
                    self.stride = dataset[2]
                    self.only_dot_text = dataset[3]
                    self.files = dataset[4]
                    if len(dataset) >= 6:
                        self.task = dataset[5]
                    if len(dataset) >= 7:
                        self.arch = dataset[6]
                if for_evaluation and self.stride != self.chunk_size:
                    raise ValueError(
                        f"Dataset file '{data_path}' was created with stride={self.stride} (overlapping), "
                        f"but for_evaluation=True requires non-overlapping chunks (stride=chunk_size={self.chunk_size}). "
                        f"Pass the source binary directory '{self.data_path}' instead."
                    )
                with open(str(data_path) + ".np","rb") as f:
                    loaded = numpy.load(f, allow_pickle=True)
                    if len(loaded[0]) == 2: # pragma: deprecated
                        # Old format: (data, func_labels) — add zero inst_labels
                        self.chunks = [
                            (torch.tensor(pair[0]), torch.tensor(pair[1]),
                             torch.zeros(len(pair[1]), dtype=torch.long))
                            for pair in loaded
                        ]
                    else:
                        self.chunks = [
                            (torch.tensor(triple[0]), torch.tensor(triple[1]), torch.tensor(triple[2]))
                            for triple in loaded
                        ]
            except Exception as e:
                print(e)
                raise e
        else:
            self.data_path = data_path
            self.chunk_size = chunk_size
            self.stride = chunk_size if for_evaluation else stride
            self.chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            self.only_dot_text = only_include_code_segment

            self.files: list[Path] = []
            print("Scanning input files for dataset in directory:", data_path)
            lst = os.listdir(data_path)
            number_files = len(lst)
            print(f"Found {number_files} files. Creating dataset...")

            for f in data_path.iterdir():
                if f.is_file():
                    self.files.append(f)

            # Randomize file order
            if randomize_file_order:
                shuffle(self.files)
            self._create_chunks()

    def _create_chunks_for_file_threaded(self, file_path: Path) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        local_chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        # Extract boundaries from the unstripped binary
        boundaries = get_function_boundaries_from_elf(file_path)

        # Extract instruction boundaries if needed
        inst_starts = set[int]()
        if self.task in ("instruction", "both"):
            inst_starts = get_instruction_boundaries_from_elf(file_path, arch=self.arch)

        # Strip debug sections and read stripped bytes
        text_section_offset = 0
        text_section_size = 0

        if not self.only_dot_text:
            with TemporaryDirectory() as tmpdir:
                stripped_path = Path(tmpdir) / "stripped_binary"
                strip_elf_debug_sections(file_path, stripped_path)
                with open(stripped_path, 'rb') as f:
                    stripped_file_bytes = f.read()
                text_section_size = len(stripped_file_bytes)
        else:
            stripped_file_bytes = b''

            try:
                with open(file_path, 'rb') as f:
                    file_bytes = f.read()
                with BytesIO(file_bytes) as stream:
                    elffile = ELFFile(stream)
                    for section in elffile.iter_sections():
                        if section.name == ".text":
                            text_section_offset = section.header['sh_offset']
                            text_section_size = section.header['sh_size']
                            stripped_file_bytes = section.data()
                            break
            except Exception as e:
                raise e

        if not stripped_file_bytes: # pragma: no cover
            print(f"Skipping {file_path.name}: No .text section or bytes found.")
            return []

        if not boundaries and self.task in ("function", "both"): # pragma: no cover
            print(f"Skipping {file_path.name}: No valid function boundaries found.")
            return []

        if not inst_starts and self.task == "instruction": # pragma: no cover
            print(f"Skipping {file_path.name}: No valid instruction boundaries found.")
            return []

        # Create function boundary labels
        func_labels = torch.zeros(len(stripped_file_bytes), dtype=torch.long)

        if self.task in ("function", "both"):
            # Mark all E-FUNC endings first
            for (offset, size) in boundaries.items():
                end_offset = offset + size - 1
                if text_section_offset <= end_offset < text_section_offset + text_section_size:
                    local_end = end_offset - text_section_offset
                    if local_end < len(func_labels):
                        func_labels[local_end] = 2  # E-FUNC

            # Mark all B-FUNC starts second
            for (offset, size) in boundaries.items():
                if text_section_offset <= offset < text_section_offset + text_section_size:
                    local_offset = offset - text_section_offset
                    if local_offset < len(func_labels):
                        func_labels[local_offset] = 1  # B-FUNC

        # Create instruction boundary labels
        inst_labels = torch.zeros(len(stripped_file_bytes), dtype=torch.long)

        if self.task in ("instruction", "both"):
            for local_offset in inst_starts:
                if 0 <= local_offset < len(inst_labels):
                    inst_labels[local_offset] = 1  # instruction start

        # Pad files shorter than chunk_size so they contribute one full chunk
        if len(stripped_file_bytes) < self.chunk_size:
            pad_len = self.chunk_size - len(stripped_file_bytes)
            stripped_file_bytes = stripped_file_bytes + bytes(pad_len)
            func_labels = torch.cat([func_labels, torch.zeros(pad_len, dtype=torch.long)])
            inst_labels = torch.cat([inst_labels, torch.zeros(pad_len, dtype=torch.long)])

        # Create overlapping chunks from the bytes
        for i in range(0, len(stripped_file_bytes) - self.chunk_size + 1, self.stride):
            chunk_bytes_raw = stripped_file_bytes[i:i + self.chunk_size]
            chunk_func_labels = func_labels[i:i + self.chunk_size]
            chunk_inst_labels = inst_labels[i:i + self.chunk_size]
            chunk_tensor = torch.tensor([b for b in chunk_bytes_raw], dtype=torch.long)
            local_chunks.append((chunk_tensor, chunk_func_labels, chunk_inst_labels))

        # Emit one final chunk anchored at the end (skip in non-overlapping mode to avoid double-counting)
        if len(stripped_file_bytes) > self.chunk_size and self.stride < self.chunk_size:
            last_aligned_start = ((len(stripped_file_bytes) - self.chunk_size) // self.stride) * self.stride
            final_start = len(stripped_file_bytes) - self.chunk_size
            if final_start > last_aligned_start:
                chunk_bytes_raw = stripped_file_bytes[final_start:]
                chunk_func_labels = func_labels[final_start:]
                chunk_inst_labels = inst_labels[final_start:]
                chunk_tensor = torch.tensor([b for b in chunk_bytes_raw], dtype=torch.long)
                local_chunks.append((chunk_tensor, chunk_func_labels, chunk_inst_labels))

        return local_chunks


    def _create_chunks(self):
        """
        Pre-chunks all binaries and stores them in memory.
        """
        import multiprocessing


        self.chunks = []

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()*9) as executor:
            futures = [
                executor.submit(self._create_chunks_for_file_threaded, file_path)
                for file_path in self.files
            ]

            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                try:
                    self.chunks.extend(future.result())
                except Exception as e:
                    print(f"Warning: skipping file due to error: {e}")

        print(f"Chunked files into {len(self.chunks)} chunks with {self.chunk_size}-sized chunks and "
              f"{self.stride} stride. (only using text section: {self.only_dot_text})")

    def get_label_counts(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (func_counts[3], inst_counts[2]) tensors of per-class label counts."""
        func_counts = torch.zeros(3, dtype=torch.long)
        inst_counts = torch.zeros(2, dtype=torch.long)
        for _, func_labels, inst_labels in self.chunks:
            for c in range(3):
                func_counts[c] += (func_labels == c).sum()
            for c in range(2):
                inst_counts[c] += (inst_labels == c).sum()
        return func_counts, inst_counts

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.chunks[idx]

    def save(self, result_path: Path):
        try:
            with open(result_path,"wb") as f:
                pickle.dump(
                    [self.data_path, self.chunk_size, self.stride, self.only_dot_text, self.files, self.task, self.arch],
                    f
                )
            with open(str(result_path) + ".np","wb") as f:
                numpy.save(f, self.chunks)
        except Exception as e:
            print(e)
            raise e
