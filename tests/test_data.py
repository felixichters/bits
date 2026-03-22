
import pytest
from pathlib import Path
import torch
import subprocess
import tempfile
from reveng_ml.data import (
    get_function_boundaries_from_elf,
    get_instruction_boundaries_from_elf,
    BinaryChunkDataset,
    strip_elf_debug_sections,
    split_dataset_files, run_strip_command,
)


@pytest.fixture(scope="module")
def sample_binary(tmp_path_factory):
    """
    Compiles a sample C file and returns the path to the unstripped binary.
    Uses tmp_path_factory so the directory persists for the lifetime of the module.
    """
    c_code = (
        '#include <stdio.h>\n'
        '\n'
        'void test() {\n'
        '    printf("This is a test function.\\n");\n'
        '}\n'
        '\n'
        'int main() {\n'
        '    printf("Hello!\\n");\n'
        '    test();\n'
        '    return 0;\n'
        '}\n'
    )
    tmp_dir = tmp_path_factory.mktemp("sample_binary")
    c_file = tmp_dir / "test.c"
    unstripped_binary = tmp_dir / "test.unstripped"

    with open(c_file, "w") as f:
        f.write(c_code)

    compile_command = ["gcc", "-o", str(unstripped_binary), str(c_file)]
    subprocess.run(compile_command, check=True)
    c_file.unlink()

    yield unstripped_binary


def disassemble_function_content(boundaries, binary):
    """
    Disassembles and prints the content of each function given its boundaries using objdump.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        for (offset, size) in boundaries.items():
            with open(Path(tmpdir) / str(offset), 'wb') as f:
                f.write(binary[offset:offset + size])

            try:
                disasm_output = subprocess.run(
                    ["objdump", "-b", "binary", "-D", "-m", "i386:x86-64", Path(tmpdir) / str(offset)],
                    capture_output=True, text=True
                ).stdout
            except FileNotFoundError:
                disasm_output = "objdump not installed; cannot disassemble."

            print(f"\n===> Disassembly of function at offset {hex(offset)}:\n")
            print(disasm_output)


def test_get_function_boundaries(sample_binary, capsys):
    """
    Tests that function boundaries are correctly extracted from an unstripped binary.
    """
    boundaries = get_function_boundaries_from_elf(sample_binary)

    with capsys.disabled():
        print()
        for (file_offset, length) in boundaries.items():
            print(f"Function at file offset {hex(file_offset)} with size of {length} bytes")

    with tempfile.TemporaryDirectory() as tmpdir:
        stripped_path = Path(tmpdir) / "stripped_binary"
        strip_elf_debug_sections(sample_binary, stripped_path)
        with open(stripped_path, 'rb') as f, capsys.disabled():
                disassemble_function_content(boundaries, binary=f.read())

    # We expect to find at least two functions: main and test
    assert len(boundaries) >= 2

    for offset, size in boundaries.items():
        assert isinstance(offset, int)
        assert isinstance(size, int)
        assert offset > 0
        assert size > 0

def test_get_function_boundaries_missing_symtab(sample_binary, capsys):
    """
    Tests that function boundaries can still be extracted from an unstripped binary,
    even if the .symtab section is missing. The extractor should fall back to using .eh_frame instead.
    Both methods should return the same function boundaries.
    """
    with(tempfile.NamedTemporaryFile()) as no_symtab_binary:
        run_strip_command(['-R', ".symtab", '-o', no_symtab_binary.name, str(sample_binary)])

        boundaries_eh_frame = get_function_boundaries_from_elf(Path(no_symtab_binary.name))
        boundaries_symtab = get_function_boundaries_from_elf(sample_binary)

        # The function boundaries extracted from the .symtab and .eh_frame should be the same
        assert boundaries_symtab == boundaries_eh_frame

def test_get_instruction_boundaries(sample_binary):
    """Tests that instruction boundaries are extracted from an unstripped binary."""
    inst_starts = get_instruction_boundaries_from_elf(sample_binary)

    # Should find many instruction starts in a compiled binary
    assert len(inst_starts) > 0

    # Every function start should also be an instruction start
    func_boundaries = get_function_boundaries_from_elf(sample_binary)
    from io import BytesIO
    from elftools.elf.elffile import ELFFile

    with open(sample_binary, 'rb') as f:
        file_bytes = f.read()
    with BytesIO(file_bytes) as stream:
        elffile = ELFFile(stream)
        for section in elffile.iter_sections():
            if section.name == ".text":
                text_offset = section.header['sh_offset']
                break

    for func_offset in func_boundaries.keys():
        local = func_offset - text_offset
        if local >= 0:
            assert local in inst_starts, f"Function start at local offset {local} should be an instruction start"


def test_binary_chunk_dataset(sample_binary, capsys):
    """
    Tests the full BinaryChunkDataset pipeline.
    """
    data_path = sample_binary.parent

    with capsys.disabled():
        dataset = BinaryChunkDataset(data_path=data_path, chunk_size=128, stride=64, randomize_file_order=True)

    # Ensure some chunks were created
    assert len(dataset) > 0

    # Check the type of the first chunk and its labels (now 3-tuple)
    chunk, func_label, inst_label = dataset[0]
    assert chunk.dtype == torch.long
    assert func_label.dtype == torch.long
    assert inst_label.dtype == torch.long

    # At least one label in all the chunks should be 1 (B-FUNC)
    found_label = False
    for _, func_label, _ in dataset:
        if 1 in func_label:
            found_label = True
            break
    assert found_label, "No function start label found in any chunk."


def test_binary_chunk_dataset_both_task(sample_binary, capsys):
    """Tests dataset with task='both' returns valid instruction labels."""
    data_path = sample_binary.parent

    with capsys.disabled():
        dataset = BinaryChunkDataset(data_path=data_path, chunk_size=128, stride=64,
                                     randomize_file_order=False, task="both")

    assert len(dataset) > 0

    chunk, func_labels, inst_labels = dataset[0]
    assert chunk.shape == func_labels.shape == inst_labels.shape

    # Instruction labels should have some 1s (instruction starts)
    found_inst = False
    for _, _, inst_labels in dataset:
        if 1 in inst_labels:
            found_inst = True
            break
    assert found_inst, "No instruction start label found in any chunk."


def test_binary_chunk_dataset_instruction_only(sample_binary, capsys):
    """Tests dataset with task='instruction' has instruction labels but zero func labels."""
    data_path = sample_binary.parent

    with capsys.disabled():
        dataset = BinaryChunkDataset(data_path=data_path, chunk_size=128, stride=64,
                                     randomize_file_order=False, task="instruction")

    assert len(dataset) > 0
    _, func_labels, inst_labels = dataset[0]

    # func_labels should be all zeros for instruction-only task
    assert func_labels.sum().item() == 0


def test_binary_chunk_dataset_with_padding(sample_binary, capsys):
    """Tests dataset with padding to fill up the last chunk if the binary is smaller than chunk_size."""
    data_path = sample_binary.parent

    with capsys.disabled():
        dataset = BinaryChunkDataset(data_path=data_path, chunk_size=100000, stride=64, randomize_file_order=True)

    # Our sample binary is very small, so we expect only one chunk with padding
    assert len(dataset) == 1

    # Ensure that the last byte of the chunk is padding (0)
    _, func_labels, inst_labels = dataset[0]
    assert func_labels[-1].item() == 0
    assert inst_labels[-1].item() == 0

def test_binary_chunk_dataset_only_text_false(sample_binary, capsys):
    """Tests dataset with only_text=False includes all sections, not just .text."""
    data_path = sample_binary.parent

    with capsys.disabled():
        dataset_full = BinaryChunkDataset(data_path, chunk_size=128, stride=64, only_include_code_segment=False)
        dataset_text = BinaryChunkDataset(data_path, chunk_size=128, stride=64, only_include_code_segment=True)

    # Ensure some chunks were created
    assert len(dataset_full) > 0
    assert len(dataset_text) > 0

    # The full dataset should have more chunks than the text-only dataset
    assert len(dataset_full) > len(dataset_text)

# ---------------------------------------------------------------------------
# Tests for split_dataset_files
# ---------------------------------------------------------------------------

def test_split_dataset_files_counts_sum_to_total(tmp_path):
    """Returns counts that sum to the total number of source files."""
    src = tmp_path / "src"
    src.mkdir()
    for i in range(10):
        (src / f"file_{i}.bin").write_bytes(b"\x00" * 8)

    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"

    counts = split_dataset_files(src, train_dir, test_dir, test_ratio=0.2, seed=42)

    assert counts["train"] + counts["test"] == 10


def test_split_dataset_files_files_moved(tmp_path):
    """Files are physically moved to train/test directories."""
    src = tmp_path / "src"
    src.mkdir()
    for i in range(10):
        (src / f"file_{i}.bin").write_bytes(b"\x00" * 8)

    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"

    counts = split_dataset_files(src, train_dir, test_dir, test_ratio=0.2, seed=42)

    assert len(list(train_dir.iterdir())) == counts["train"]
    assert len(list(test_dir.iterdir())) == counts["test"]
    # Source directory should now be empty
    assert list(src.iterdir()) == []


def test_split_dataset_files_test_ratio_respected(tmp_path):
    """The test split is approximately the requested ratio."""
    src = tmp_path / "src"
    src.mkdir()
    n = 100
    for i in range(n):
        (src / f"file_{i}.bin").write_bytes(b"\x00" * 8)

    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"

    counts = split_dataset_files(src, train_dir, test_dir, test_ratio=0.2, seed=42)

    # Allow +-2 files of tolerance for rounding
    assert abs(counts["test"] - 20) <= 2


def test_split_dataset_files_reproducible(tmp_path):
    """Same seed and same initial file set produces the same counts and filenames."""
    src1 = tmp_path / "src1"
    src2 = tmp_path / "src2"
    src1.mkdir()
    src2.mkdir()
    filenames = [f"file_{i:02d}.bin" for i in range(10)]
    for name in filenames:
        (src1 / name).write_bytes(b"\xAB" * 8)
        (src2 / name).write_bytes(b"\xAB" * 8)

    train1 = tmp_path / "train1"
    test1 = tmp_path / "test1"
    train2 = tmp_path / "train2"
    test2 = tmp_path / "test2"

    counts1 = split_dataset_files(src1, train1, test1, test_ratio=0.3, seed=7)
    counts2 = split_dataset_files(src2, train2, test2, test_ratio=0.3, seed=7)

    # Counts must be identical when seed and number of files are identical
    assert counts1["train"] == counts2["train"]
    assert counts1["test"] == counts2["test"]


@pytest.mark.parametrize("bad_ratio", [0, 1, -0.1, 1.1, -1])
def test_split_dataset_files_invalid_ratio(tmp_path, bad_ratio):
    """Raises ValueError for out-of-range test_ratio values."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "file.bin").write_bytes(b"\x00" * 8)
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"

    with pytest.raises(ValueError):
        split_dataset_files(src, train_dir, test_dir, test_ratio=bad_ratio)


def test_split_dataset_files_empty_directory(tmp_path):
    """Raises ValueError when the source directory is empty."""
    src = tmp_path / "src"
    src.mkdir()
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"

    with pytest.raises(ValueError):
        split_dataset_files(src, train_dir, test_dir, test_ratio=0.2)


# ---------------------------------------------------------------------------
# Tests for BinaryChunkDataset.get_label_counts
# ---------------------------------------------------------------------------

def test_get_label_counts(sample_binary, capsys):
    """get_label_counts returns a tuple of (func_counts[3], inst_counts[2])."""
    data_path = sample_binary.parent

    with capsys.disabled():
        dataset = BinaryChunkDataset(data_path=data_path, chunk_size=128, stride=64, randomize_file_order=False)

    func_counts, inst_counts = dataset.get_label_counts()

    assert func_counts.shape == (3,), f"Expected shape (3,), got {func_counts.shape}"
    assert inst_counts.shape == (2,), f"Expected shape (2,), got {inst_counts.shape}"
    assert (func_counts >= 0).all(), "All func counts must be non-negative"
    assert (inst_counts >= 0).all(), "All inst counts must be non-negative"

    total_tokens = sum(func_label.numel() for _, func_label, _ in dataset)
    assert func_counts.sum().item() == total_tokens


# ---------------------------------------------------------------------------
# Tests for BinaryChunkDataset.save / reload
# ---------------------------------------------------------------------------

def test_dataset_save_load(sample_binary, tmp_path, capsys):
    """Saving and reloading a dataset yields identical length and first chunk."""
    data_path = sample_binary.parent

    with capsys.disabled():
        dataset = BinaryChunkDataset(data_path=data_path, chunk_size=128, stride=64, randomize_file_order=False)

    save_path = tmp_path / "dataset.bin"
    dataset.save(save_path)

    with capsys.disabled():
        reloaded = BinaryChunkDataset(data_path=save_path, chunk_size=128, stride=64, randomize_file_order=False)

    assert len(reloaded) == len(dataset), "Reloaded dataset length must match original"

    orig_chunk, orig_func_label, orig_inst_label = dataset[0]
    reloaded_chunk, reloaded_func_label, reloaded_inst_label = reloaded[0]
    assert torch.equal(orig_chunk, reloaded_chunk), "First chunk bytes must match after reload"
    assert torch.equal(orig_func_label, reloaded_func_label), "First chunk func labels must match after reload"
    assert torch.equal(orig_inst_label, reloaded_inst_label), "First chunk inst labels must match after reload"
