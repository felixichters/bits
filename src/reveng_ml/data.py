"""
Data loading and preprocessing for the RevEng project.
"""
from pathlib import Path
from io import BytesIO
import subprocess
from tempfile import TemporaryDirectory
from random import shuffle
import torch
from torch.utils.data import Dataset
from elftools.elf.elffile import ELFFile
from elftools.common.exceptions import ELFError
from elftools.dwarf import callframe
import os
import pickle
import tqdm
import numpy 

def strip_elf_debug_sections(file_path: Path, output_path: Path):
    """
    Strips debug sections from an ELF file using 'strip' CLI tool.

    Args:
        file_path (Path): Input ELF file path.
        output_path (Path): Output ELF file path without debug sections.
    """
    try:
        subprocess.run(['strip', '--strip-debug', '-o', str(output_path), str(file_path)],
                       check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error stripping debug sections from {file_path}: {e.stderr.decode().strip()}")
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
    file_bytes = b''
    try:
        with open(file_path, 'rb') as f:
            file_bytes = f.read()

        with BytesIO(file_bytes) as stream:
            elffile = ELFFile(stream)

            # Build a map from virtual address to file offset from program segments
            # This is needed because function addresses are given in virtual addresses
            va_to_offset_map = []
            for segment in elffile.iter_segments():
                if segment['p_type'] == 'PT_LOAD':
                    va_to_offset_map.append({
                        'vaddr_start': segment['p_vaddr'],
                        'vaddr_end': segment['p_vaddr'] + segment['p_memsz'],
                        'offset': segment['p_offset']
                    })

            # Access .eh_frame section
            dwarf_info = elffile.get_dwarf_info()
            if not dwarf_info:
                print(f"No DWARF info found in {file_path}, cannot get CFI entries.")
                return {}

            cfi_entries = dwarf_info.EH_CFI_entries()
            for entry in cfi_entries:
                # Find all FDE (Frame Descriptor Entries) which correspond to functions
                if not isinstance(entry, callframe.FDE):
                    continue
                
                func_va = entry.header['initial_location']
                func_size = entry.header['address_range']

                # Map virtual address to file offset
                file_offset = None
                for mapping in va_to_offset_map:
                    if mapping['vaddr_start'] <= func_va < mapping['vaddr_end']:
                        offset_in_segment = func_va - mapping['vaddr_start']
                        file_offset = mapping['offset'] + offset_in_segment
                        break
                
                if file_offset is not None and 0 <= file_offset < len(file_bytes):
                    boundaries[file_offset] = func_size

            return boundaries

    except ELFError as e:
        print(f"Error processing ELF file {file_path}: {e}")
        return {}


class BinaryChunkDataset(Dataset):
    """
    PyTorch Dataset for binary files.

    This dataset represents binary files from a directory, and their extracted function boundaries
    to create labels, and provides chunks of the binary and corresponding labels for training.
    """
    def __init__(self, data_path: Path, chunk_size=510, stride=255, randomizeFileOrder=True, onlyIncludeCodeSegment=True):
        """
        Args:
            data_path (Path): Directory containing the *unstripped* binary files or path to dataset file.
            chunk_size (int): The size of each data chunk.
            stride (int): The step size to move when creating overlapping chunks.
        """
        if data_path.is_file():
            try:
                with open(data_path,"rb") as f:
                    dataset = pickle.load(f)
                    self.data_path = dataset[0]
                    self.chunk_size = dataset[1]
                    self.stride = dataset[2]
                    self.onlyDotText = dataset[3]
                    self.files = dataset[4]
                with open(str(data_path) + ".np","rb") as f:
                    self.chunks = [(torch.tensor(label_data_pair[0]),torch.tensor(label_data_pair[1])) for label_data_pair in numpy.load(f)]
            except Exception as e:
                print(e)
                raise e
        else:
            self.data_path = data_path
            self.chunk_size = chunk_size
            self.stride = stride
            self.chunks: list[tuple[torch.Tensor, torch.Tensor]] = []
            self.onlyDotText = onlyIncludeCodeSegment
            
            self.files = []
            print("Scanning input files for dataset in directory:", data_path)
            lst = os.listdir(data_path)
            number_files = len(lst)
            print(f"Found {number_files} files. Creating dataset...")

            for f in data_path.iterdir():
                if f.is_file():
                    self.files.append(f)
                    # Skip ELF validation for performance reasons
                    #try:
                    #    with open(f, 'rb') as a_file:
                    #        # Check for ELF magic number
                    #        if a_file.read(4) == b'\x7fELF':
                    #            self.files.append(f)
                    #except IOError:
                    #    pass # Ignore files we can't read
            
            # Randomize file order
            if randomizeFileOrder:
                shuffle(self.files)
            self._create_chunks()
 
    def _create_chunks(self):
        """
        Pre-chunks all binaries and stores them in memory.
        """
        #progress = tqdm.tqdm(range(len(self.files)))
        for file_path in tqdm.tqdm(self.files):
            #print(f"Processing {file_path.name}...")
            
            # Extract boundaries from the unstripped binary
            boundaries = get_function_boundaries_from_elf(file_path)
            
            # Strip debug sections and read stripped bytes
            if self.onlyDotText == False:
                with TemporaryDirectory() as tmpdir:
                    stripped_path = Path(tmpdir) / "stripped_binary"
                    strip_elf_debug_sections(file_path, stripped_path)
                    with open(stripped_path, 'rb') as f:
                        stripped_file_bytes = f.read()
            else:
                stripped_file_bytes = b''
                try:
                    with open(file_path, 'rb') as f:
                        file_bytes = f.read()
                    with BytesIO(file_bytes) as stream:
                        elffile = ELFFile(stream)
                        for section in elffile.iter_sections():
                            if section.name.startswith(".text"):
                                textSectionOffset = section.header['sh_offset']
                                stripped_file_bytes = section.data()
                except Exception as e:
                    raise e
            #progress.iter()
            
            if not stripped_file_bytes or not boundaries:
                print(f"Skipping {file_path.name}: No valid boundaries or bytes found.")
                continue

            # Create a label vector for the entire file
            labels = torch.zeros(len(stripped_file_bytes), dtype=torch.long)
            for (offset, size) in boundaries.items():
                if offset - textSectionOffset < len(labels):
                    labels[offset - textSectionOffset] = 1 # Mark 'B-FUNC' (Beginning of a function)
                if offset - textSectionOffset + size - 1 < len(labels):
                    labels[offset - textSectionOffset + size - 1] = 2 # Mark 'E-FUNC' (End of a function)

            # Create overlapping chunks from the unstripped bytes
            # TODO: maybe a better approach could be used here?
            for i in range(0, len(stripped_file_bytes) - self.chunk_size + 1, self.stride):
                chunk_bytes_raw = stripped_file_bytes[i:i + self.chunk_size]
                chunk_labels = labels[i:i + self.chunk_size]

                chunk_tensor = torch.tensor([b for b in chunk_bytes_raw], dtype=torch.long)
                self.chunks.append((chunk_tensor, chunk_labels))

            #print(f"Chunked {file_path.name} into {len(stripped_file_bytes) // self.stride} chunks.")
        print(f"Chunked files into {len(self.chunks)} chunks with {self.chunk_size} chunks and {self.stride} stride.(only using text section: {self.onlyDotText})")
 
    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.chunks[idx]

    def save(self, result_path: Path):
        try:
            with open(result_path,"wb") as f:
                pickle.dump([self.data_path, self.chunk_size, self.stride, self.onlyDotText, self.files],f)
            with open(str(result_path) + ".np","wb") as f:
                numpy.save(f,self.chunks)
        except Exception as e:
            print(e)
            raise e
