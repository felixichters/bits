from pathlib import Path
from io import BytesIO
import subprocess
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import Dataset
from elftools.elf.elffile import ELFFile
from elftools.common.exceptions import ELFError
from elftools.dwarf import callframe

"""
import sys
#import data module from main project
sys.path.append("../../")
import data
"""


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

def get_function_boundaries_from_elf(file_path: Path):
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



class XDAChunkDataset(Dataset):
    """
    PyTorch Dataset for binary files.

    This dataset reads binary files from a directory, extracts function boundaries
    to create labels, and provides chunks of the binary and corresponding labels.
    Used to evaluate the XDA-Model, therefore independent Class from BinaryChunkDataset 
    to allow changes to that class without affecting XDA.
    
    """
    def __init__(self, data_dir: Path, chunk_size=510, stride=255):
        """
        Args:
            data_dir (Path): Directory containing the *unstripped* binary files.
            chunk_size (int): The size of each data chunk. maximum of 510 allowed
            stride (int): The step size to move when creating overlapping chunks.
        """
        if chunk_size > 510:
            raise Exception("invalid chunk size for XDA: " + str(chunk_size))
        
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.stride = stride
        self.chunks: list[tuple[torch.Tensor, torch.Tensor]] = []

        self.files = []
        for f in self.data_dir.iterdir():
            if f.is_file():
                try:
                    with open(f, 'rb') as a_file:
                        # Check for ELF magic number
                        if a_file.read(4) == b'\x7fELF':
                            self.files.append(f)
                except IOError:
                    pass # Ignore files we can't read
        
        self._create_chunks()

    def _create_chunks(self):
        """
        Pre-chunks all binaries and stores them in memory.
        """
        for file_path in self.files:
            print(f"Processing {file_path.name}...")
            
            # Extract boundaries from the unstripped binary
            boundaries = get_function_boundaries_from_elf(file_path)
            
            # Strip debug sections and read stripped bytes
            with TemporaryDirectory() as tmpdir:
                stripped_path = Path(tmpdir) / "stripped_binary"
                strip_elf_debug_sections(file_path, stripped_path)
                with open(stripped_path, 'rb') as f:
                    stripped_file_bytes = f.read()
            
            if not stripped_file_bytes or not boundaries:
                print(f"Skipping {file_path.name}: No valid boundaries or bytes found.")
                continue

            # Create a label vector for the entire file
            labels = torch.zeros(len(stripped_file_bytes), dtype=torch.long)
            for (offset, size) in boundaries.items():
                if offset < len(labels):
                    labels[offset] = 1 # Mark 'B-FUNC' (Beginning of a function)
                if offset + size - 1 < len(labels):
                    labels[offset + size - 1] = 2 # Mark 'E-FUNC' (End of a function)

            # Create overlapping chunks from the unstripped bytes
            # TODO: maybe a better approach could be used here?
            for i in range(0, len(stripped_file_bytes) - self.chunk_size + 1, self.stride):
                chunk_bytes_raw = stripped_file_bytes[i:i + self.chunk_size]
                chunk_labels = labels[i:i + self.chunk_size]

                chunk_tensor = torch.tensor([b for b in chunk_bytes_raw], dtype=torch.long)
                self.chunks.append((chunk_tensor, chunk_labels))

            print(f"Chunked {file_path.name} into {len(stripped_file_bytes) // self.stride} chunks.")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int):
        return self.chunks[idx]
        
    def toFile(path):
        """
        saves 
        """

def read_result_files_XDA_original(path):
    if os.path.is_dir(path):
        files = os.listdir(path)
    else:
        files = [path]

    labels = []
    
    for filepath in files:
        f = open(f'{filepath}', 'r')
        labels_file = []
        for line in f:
            line_split = line.strip().split()
            if line_split[1] == 'F':
                labels_file.append(1)
            elif line_split[1] == 'R':
                labels_file.append(2)
            elif line_split[1] == '-':
                labels_file.append(0)
            else:
                raise Exception("invalid input symbol in \"read__files\"")
        labels.append(labels_file)
        f.close()
        
    return torch.tensor(labels)
    

def eval_results_F1(results,truths):
    for result,truth in zip(results,truths):
        for res_elem,truth_elem in zip(result,truth):
            if res_elem == truth_elem:
                if res_elem == 1 or res_elem == 2:
                    TP += 1
                elif res_elem == 0:
                    TN += 1
            elif res_elem == 2 or res_elem == 1:
                FP += 1
            else:
                FN += 1
                
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    return F1

from fairseq.models.roberta import RobertaModel
import torch
from collections import defaultdict
import pickle

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def infer_XDA(dataset):
    """
    infer result(s) by using XDA-model.
    Params:
        path: the path of a file to infer or a folder whose files get inferred
    Returns: Tensor in the form: [file1Result,file2Result,...] where each result is a tensor with one value (start of function, end of function or neither) per inferred byte 
    """
    
    
    
    xda = RobertaModel.from_pretrained('checkpoints/finetune_msvs_funcbound_64', 'checkpoint_best.pt',
                                       'data-bin/funcbound_msvs_64', bpe=None, user_dir='finetune_tasks')
    
    xda.eval()
    
    print("Starting XDA inference...")
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(DataLoader(dataset, batch_size=1, shuffle=False), desc="Inferring", leave=False)

    with torch.no_grad():
        for batch_data, batch_labels in progress_bar:
            #print(batch_data)#TODO remove
            #print(batch_data)
            #print(' '.join([hex(b)[2:].ljust(2,'0') for b in batch_data[0]]))
            encoded_tokens = xda.encode(' '.join([hex(b)[2:].ljust(2,'0') for b in batch_data[0]]))
            #print(list(zip([hex(b)[2:].ljust(2,'0') for b in batch_data[0]],[a.item() for a in encoded_tokens])))
            # Get model predictions
            #print(encoded_tokens)
            logprobs = xda.predict('funcbound', encoded_tokens)
            #print(logprobs)
            #predictions = logprobs.argmax(dim=2)[0]
            #sample_pred = torch.mode(predictions).values.item()
            #all_preds.append(sample_pred)
            #all_labels.append(batch_labels.flatten()[0].item())
            predictions = logprobs.argmax(dim=2).view(-1).data
            
            all_preds.extend(predictions)
            all_labels.extend(batch_labels.cpu().numpy().flatten())


    print("XDA-Inference complete.")
        
    return all_preds,all_labels

#iterates over each Byte of the input in all the chunks of a XDA-dataset-object
def flat_iter_binaryChunks_binaryBytes(xda_dataset : XDAChunkDataset):
    for row in xda_dataset[:]:
        for element in row[0]:
            yield element.item()
            
            
import hashlib
def hash_iterable_streaming(iterable) -> str:
    
    hasher = hashlib.blake2b()

    for item in iterable:
        hasher.update(repr(item).encode("utf-8"))

    return hasher.hexdigest()
  
import os
import sys
import itertools

def main(datasetPath:Path, resultFilePath:Path):

    os.chdir(Path(os.getcwd()) / "XDA")
    
    datasetPath = Path(os.path.abspath(Path(datasetPath)))
    resultFilePath = Path(os.path.abspath(Path(resultFilePath)))

    
    with open(datasetPath,"rb") as f:
        datasetInfo = pickle.load(f)
        dataset = XDAChunkDataset(datasetInfo[0], datasetInfo[1], datasetInfo[2])

    
    datasetHash = hash_iterable_streaming(flat_iter_binaryChunks_binaryBytes(dataset[:]))
    resultFilename = str(datasetHash) + ".dataset"

    #create result file path for this input-dataset
    resultPath = resultFilePath.parent / resultFilename
    
    # avoid unnecessary inference if already done
    if not resultPath.is_file():
        try:
            all_preds,all_labels = infer_XDA(dataset)
            all_preds = [a.item() for a in all_preds]
        except Exception as e:
            raise RuntimeError("Failed to infer dataset with xda: ") from e
        #save to file
        with open(resultPath,"wb") as resultFile:
            pickle.dump([all_labels,all_preds], resultFile, protocol=pickle.HIGHEST_PROTOCOL)
            
            #avoid datasets beeing added into git repository
            with open(".gitignore","a") as f:
                f.write(resultFilename)
        

        
    #create hardlink to the actual result file
    if resultFilePath.is_file():
        os.unlink(resultFilePath)
    os.link(resultPath, resultFilePath)
    
    return 0
    
    
if __name__ == "__main__":
    path1 = os.path.abspath(sys.argv[1])
    path2 = os.path.abspath(sys.argv[2])
    #os.chdir(os.path.dirname(Path(sys.argv[0])))
    #TODO: for testing:
    #path1 = Path("dataset.info")
    #path2 = Path("result.inferred")
    #------------------
    sys.exit(main(path1,path2))
    
    
    
    
    