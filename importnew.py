import os
import lzma
import PyPDF2
from tqdm import tqdm

def file_extract(directory):
    files = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and (filename.endswith(".xz") or filename.endswith(".pdf")):
            files.append(filename)
    return files

def process_file(file_path):
    """Extract text from either XZ or PDF file"""
    if file_path.endswith(".xz"):
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            return infile.read()
    elif file_path.endswith(".pdf"):
        text = ""
        try:
            with open(file_path, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error processing PDF file {file_path}: {e}")
            return ""
    return ""

folder_path = "C:/Users/Dell/Documents/Assignment2012/LLM project/PDF files"
train_output_file = "train_split.txt"
val_output_file = "val_split.txt"
vocab_files = "vocab.txt"
split_files = int(input("How many files you want to split each set into? "))  # number of files for train and val sets

files = file_extract(folder_path)
total_files = len(files)

# Split into train (90%) and validation (10%) sets
split_index = int(total_files * 0.9)
files_train = files[:split_index]
files_val = files[split_index:]

vocab = set()

# Process training files
train_max_count = len(files_train) // split_files if split_files != 0 else len(files_train)
for i in range(split_files):
    with open(train_output_file.format(i), "w", encoding="utf-8") as out_file:
        batch = files_train[i * train_max_count:min((i + 1) * train_max_count, len(files_train))]
        for filename in tqdm(batch, desc=f"Processing train batch {i+1}/{split_files}"):
            file_path = os.path.join(folder_path, filename)
            text = process_file(file_path)
            out_file.write(text)
            characters = set(text)
            vocab.update(characters)

# Process validation files
val_max_count = len(files_val) // split_files if split_files != 0 else len(files_val)
for i in range(split_files):
    with open(val_output_file.format(i), "w", encoding="utf-8") as out_file:
        batch = files_val[i * val_max_count:min((i + 1) * val_max_count, len(files_val))]
        for filename in tqdm(batch, desc=f"Processing val batch {i+1}/{split_files}"):
            file_path = os.path.join(folder_path, filename)
            text = process_file(file_path)
            out_file.write(text)
            characters = set(text)
            vocab.update(characters)

# Write vocabulary
with open(vocab_files, "w", encoding='utf-8') as vfile:
    for char in vocab:
        vfile.write(char + '\n')

print(f"Processing complete. Created {split_files} training files and {split_files} validation files.")
print(f"Training files: {len(files_train)} files")
print(f"Validation files: {len(files_val)} files")