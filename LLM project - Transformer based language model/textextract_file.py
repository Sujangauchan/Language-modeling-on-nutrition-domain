import os
import lzma
from tqdm import tqdm

def file_extract(directory):
   files= []
   for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)):
           files.append(filename)
   return files

folder_path="/Users/sujangauchan/Desktop/openwebtext"
output_file="output{}.txt"
vocab_files="vocab.txt"
split_files= int(input("How many files you want to split this into?")) # determines the no of files the final txt files are divided into


files = file_extract(folder_path)

total_files = len(files)

max_count = total_files // split_files if split_files != 0 else total_files

vocab = set()

for i in range(split_files):
   with open(output_file.format(i),"w", encoding="utf-8") as output_file:
      for count, filename in enumerate(tqdm(files[:max_count], total=max_count)):
         if count >= max_count:
            break
         file_path = os.path.join(folder_path, filename)
         with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
            output_file.write(text)
            characters = set(text) 
            vocab.update(characters)
   files = files[max_count:]
   
with open(vocab_files,"w", encoding='utf-8') as vfile:
   for char in vocab:
      vfile.write(char +'\n')
