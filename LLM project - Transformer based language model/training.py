import torch ## torch let's us create tensors and also provides helper functions
import torch.nn as nn ## torch.nn gives us nn.module() and nn.Linear()
from torch.nn import functional as F # This gives us the softmax()
import mmap
import random
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re 
import os 

parser = argparse.ArgumentParser(description = 'Transformer Language Model Training')

train_losses_history = []
val_losses_history = []
iterations_history = []

# Hyperparameters 
block_size = 64        # Sequence length in words.
batch_size = 32        # Number of sequences in a batch.
max_iters = 3000       # Total training iterations.
learning_rate = 3e-4   # Learning rate.
eval_iters = 100       # How often to evaluate.
n_embd = 384           # Embedding dimension.
n_head = 8             # Number of attention heads.
n_layer = 8            # Number of transformer layers.
dropout = 0.1          # Dropout rate.

AVG_CHARS_PER_WORD_ESTIMATE = 6 
CHUNK_SIZE_WORDS = batch_size * block_size * 2 

device= 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Word Tokenization and Vocabulary Setup

train_file_path = 'C:/Users/Dell/Documents/Assignment2012/LLM project - V3 traned word level - Copy/train_split.txt'
val_file_path = 'C:/Users/Dell/Documents/Assignment2012/LLM project - V3 traned word level - Copy/val_split.txt'

# Files for storing/loading pre-processed vocabulary
word_to_int_file = 'word_to_int.pkl'
int_to_word_file = 'int_to_word.pkl'
vocab_list_file = 'vocab_list.txt' 

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>' 

try:
    print(f"Loading pre-built vocabulary from {word_to_int_file}, {int_to_word_file}, and {vocab_list_file}...")
    with open(word_to_int_file, 'rb') as f:
        word_to_int = pickle.load(f)
    with open(int_to_word_file, 'rb') as f:
        int_to_word = pickle.load(f)
    vocab_size = len(word_to_int)
    if vocab_size <= 2: 
        raise FileNotFoundError 
    print(f"Vocabulary loaded. Size: {vocab_size}")
except FileNotFoundError:
    print(f"Vocabulary files not found. Building vocabulary from corpus files:\n  Train: {train_file_path}\n  Validation: {val_file_path}")
    print("WARNING: Reading the entire train and validation files into memory for vocabulary building.")
    
    total_corpus_size_gb = 0
    try:
        train_size_gb = os.path.getsize(train_file_path)/(1024**3)
        val_size_gb = os.path.getsize(val_file_path)/(1024**3)
        total_corpus_size_gb = train_size_gb + val_size_gb
        print(f"  Train file size: {train_size_gb:.2f}GB\n  Validation file size: {val_size_gb:.2f}GB\n  Total estimated corpus size: {total_corpus_size_gb:.2f}GB.")
    except FileNotFoundError:
        print(f"Error: One or both data files not found. Please check paths:\n  Train: {train_file_path}\n  Validation: {val_file_path}")
        print("Cannot determine corpus size or build vocabulary.")
        exit()
        

    all_tokens_for_vocab = []
    for current_file_path in [train_file_path, val_file_path]:
        print(f"  Processing for vocabulary: {current_file_path}...")
        try:
            with open(current_file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            file_tokens = re.findall(r'\w+|[^\s\w]+', text_content.lower())
            all_tokens_for_vocab.extend(file_tokens)
            del text_content 
            del file_tokens  
            print(f"    Successfully read and tokenized {current_file_path}.")
        except MemoryError:
            print(f"MemoryError: Could not load the entire file {current_file_path} into RAM for vocabulary building.")
            print("Vocabulary building failed. Please use a memory-efficient approach for large files.")
            exit()
        except FileNotFoundError:
            print(f"Error: File not found: {current_file_path}.")
            print("Vocabulary building failed.")
            exit()
            
    if not all_tokens_for_vocab:
        print("Error: No tokens were extracted from the corpus files. Vocabulary cannot be built.")
        exit()

    print("  Creating unique word list for vocabulary...")
    unique_words = sorted(list(set(all_tokens_for_vocab)))
    del all_tokens_for_vocab # Attempt to free memory

    final_words_list = []
    if PAD_TOKEN not in unique_words:
        final_words_list.append(PAD_TOKEN)
    if UNK_TOKEN not in unique_words:
        final_words_list.append(UNK_TOKEN)
    
    for word in unique_words:
        if word not in [PAD_TOKEN, UNK_TOKEN]: # Avoid duplicates if they were somehow in unique_words
            final_words_list.append(word)
    del unique_words

    vocab_size = len(final_words_list)
    print(f"Vocabulary built. Size: {vocab_size}")

    word_to_int = { word:i for i,word in enumerate(final_words_list)}
    int_to_word = { i:word for i,word in enumerate(final_words_list)}

    print("Saving vocabulary...")
    with open(word_to_int_file, 'wb') as f:
        pickle.dump(word_to_int, f)
    with open(int_to_word_file, 'wb') as f:
        pickle.dump(int_to_word, f)
    with open(vocab_list_file, 'w', encoding='utf-8') as f:
        for word in final_words_list:
            f.write(word + '\n')
    print(f"Vocabulary saved to {word_to_int_file}, {int_to_word_file}, and {vocab_list_file}")


def encode(text_string):
    tokens = re.findall(r'\w+|[^\s\w]+', text_string.lower())
    return [word_to_int.get(token, word_to_int.get(UNK_TOKEN)) for token in tokens] 

def decode(list_of_ints):
    return ' '.join([int_to_word.get(i, UNK_TOKEN) for i in list_of_ints])

def get_random_chunk(split):
    if split == 'train':
        filename = train_file_path 
    else:
        filename = val_file_path   
        
    num_bytes_to_read = CHUNK_SIZE_WORDS * AVG_CHARS_PER_WORD_ESTIMATE

    try:
        with open(filename, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                file_size = len(mm)
                if file_size == 0:
                    return torch.full((CHUNK_SIZE_WORDS,), word_to_int[PAD_TOKEN], dtype=torch.long)
                actual_bytes_to_read = min(num_bytes_to_read, file_size)
                start_pos = 0
                if file_size > actual_bytes_to_read:
                    start_pos = random.randint(0, file_size - actual_bytes_to_read)
                mm.seek(start_pos)
                block_bytes = mm.read(actual_bytes_to_read)
                decoded_block_text = block_bytes.decode('utf-8', errors='ignore').replace('\r', '')
                tokens_from_chunk = re.findall(r'\w+|[^\s\w]+', decoded_block_text.lower())
                encoded_tokens = [word_to_int.get(token, word_to_int[UNK_TOKEN]) for token in tokens_from_chunk]
                if len(encoded_tokens) < CHUNK_SIZE_WORDS:
                    padding_needed = CHUNK_SIZE_WORDS - len(encoded_tokens)
                    encoded_tokens.extend([word_to_int[PAD_TOKEN]] * padding_needed)
                else:
                    encoded_tokens = encoded_tokens[:CHUNK_SIZE_WORDS]
                data = torch.tensor(encoded_tokens, dtype=torch.long)
        return data
    except FileNotFoundError:
        print(f"Error: Data file {filename} not found during get_random_chunk. Returning PAD tokens.")
        return torch.full((CHUNK_SIZE_WORDS,), word_to_int[PAD_TOKEN], dtype=torch.long)
    except Exception as e:
        print(f"An error occurred in get_random_chunk with file {filename}: {e}")
        return torch.full((CHUNK_SIZE_WORDS,), word_to_int[PAD_TOKEN], dtype=torch.long)


def get_batch(split):
    data = get_random_chunk(split) 
    if len(data) < block_size + 1:
        # print(f"Warning: Data chunk too small ({len(data)} tokens) for block_size {block_size}. Returning PAD batch.")
        x = torch.full((batch_size, block_size), word_to_int[PAD_TOKEN], dtype=torch.long, device=device)
        y = torch.full((batch_size, block_size), word_to_int[PAD_TOKEN], dtype=torch.long, device=device)
        return x, y
    max_start_idx = len(data) - block_size - 1 
    if max_start_idx < 0 :
        # print(f"Warning: Not enough data ({len(data)} tokens) to form even one sequence of length {block_size}. Returning PAD batch.")
        x = torch.full((batch_size, block_size), word_to_int[PAD_TOKEN], dtype=torch.long, device=device)
        y = torch.full((batch_size, block_size), word_to_int[PAD_TOKEN], dtype=torch.long, device=device)
        return x,y
    ix = torch.randint(0, max_start_idx + 1, (batch_size,))
    x_list = [data[i : i + block_size] for i in ix]
    y_list = [data[i + 1 : i + block_size + 1] for i in ix]
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            if loss is not None :
                 losses[k]= loss.item()
            else: # If loss is None (e.g., all targets were PAD tokens and ignored)
                 losses[k] = float('nan') # Use NaN to indicate an issue or ignorable batch
        # Calculate mean ignoring NaNs if any
        valid_losses = losses[~torch.isnan(losses)]
        if len(valid_losses) > 0:
            out[split] = valid_losses.mean()
        else: # All batches resulted in NaN loss (e.g. all PADs)
            out[split] = torch.tensor(float('nan')) 
    model.train()
    return out
    
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape 
        k = self.key(x) 
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        v = self.value(x) 
        out = wei @ v 
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd) 
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
        
class FeedForward(nn.Module) :
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4* n_embd, n_embd), 
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return  self.net(x)

class Block(nn.Module): 
    def __init__(self, n_embd, n_head):
        super().__init__()
        if n_embd % n_head != 0: # Check for divisibility
            raise ValueError(f"Embedding dimension n_embd ({n_embd}) must be divisible by number of heads n_head ({n_head}).")
        head_size = n_embd // n_head 
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class TransformerLanguageModel (nn.Module):
    def __init__(self, vocab_size_param):
        super().__init__()
        self.padding_token_id = word_to_int.get(PAD_TOKEN)
        if self.padding_token_id is None:
            self.padding_token_id = 0 
            
        self.token_embedding_table = nn.Embedding(vocab_size_param, n_embd, padding_idx=self.padding_token_id)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) 
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) 
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size_param)
        self.apply(self._init_weights)   

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean =0.0, std=0.02)
            if module.padding_idx is not None: 
                 with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def forward(self, index, targets=None):
        B, T = index.shape 
        tok_emb = self.token_embedding_table(index) 
        pos_emb = self.position_embedding_table(torch.arange(T, device= index.device)) # pos_emb for (0, ..., T-1)
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 
        loss = None
        if targets is not None:
            B_logits, T_logits, C_logits = logits.shape
            logits_reshaped = logits.view(B_logits*T_logits, C_logits)
            targets_reshaped = targets.view(B_logits*T_logits)
            # Use the padding_idx obtained during model initialization
            loss = F.cross_entropy(logits_reshaped, targets_reshaped, ignore_index=self.padding_token_id)
        return logits, loss
        
    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop index to the last block_size tokens to ensure it fits position_embedding_table
            index_cond = index[:, -block_size:]
            logits, _ = self.forward(index_cond) # Loss is not needed for generation
            logits = logits[:, -1, :] # Focus only on the last time step -> (B, C)
            probs = F.softmax(logits, dim=-1) # Apply softmax to get probabilities
            index_next = torch.multinomial(probs, num_samples=1) # Sample from the distribution (B, 1)
            index = torch.cat((index, index_next), dim=1) # Append sampled index (B, T+1)
        return index

def plot_training_history():
    if not iterations_history: # Check if there's anything to plot
        print("No training history recorded (iterations_history is empty). Skipping plot.")
        return

    plt.figure(figsize=(12, 7))
    plt.plot(iterations_history, train_losses_history, 'b-', label='Training Loss', alpha=0.8)
    plt.plot(iterations_history, val_losses_history, 'r-', label='Validation Loss', alpha=0.8)
    plt.title('Word-Level Transformer Language Model Training', fontsize=16)
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.minorticks_on()
    param_text = (f"Hyperparameters:\n"
                  f"Tokenizer: Word-level (Regex, Lowercase)\n"
                  f"Block size (seq_len): {block_size} words\n"
                  f"Batch size: {batch_size}\n"
                  f"Learning rate: {learning_rate}\n"
                  f"Embedding dim: {n_embd}\n"
                  f"Heads: {n_head}\nLayers: {n_layer}\n"
                  f"Dropout: {dropout}\n"
                  f"Vocab size: {vocab_size}\nDevice: {device.upper()}")
    plt.figtext(0.01, 0.01, param_text, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
    plt.tight_layout(rect=[0.12, 0.12, 0.98, 0.95]) 
    plt.savefig('word_language_model_training.png', dpi=300) 
    plt.savefig('word_language_model_training.jpg', dpi=300) 
    plt.close()
    print("Training visualization saved as 'word_language_model_training.png/jpg'")

# Main Training Loop

model = TransformerLanguageModel(vocab_size).to(device) # Pass vocab_size to model
total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params:,}")
optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate)

print(f"Starting training for {max_iters} iterations...")
for iter_num in range(max_iters):
    if iter_num % eval_iters == 0 or iter_num == max_iters -1 :
        losses = estimate_loss()
        train_loss_val = losses['train'].item() if not torch.isnan(losses['train']) else float('nan')
        val_loss_val = losses['val'].item() if not torch.isnan(losses['val']) else float('nan')
        print(f"step: {iter_num}, train loss: {train_loss_val:.4f}, val loss: {val_loss_val:.4f}")
        
        if not np.isnan(train_loss_val):
            train_losses_history.append(train_loss_val)
            
        if not np.isnan(val_loss_val):
            val_losses_history.append(val_loss_val)
        # else:
        #     val_losses_history.append(float('nan'))
        if not np.isnan(train_loss_val) and not np.isnan(val_loss_val) : 
             iterations_history.append(iter_num)


    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    if loss is not None: 
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iter_num > 0 and iter_num % (max_iters//20) == 0 : # Print current training loss periodically
             print(f"  Iteration {iter_num}: current batch train loss {loss.item():.4f}")
    else:
        print(f"Warning: Skipping backward pass for iteration {iter_num} due to None loss (possibly all PAD batch).")


final_loss_value = loss.item() if loss is not None and hasattr(loss, 'item') else float('nan')
print(f"Training finished. Final batch loss: {final_loss_value:.4f}")

model_save_path = 'model-v1.pkl' 
with open(model_save_path, 'wb') as f:
    pickle.dump(model, f) 
print(f"Model saved to {model_save_path}.")

plot_training_history()